"""Loader for Meta's public **Multiface** dataset.

Multiface (https://github.com/facebookresearch/multiface) is the open multi-view
face-capture dataset behind the Deep Appearance / Codec Avatar line. This loader
mirrors the layout used by Meta's own ``dataset.py`` so that, after an expensive
download, training "just works" without surprises. Expected per-subject layout
(directory names are auto-detected across releases)::

    <root>/<subject>/
        tracked_mesh/<EXPR>/<frame>.bin            # float32 vertices (V*3,)
        tracked_mesh/<EXPR>/<frame>.obj            # shared topology (faces, UVs) [optional]
        tracked_mesh/<EXPR>/<frame>_transform.txt  # per-frame head pose (3x4 / 4x4) [optional]
        unwrapped_uv_1024/<EXPR>/average/<frame>.png  # view-averaged texture (SHIPPED)
        unwrapped_uv_1024/<EXPR>/<CAM>/<frame>.png    # per-camera unwrapped texture
        vert_mean.bin                              # shipped geometry mean (V*3,) [optional]
        vert_var.txt                               # shipped geometry variance (scalar) [optional]
        KRT                                        # per-camera intrinsics + extrinsics

A training sample is a (subject, expression, frame, camera) tuple: the decoder
target is that camera's unwrapped texture conditioned on its view direction; the
encoder input is the view-averaged texture for the same (expression, frame).

This loader follows Meta's ``dataset.py`` on four points that matter for fidelity,
each with a graceful fallback when the shipped file is missing:

1. **View-averaged texture** is read from the shipped ``average/`` directory
   (fallback: average the per-camera textures and cache to ``<tex>_avg/``).
2. **Geometry mean/std** come from shipped ``vert_mean.bin`` + ``vert_var.txt``
   (fallback: compute from a subsample of meshes and cache).
3. **View direction** uses the per-frame head pose ``<frame>_transform.txt`` to
   place the camera in the head-normalized frame the tracked mesh lives in
   (fallback: world-space head-centroid -> camera direction).
4. **Valid-texel mask** marks empty UV regions (texel == 0) so they are excluded
   from the texture loss (fallback: all-ones).

Unlike Meta's ``dataset.py`` we keep textures in **[0, 1]** (the decoder ends in a
sigmoid) rather than mean/std-normalizing them, and we do **not** vertically flip
the UV image: the encoder input (``average/``) and the decoder target (per-camera)
are read in the same orientation, so absolute UV flip is irrelevant to training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from codec_avatars.data.transforms import camera_center, load_texture, view_direction
from codec_avatars.utils.mesh import load_obj, read_bin_vertices

_TEX_DIR_CANDIDATES = ("unwrapped_uv_1024", "unwrapped_uv", "unwrapped_uv_512", "unwrapped_texture")
_MESH_EXTS = (".bin", ".obj", ".ply")
_AVG_DIRNAME = "average"  # shipped view-averaged texture lives here; never a camera


def parse_krt(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Parse a Multiface ``KRT`` file -> {camera: {K(3,3), R(3,3), T(3)}}.

    Each camera block is: a name line, a 3x3 intrinsic matrix, a distortion row
    (unused), a 3x4 extrinsic matrix, then a blank separator. Blank lines are
    stripped first, so blocks are 8 non-blank lines apart.
    """
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    cams: dict[str, dict[str, np.ndarray]] = {}
    j = 0
    while j < len(lines):
        toks = lines[j].split()
        if len(toks) != 1:  # not a camera-id line; resync
            j += 1
            continue
        try:
            cam = toks[0]
            K = np.array([[float(x) for x in lines[j + r].split()[:3]] for r in range(1, 4)], dtype=np.float64)
            # lines[j+4] is the 5-coeff distortion row (unused here)
            rt = np.array([[float(x) for x in lines[j + r].split()[:4]] for r in range(5, 8)], dtype=np.float64)
            cams[cam] = {"K": K, "R": rt[:, :3], "T": rt[:, 3]}
            j += 8
        except (IndexError, ValueError):
            j += 1
    return cams


def _find_dir(parent: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        p = parent / name
        if p.is_dir():
            return p
    return None


def _find_mesh(mesh_root: Path, expr: str, frame: str) -> Path | None:
    for ext in _MESH_EXTS:
        p = mesh_root / expr / f"{frame}{ext}"
        if p.exists():
            return p
    return None


class MultifaceDataset(Dataset):
    def __init__(
        self,
        root: str,
        n_vertices: int,
        tex_size: int,
        tex_channels: int = 3,
        view_dim: int = 3,
        subjects: list[str] | None = None,
        cameras: list[str] | None = None,
        expressions: list[str] | None = None,
        max_frames_per_seq: int | None = None,
        stats_max_meshes: int = 200,
    ):
        self.root = Path(root)
        self.cfg_n_vertices = n_vertices
        self.tex_size = tex_size
        self.tex_channels = tex_channels
        self.view_dim = view_dim
        self.cameras_filter = set(cameras) if cameras else None
        self.expr_filter = set(expressions) if expressions else None
        self.max_frames_per_seq = max_frames_per_seq

        if not self.root.is_dir():
            raise FileNotFoundError(f"Multiface root not found: {self.root}")

        subj_dirs = (
            [self.root / s for s in subjects]
            if subjects
            else sorted(p for p in self.root.iterdir() if p.is_dir())
        )

        self.samples: list[tuple] = []  # (tex_path, mesh_path, transf_path, subject_idx, cam)
        self.subjects: list[Path] = []
        self.krt: list[dict] = []
        self.tex_dirs: list[Path] = []
        self.mesh_dirs: list[Path] = []
        self._topology: dict[int, tuple] = {}
        self._campos_world: dict[tuple[int, str], np.ndarray] = {}

        for sd in subj_dirs:
            tex_dir = _find_dir(sd, _TEX_DIR_CANDIDATES)
            mesh_dir = sd / "tracked_mesh"
            krt_path = sd / "KRT"
            if tex_dir is None or not mesh_dir.is_dir() or not krt_path.exists():
                continue
            si = len(self.subjects)
            krt = parse_krt(krt_path)
            self.subjects.append(sd)
            self.krt.append(krt)
            self.tex_dirs.append(tex_dir)
            self.mesh_dirs.append(mesh_dir)
            # world-space camera centers: c = -R^T t  (Meta: -extrin[:3,:3].T @ extrin[:3,3])
            for cam, p in krt.items():
                self._campos_world[(si, cam)] = camera_center(p["R"], p["T"])
            self._index_subject(si, tex_dir, mesh_dir)

        if not self.samples:
            raise RuntimeError(
                f"No usable samples under {self.root}. Expected per-subject "
                f"tracked_mesh/, one of {_TEX_DIR_CANDIDATES}, and a KRT file."
            )

        self.vertex_count = self._infer_vertex_count()
        self._mean, self._std = self._load_or_compute_stats(stats_max_meshes)

    # --- indexing -----------------------------------------------------------------
    def _index_subject(self, si: int, tex_dir: Path, mesh_dir: Path) -> None:
        krt = self.krt[si]
        for expr_dir in sorted(p for p in tex_dir.iterdir() if p.is_dir()):
            expr = expr_dir.name
            if self.expr_filter and expr not in self.expr_filter:
                continue
            for cam_dir in sorted(p for p in expr_dir.iterdir() if p.is_dir()):
                cam = cam_dir.name
                if cam == _AVG_DIRNAME or cam not in krt:  # skip the shipped average/ dir
                    continue
                if self.cameras_filter and cam not in self.cameras_filter:
                    continue
                frames = sorted(cam_dir.glob("*.png"))
                if self.max_frames_per_seq:
                    step = max(1, len(frames) // self.max_frames_per_seq)
                    frames = frames[::step][: self.max_frames_per_seq]
                for tex_path in frames:
                    frame = tex_path.stem
                    mesh_path = _find_mesh(mesh_dir, expr, frame)
                    if mesh_path is None:
                        continue
                    transf_path = mesh_dir / expr / f"{frame}_transform.txt"
                    self.samples.append((tex_path, mesh_path, transf_path, si, cam))

    def _infer_vertex_count(self) -> int:
        _, mesh_path, _, _, _ = self.samples[0]
        v = self._read_mesh(mesh_path)
        return int(v.shape[0])

    @staticmethod
    def _read_mesh(mesh_path: Path) -> np.ndarray:
        if mesh_path.suffix == ".bin":
            return read_bin_vertices(mesh_path)
        v, _, _, _ = load_obj(mesh_path)
        return v

    # --- geometry statistics ------------------------------------------------------
    def geometry_stats(self) -> tuple[np.ndarray, float]:
        return self._mean, self._std

    @property
    def n_vertices(self) -> int:
        return self.vertex_count

    def _load_shipped_stats(self, subject_dir: Path) -> tuple[np.ndarray, float] | None:
        """Load Meta's shipped ``vert_mean.bin`` (V*3,) + ``vert_var.txt`` (scalar)."""
        mean_path = subject_dir / "vert_mean.bin"
        var_path = subject_dir / "vert_var.txt"
        if not (mean_path.exists() and var_path.exists()):
            return None
        try:
            mean = np.fromfile(str(mean_path), dtype="<f4")
            if mean.size != self.vertex_count * 3:
                return None
            var = float(np.genfromtxt(str(var_path)))
            std = float(np.sqrt(max(var, 0.0))) + 1e-6
            return mean.reshape(self.vertex_count, 3).astype(np.float32), std
        except (ValueError, OSError):
            return None

    def _load_or_compute_stats(self, max_meshes: int) -> tuple[np.ndarray, float]:
        # 1. Shipped per-subject stats (the common single-subject case). These are
        #    exact and match Meta's training; prefer them over anything computed.
        if len(self.subjects) == 1:
            shipped = self._load_shipped_stats(self.subjects[0])
            if shipped is not None:
                return shipped

        # 2. Previously cached computed stats.
        cache = self.root / "codec_geom_stats.npz"
        if cache.exists():
            d = np.load(cache)
            if d["mean"].shape[0] == self.vertex_count:
                return d["mean"].astype(np.float32), float(d["std"])

        # 3. Compute from a subsample of unique meshes across the index.
        seen: dict[str, Path] = {}
        for _, mesh_path, _, _, _ in self.samples:
            seen.setdefault(str(mesh_path), mesh_path)
        paths = list(seen.values())
        if len(paths) > max_meshes:
            idx = np.linspace(0, len(paths) - 1, max_meshes).astype(int)
            paths = [paths[i] for i in idx]
        acc = np.zeros((self.vertex_count, 3), dtype=np.float64)
        cnt = 0
        for p in paths:
            v = self._read_mesh(p).astype(np.float64)
            if v.shape[0] != self.vertex_count:
                continue
            acc += v
            cnt += 1
        mean = (acc / max(cnt, 1)).astype(np.float32)
        sq = 0.0
        for p in paths:
            v = self._read_mesh(p).astype(np.float64)
            if v.shape[0] != self.vertex_count:
                continue
            sq += float(((v - mean) ** 2).sum())
        std = float(np.sqrt(sq / max(cnt * self.vertex_count * 3, 1)) + 1e-6)
        try:
            np.savez(cache, mean=mean, std=np.float32(std))
        except OSError:
            pass
        return mean, std

    # --- view-averaged texture ----------------------------------------------------
    def _avg_texture(self, si: int, expr: str, frame: str) -> np.ndarray:
        """Shipped ``average/<frame>.png`` if present, else compute + cache."""
        tex_dir = self.tex_dirs[si]
        shipped = tex_dir / expr / _AVG_DIRNAME / f"{frame}.png"
        if shipped.exists():
            return load_texture(shipped, self.tex_size, self.tex_channels)

        cache = tex_dir.parent / f"{tex_dir.name}_avg" / expr / f"{frame}.png"
        if cache.exists():
            return load_texture(cache, self.tex_size, self.tex_channels)

        acc = None
        n = 0
        for cam_dir in sorted(p for p in (tex_dir / expr).iterdir() if p.is_dir()):
            if cam_dir.name == _AVG_DIRNAME:
                continue
            f = cam_dir / f"{frame}.png"
            if not f.exists():
                continue
            t = load_texture(f, self.tex_size, self.tex_channels)
            acc = t if acc is None else acc + t
            n += 1
        if acc is None:
            return np.zeros((self.tex_channels, self.tex_size, self.tex_size), np.float32)
        avg = (acc / n).astype(np.float32)
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            from codec_avatars.data.transforms import save_texture

            save_texture(cache, avg)
        except OSError:
            pass
        return avg

    # --- view direction -----------------------------------------------------------
    def _view_direction(self, si: int, cam: str, transf_path: Path, verts: np.ndarray) -> np.ndarray:
        """Camera direction in the head-normalized frame (Meta's convention).

        ``campos_head = R_f^T (campos_world - t_f)`` where (R_f, t_f) is the
        per-frame head pose. Falls back to a world-space head-centroid direction
        when the transform file is absent or malformed.
        """
        params = self.krt[si][cam]
        if transf_path is not None and transf_path.exists():
            try:
                transf = np.genfromtxt(str(transf_path))
                if transf.ndim == 2 and transf.shape[0] >= 3 and transf.shape[1] >= 4:
                    R_f = transf[:3, :3]
                    t_f = transf[:3, 3]
                    campos_head = R_f.T @ (self._campos_world[(si, cam)] - t_f)
                    norm = np.linalg.norm(campos_head)
                    if norm > 1e-8:
                        return (campos_head / norm).astype(np.float32)[: self.view_dim]
            except (ValueError, OSError, IndexError):
                pass
        return view_direction(params["R"], params["T"], verts.mean(axis=0))[: self.view_dim]

    @staticmethod
    def _valid_mask(tex: np.ndarray) -> np.ndarray:
        """Valid-texel mask (1, H, W): 0 in empty UV regions (all channels == 0)."""
        valid = (tex.max(axis=0, keepdims=True) > 0.0).astype(np.float32)
        return valid

    def topology(self, si: int = 0) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (faces, uvs) from a subject's first OBJ, if available (cached)."""
        if si in self._topology:
            return self._topology[si]
        for obj in sorted(self.mesh_dirs[si].rglob("*.obj"))[:1]:
            _, faces, uvs, _ = load_obj(obj)
            self._topology[si] = (faces, uvs)
            return self._topology[si]
        self._topology[si] = None
        return None

    # --- Dataset API --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        tex_path, mesh_path, transf_path, si, cam = self.samples[i]
        expr = tex_path.parent.parent.name
        frame = tex_path.stem

        verts = self._read_mesh(mesh_path).astype(np.float32)
        verts_std = (verts - self._mean) / self._std

        tex = load_texture(tex_path, self.tex_size, self.tex_channels)
        avg_tex = self._avg_texture(si, expr, frame)
        mask = self._valid_mask(tex)

        view = self._view_direction(si, cam, transf_path, verts)

        return {
            "verts": torch.from_numpy(verts_std),
            "avg_tex": torch.from_numpy(avg_tex),
            "tex": torch.from_numpy(tex),
            "view": torch.from_numpy(np.ascontiguousarray(view)),
            "tex_mask": torch.from_numpy(mask),
        }
