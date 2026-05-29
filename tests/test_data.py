import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from codec_avatars.config import Config, DataConfig, ModelConfig
from codec_avatars.data import build_base_dataset, build_datasets
from codec_avatars.data.multiface import MultifaceDataset, parse_krt
from codec_avatars.data.synthetic import SyntheticCodecDataset
from codec_avatars.data.transforms import camera_center, view_direction


def test_synthetic_item_contract():
    ds = SyntheticCodecDataset(n_vertices=50, tex_size=32, size=10, n_identities=2, seed=1)
    item = ds[0]
    assert item["verts"].shape == (50, 3)
    assert item["avg_tex"].shape == (3, 32, 32)
    assert item["tex"].shape == (3, 32, 32)
    assert item["view"].shape == (3,)
    assert item["tex_mask"].shape == (1, 32, 32)
    # textures bounded; view unit-norm
    assert float(item["tex"].min()) >= 0.0 and float(item["tex"].max()) <= 1.0
    assert abs(float(item["view"].norm()) - 1.0) < 1e-4


def test_synthetic_is_deterministic():
    a = SyntheticCodecDataset(n_vertices=20, tex_size=16, size=5, seed=7)[3]
    b = SyntheticCodecDataset(n_vertices=20, tex_size=16, size=5, seed=7)[3]
    assert torch.allclose(a["tex"], b["tex"])
    assert torch.allclose(a["verts"], b["verts"])


def test_geometry_stats_shapes():
    ds = SyntheticCodecDataset(n_vertices=40, tex_size=16, size=8, seed=0)
    mean, std = ds.geometry_stats()
    assert mean.shape == (40, 3) and std > 0


def test_build_datasets_split_and_collate():
    cfg = Config(
        model=ModelConfig(n_vertices=30, tex_size=16, latent_dim=8,
                          enc_tex_channels=(8, 16), dec_tex_channels=(16, 8), mesh_hidden=(16,)),
        data=DataConfig(name="synthetic", synthetic_size=20, val_fraction=0.25),
    )
    train_ds, val_ds, (mean, std), nv = build_datasets(cfg)
    assert nv == 30
    assert len(train_ds) + len(val_ds) == 20
    batch = next(iter(DataLoader(train_ds, batch_size=4)))
    assert batch["avg_tex"].shape == (4, 3, 16, 16)

    base, _, _ = build_base_dataset(cfg)
    assert len(base) == 20


def test_view_direction_math():
    R = np.eye(3)
    T = np.array([0.0, 0.0, -5.0])  # camera center at (0,0,5)
    assert np.allclose(camera_center(R, T), [0, 0, 5])
    d = view_direction(R, T, head_center=np.zeros(3))
    assert np.allclose(d, [0, 0, 1], atol=1e-5)


# --- Multiface loader (on-disk fixture) ------------------------------------------
# The Multiface loader only runs after an expensive download, so exercise every
# shipped-file path + fallback here against a tiny synthetic dataset on disk.

_V = 16  # vertices in the fixture topology
_TS = 8  # texture size
_CAMS = {  # extrinsics chosen so camera centers are on +Z / +X axes
    "400": (np.eye(3), np.array([0.0, 0.0, -1000.0])),  # center (0,0,1000)
    "401": (np.eye(3), np.array([-1000.0, 0.0, 0.0])),  # center (1000,0,0)
}


def _write_krt(path):
    blocks = []
    for cam, (R, T) in _CAMS.items():
        K = np.array([[1000.0, 0, _TS / 2], [0, 1000.0, _TS / 2], [0, 0, 1]])
        rt = np.hstack([R, T.reshape(3, 1)])
        rows = [cam]
        rows += [" ".join(f"{x:.6f}" for x in K[r]) for r in range(3)]
        rows.append("0.0 0.0 0.0 0.0 0.0")  # distortion (unused)
        rows += [" ".join(f"{x:.6f}" for x in rt[r]) for r in range(3)]
        blocks.append("\n".join(rows))
    path.write_text("\n\n".join(blocks) + "\n")


def _write_png(path, value, hole_top_half=False):
    arr = np.full((_TS, _TS, 3), value, dtype=np.uint8)
    if hole_top_half:
        arr[: _TS // 2] = 0  # empty UV region -> must be masked out
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(str(path))


def _make_multiface(root, expr="EXP_neutral", frame="000000", shipped=True):
    subj = root / "S1"
    mesh_dir = subj / "tracked_mesh" / expr
    tex_dir = subj / "unwrapped_uv_1024" / expr
    mesh_dir.mkdir(parents=True)
    _write_krt(subj / "KRT")

    verts = np.linspace(-50, 50, _V * 3).astype("<f4")
    verts.tofile(str(mesh_dir / f"{frame}.bin"))
    # minimal OBJ topology so .topology() has something to return
    (mesh_dir / f"{frame}.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n"
    )

    for cam in _CAMS:
        _write_png(tex_dir / cam / f"{frame}.png", value=80, hole_top_half=True)

    if shipped:
        # per-frame head pose (identity -> view == world camera direction)
        transf = np.hstack([np.eye(3), np.zeros((3, 1))])
        np.savetxt(str(mesh_dir / f"{frame}_transform.txt"), transf)
        # shipped view-averaged texture (distinct constant value)
        _write_png(tex_dir / "average" / f"{frame}.png", value=128)
        # shipped geometry stats
        np.full(_V * 3, 7.0, dtype="<f4").tofile(str(subj / "vert_mean.bin"))
        (subj / "vert_var.txt").write_text("100.0\n")
    return subj


def test_parse_krt_roundtrip(tmp_path):
    _write_krt(tmp_path / "KRT")
    cams = parse_krt(tmp_path / "KRT")
    assert set(cams) == set(_CAMS)
    assert cams["400"]["K"].shape == (3, 3)
    assert cams["400"]["R"].shape == (3, 3)
    assert cams["400"]["T"].shape == (3,)
    assert np.allclose(camera_center(cams["400"]["R"], cams["400"]["T"]), [0, 0, 1000])


def test_multiface_uses_shipped_files(tmp_path):
    _make_multiface(tmp_path, shipped=True)
    ds = MultifaceDataset(root=str(tmp_path), n_vertices=999, tex_size=_TS)

    # vertex count auto-detected from data (not the config value)
    assert ds.n_vertices == _V
    # one frame x two real cameras; the average/ dir is NOT treated as a camera
    assert len(ds) == 2

    # shipped vert_mean.bin (=7) + vert_var.txt (=100 -> std 10) are used verbatim
    mean, std = ds.geometry_stats()
    assert mean.shape == (_V, 3) and np.allclose(mean, 7.0)
    assert abs(std - 10.0) < 1e-3

    item = ds[0]
    assert item["verts"].shape == (_V, 3)
    assert item["avg_tex"].shape == (3, _TS, _TS)
    assert item["tex"].shape == (3, _TS, _TS)
    assert item["view"].shape == (3,)
    assert item["tex_mask"].shape == (1, _TS, _TS)

    # shipped average texture (128/255) is read, not recomputed from cameras (80/255)
    assert abs(float(item["avg_tex"].mean()) - 128 / 255) < 1e-2

    # valid mask: top half is an empty UV hole (0), bottom half valid (1)
    m = item["tex_mask"][0]
    assert float(m[: _TS // 2].max()) == 0.0
    assert float(m[_TS // 2 :].min()) == 1.0

    # transform-based view: identity head pose -> unit world camera direction
    assert abs(float(item["view"].norm()) - 1.0) < 1e-4


def test_multiface_view_matches_transform(tmp_path):
    _make_multiface(tmp_path, shipped=True)
    ds = MultifaceDataset(root=str(tmp_path), n_vertices=_V, tex_size=_TS)
    views = {ds.samples[i][4]: ds[i]["view"].numpy() for i in range(len(ds))}
    # camera 400 center at (0,0,1000) -> +Z ; 401 at (1000,0,0) -> +X
    assert np.allclose(views["400"], [0, 0, 1], atol=1e-4)
    assert np.allclose(views["401"], [1, 0, 0], atol=1e-4)


def test_multiface_fallbacks_when_unshipped(tmp_path):
    # No average/, no vert_mean/var, no transform -> loader must still work.
    _make_multiface(tmp_path, shipped=False)
    ds = MultifaceDataset(root=str(tmp_path), n_vertices=_V, tex_size=_TS)
    assert len(ds) == 2

    # stats computed from meshes (single mesh -> std clamps to the +1e-6 floor)
    mean, std = ds.geometry_stats()
    assert mean.shape == (_V, 3) and std > 0

    item = ds[0]
    # recomputed average: bottom (valid) half == the per-camera value (80/255),
    # top half stays 0 because every camera texture has the hole there
    bottom = item["avg_tex"][:, _TS // 2 :, :]
    assert abs(float(bottom.mean()) - 80 / 255) < 1e-2
    # centroid-fallback view is still a unit vector
    assert abs(float(item["view"].norm()) - 1.0) < 1e-4


def test_multiface_topology(tmp_path):
    _make_multiface(tmp_path, shipped=True)
    ds = MultifaceDataset(root=str(tmp_path), n_vertices=_V, tex_size=_TS)
    topo = ds.topology(0)
    assert topo is not None
    faces, uvs = topo
    assert faces.shape[1] == 3 and uvs.shape[1] == 2
