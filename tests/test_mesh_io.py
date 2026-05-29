import numpy as np

from codec_avatars.data.transforms import load_texture, save_texture
from codec_avatars.utils.mesh import load_obj, read_bin_vertices, save_obj


def test_obj_roundtrip_with_faces_and_uvs(tmp_path):
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    uvs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    p = tmp_path / "m.obj"
    save_obj(p, verts, faces=faces, uvs=uvs, face_uvs=faces, texture_name="t.png")
    assert (tmp_path / "m.mtl").exists()

    v2, f2, vt2, fvt2 = load_obj(p)
    assert np.allclose(v2, verts)
    assert np.array_equal(np.sort(f2, axis=1), np.sort(faces, axis=1))
    assert vt2 is not None and np.allclose(vt2, uvs)


def test_read_bin_vertices(tmp_path):
    arr = np.random.randn(17, 3).astype("<f4")
    p = tmp_path / "v.bin"
    arr.tofile(p)
    out = read_bin_vertices(p)
    assert out.shape == (17, 3)
    assert np.allclose(out, arr)


def test_texture_io_roundtrip(tmp_path):
    tex = np.clip(np.random.rand(3, 32, 32).astype(np.float32), 0, 1)
    p = tmp_path / "t.png"
    save_texture(p, tex)
    back = load_texture(p, size=32, channels=3)
    assert back.shape == (3, 32, 32)
    # PNG is 8-bit, so allow quantization error
    assert np.abs(back - tex).mean() < 0.02
