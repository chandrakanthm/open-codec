"""Model package: the CodecAvatar conditional VAE and its sub-modules."""

from codec_avatars.models.codec_avatar import CodecAvatar, build_model
from codec_avatars.models.decoder import AppearanceDecoder, MeshDecoder, TextureDecoder
from codec_avatars.models.encoder import AppearanceEncoder

__all__ = [
    "CodecAvatar",
    "build_model",
    "AppearanceEncoder",
    "AppearanceDecoder",
    "MeshDecoder",
    "TextureDecoder",
]
