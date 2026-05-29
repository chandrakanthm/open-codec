"""Open-source neural Codec Avatars.

A Deep Appearance Model (conditional VAE) that encodes a face into a compact
latent *code* and decodes it into a view-dependent 3D avatar (mesh geometry +
texture). This is the open, documented lineage behind Meta Reality Labs'
Codec Avatars; it is the architecture you can actually train on public data
(e.g. the Multiface dataset).
"""

__version__ = "0.1.0"

from codec_avatars.config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
