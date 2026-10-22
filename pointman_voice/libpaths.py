"""Where Linux finds the native libraries the voice models load, set before anything imports them.

The dynamic loader reads LD_LIBRARY_PATH once, when a process starts, so the extension starts itself again
with it set (os.execve keeps its pid, stdin and stdout, so the node sees nothing). On any Linux that's the
CUDA libraries pip put in the environment (cuBLAS, cuDNN, the CUDA runtime), which faster-whisper's
CTranslate2 opens by name. On NixOS also the GPU driver's folder and a libstdc++, which NixOS keeps out of
the usual places, and certifi's certificates for the model fetch.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

DONE = "POINTMAN_VOICE_LIBS"


def _nvidia_libs() -> list[str]:
    site = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    return sorted(str(p) for p in (site / "nvidia").glob("*/lib") if p.is_dir())


def _version(path: Path) -> tuple:
    m = re.search(r"-gcc-(\d+(?:\.\d+)*)-lib$", path.parent.name)
    return tuple(int(x) for x in m.group(1).split(".")) if m else ()


def _libstdcxx(store: Path = Path("/nix/store")) -> list[str]:
    """The newest gcc's lib folder in the Nix store that has libstdc++, unless nix-ld names one already."""
    named = [p for p in os.environ.get("NIX_LD_LIBRARY_PATH", "").split(":") if p]
    if named:
        return named
    found = sorted(store.glob("*-gcc-*-lib/lib"), key=_version)
    found = [p for p in found if (p / "libstdc++.so.6").exists()]
    return [str(found[-1])] if found else []


def wanted(nixos: bool) -> dict[str, str]:
    """The environment to add, or {} when nothing's missing."""
    paths = []
    if nixos:
        paths += [p for p in ["/run/opengl-driver/lib"] if Path(p).is_dir()] + _libstdcxx()
    paths += _nvidia_libs()
    have = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    add = [p for p in paths if p not in have]
    env = {}
    if add:
        env["LD_LIBRARY_PATH"] = ":".join(add + have)
    if nixos and "SSL_CERT_FILE" not in os.environ:
        try:
            import certifi

            env["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            pass
    return env


def ensure() -> None:
    """On Linux, start again once with the libraries' folders on the loader's path."""
    if not sys.platform.startswith("linux") or os.environ.get(DONE):
        return
    env = wanted(Path("/etc/NIXOS").exists())
    if not env:
        return
    os.execve(sys.executable, [sys.executable, "-m", "pointman_voice", *sys.argv[1:]],
              {**os.environ, **env, DONE: "1"})
