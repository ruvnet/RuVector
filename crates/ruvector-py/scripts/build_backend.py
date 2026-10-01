"""Delegate to maturin, bundling the upstream HNSW patch for portable sdists.

Maturin packages path dependencies but does not rewrite/include Cargo patches.
Only build_sdist needs this shim; wheel/editable hooks are delegated unchanged.
"""

from __future__ import annotations

import io
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import maturin

build_wheel = maturin.build_wheel
build_editable = maturin.build_editable
prepare_metadata_for_build_wheel = maturin.prepare_metadata_for_build_wheel
prepare_metadata_for_build_editable = maturin.prepare_metadata_for_build_editable
get_requires_for_build_wheel = maturin.get_requires_for_build_wheel
get_requires_for_build_editable = maturin.get_requires_for_build_editable
get_requires_for_build_sdist = maturin.get_requires_for_build_sdist


def build_sdist(
    sdist_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    filename = maturin.build_sdist(sdist_directory, config_settings)
    archive = Path(sdist_directory) / filename
    project = Path(__file__).resolve().parents[1]
    patch = (project / "../../patches/hnsw_rs").resolve()
    # In an unpacked sdist the patch sits alongside the Cargo path dependencies.
    if not patch.is_dir():
        patch = project / "hnsw_rs"
    # A previously built sdist may already carry the normalized local patch.
    if not patch.is_dir():
        with tarfile.open(archive, "r:gz") as source:
            if any(m.name.endswith("/hnsw_rs/Cargo.toml") for m in source.getmembers()):
                return filename
        raise RuntimeError("the upstream HNSW patch is missing")

    fd, temporary = tempfile.mkstemp(prefix=".sdist-", dir=archive.parent)
    os.close(fd)
    staged = Path(temporary)
    try:
        normalized = False
        with tarfile.open(archive, "r:gz") as source, tarfile.open(staged, "w:gz") as target:
            entries = source.getmembers()
            root = entries[0].name.split("/", 1)[0]
            for entry in entries:
                content = source.extractfile(entry) if entry.isfile() else None
                if entry.name.endswith("/ruvector-py/Cargo.toml"):
                    assert content is not None
                    manifest = content.read().decode()
                    old = 'path = "../../patches/hnsw_rs"'
                    if old not in manifest and 'path = "../hnsw_rs"' not in manifest:
                        raise RuntimeError("unexpected HNSW patch path in maturin source archive")
                    replacement = manifest.replace(old, 'path = "../hnsw_rs"').encode()
                    entry.size = len(replacement)
                    content = io.BytesIO(replacement)
                    normalized = True
                target.addfile(entry, content)
            if not normalized:
                raise RuntimeError("maturin source archive omitted the binding manifest")
            for path in sorted(patch.rglob("*")):
                relative = path.relative_to(patch)
                if path.is_file() and not any(
                    part in {"target", ".git", "__pycache__"} for part in relative.parts
                ):
                    target.add(path, arcname=f"{root}/hnsw_rs/{relative.as_posix()}")
        staged.replace(archive)
    finally:
        staged.unlink(missing_ok=True)
    return filename
