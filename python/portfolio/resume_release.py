#!/usr/bin/env python3
"""Select missing ORIGINAL 0.2.0 artifacts only after a read-only PyPI precheck.

No rebuilding, uploading, credentials, or skip-existing behavior. All artifacts
must match the frozen successful CI inventory, including already published files.
Runs in an owned CI workspace. POSIX write-mode checks do not model Windows ACLs;
Windows relies on the runner workspace ACL boundary. This is not a sandbox against
concurrent malicious processes or untrusted Windows junctions on older Python.
"""

from __future__ import annotations

import argparse
import email
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import urllib.error
import urllib.request
import zipfile

VERSION = "0.2.0"
RUNTIME = "0.1.3"
SOURCE_SHA = "fccf36e9a76d35bce6750dfd3b25802cae3d683b"
SOURCE_RUN = 38169271248
SOURCE_ARTIFACT = 11713787141
ARCHIVE_SHA256 = "b4c92cae86c753305abef7aad395fdd9d91c7e84c3c887342defa7a3e4607241"
PACKAGES = tuple(
    "ruvector-" + name
    for name in (
        "attention",
        "cluster",
        "core",
        "gnn",
        "graph",
        "rerank",
        "router",
        "ruvllm",
        "rvf",
        "rvf-wasm",
        "sona",
    )
)
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_MEMBERS = 256
MAX_EXPANDED_BYTES = 16 * 1024 * 1024
MAX_METADATA_BYTES = 256 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
# Filled from the immutable CI artifact, never from the live working tree.
INVENTORY_SHA256 = "ca4c4b47df73a26619c048ed1eb02013bfd96484aeb88e20fe16464ef1e0a32d"
DEFAULT_INVENTORY = Path(__file__).with_name("frozen-release-0.2.0.json")


class GateError(ValueError):
    """Sanitized permanent validation failure."""


class RegistryError(RuntimeError):
    """Unavailable registry; retry only after fresh read-only inspection."""


def expected_files() -> dict[str, str]:
    return {
        filename: package
        for package in PACKAGES
        for filename in (
            f"{package.replace('-', '_')}-{VERSION}-py3-none-any.whl",
            f"{package.replace('-', '_')}-{VERSION}.tar.gz",
        )
    }


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def safe_path(path: Path) -> None:
    """Refuse symlink components; callers must own their working directories."""
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise GateError("symlink path component")


def read_regular(path: Path) -> bytes:
    safe_path(path)
    try:
        preliminary = path.lstat()
    except OSError as exc:
        raise GateError("artifact unavailable") from exc
    if not stat.S_ISREG(preliminary.st_mode):
        raise GateError("artifact must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise GateError("artifact cannot be opened safely") from exc
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size <= 0
            or info.st_size > MAX_FILE_BYTES
        ):
            raise GateError("artifact type, links, or size exceeds bounds")
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) != info.st_size or len(data) > MAX_FILE_BYTES:
        raise GateError("artifact changed or exceeds bounds")
    return data


def member_path(name: str, prefix: str, *, gitignore: bool = False) -> None:
    if not isinstance(name, str) or "\\" in name or "\x00" in name:
        raise GateError("unsafe archive path")
    parts = PurePosixPath(name).parts
    if not name.startswith(prefix) or name.startswith("/") or ".." in parts:
        raise GateError("unexpected archive path")
    if any(
        part in {"__pycache__", "node_modules", "target"}
        or (part.startswith(".") and not (gitignore and name == prefix + ".gitignore"))
        for part in parts
    ):
        raise GateError("forbidden archive content")
    if PurePosixPath(name).suffix in {".pem", ".key", ".so", ".pyd"}:
        raise GateError("forbidden archive content")


def metadata(data: bytes, package: str) -> None:
    if len(data) > MAX_METADATA_BYTES:
        raise GateError("metadata exceeds bounds")
    parsed = email.message_from_bytes(data)
    if parsed.get_all("Name") != [package] or parsed.get_all("Version") != [VERSION]:
        raise GateError("distribution metadata identity mismatch")
    runtime_requirements = [
        value
        for value in parsed.get_all("Requires-Dist", [])
        if re.match(r"^ruvector(?:\s|\(|[<>=!~;\[]|$)", value, re.I)
    ]
    if len(runtime_requirements) != 1 or not re.fullmatch(
        r"ruvector\s*(?:\(\s*==\s*0\.1\.3\s*\)|==\s*0\.1\.3)\s*",
        runtime_requirements[0],
        re.I,
    ):
        raise GateError("distribution runtime must be exactly ruvector==0.1.3")


def validate_archive(path: Path, package: str) -> None:
    normalized = package.replace("-", "_")
    if path.name.endswith(".whl"):
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > MAX_MEMBERS or len({m.filename for m in members}) != len(
                members
            ):
                raise GateError("archive has too many or duplicate members")
            if sum(m.file_size for m in members) > MAX_EXPANDED_BYTES:
                raise GateError("archive expanded size exceeds bounds")
            found = []
            for member in members:
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode) or member.flag_bits & 1:
                    raise GateError("wheel symlink or encrypted content")
                prefix = normalized + "/"
                if member.filename.startswith(f"{normalized}-{VERSION}.dist-info/"):
                    prefix = f"{normalized}-{VERSION}.dist-info/"
                member_path(member.filename, prefix)
                if member.filename.endswith(".dist-info/METADATA"):
                    if member.file_size > MAX_METADATA_BYTES:
                        raise GateError("metadata exceeds bounds")
                    found.append(archive.read(member))
            if len(found) != 1:
                raise GateError("missing or ambiguous wheel metadata")
            metadata(found[0], package)
    else:
        with tarfile.open(path, mode="r:gz") as archive:
            count, expanded, found = 0, 0, []
            prefix = f"{normalized}-{VERSION}/"
            allowed = {
                "src",
                "README.md",
                "LICENSE",
                "pyproject.toml",
                "PKG-INFO",
                ".gitignore",
            }
            seen = set()
            for member in archive:
                count += 1
                expanded += member.size
                if count > MAX_MEMBERS or expanded > MAX_EXPANDED_BYTES:
                    raise GateError("sdist allocation exceeds bounds")
                if member.name in seen or not (member.isfile() or member.isdir()):
                    raise GateError("duplicate or unsafe sdist member")
                seen.add(member.name)
                member_path(member.name, prefix, gitignore=True)
                if member.name[len(prefix) :].split("/")[0] not in allowed:
                    raise GateError("unexpected sdist content")
                if member.name == prefix + "PKG-INFO":
                    if member.size > MAX_METADATA_BYTES:
                        raise GateError("metadata exceeds bounds")
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise GateError("missing sdist metadata")
                    found.append(stream.read(MAX_METADATA_BYTES + 1))
            if len(found) != 1:
                raise GateError("missing or ambiguous sdist metadata")
            metadata(found[0], package)


def fetch_json(package: str, version: str) -> dict | None:
    request = urllib.request.Request(
        f"https://pypi.org/pypi/{package}/{version}/json",
        headers={"User-Agent": "RuVector-frozen-release-precheck/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            if response.status != 200:
                raise RegistryError("unexpected registry response")
            data = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise RegistryError(f"registry HTTP {exc.code}; no files selected") from exc
    except OSError as exc:
        raise RegistryError("registry unavailable; no files selected") from exc
    if len(data) > MAX_RESPONSE_BYTES:
        raise RegistryError("registry response exceeds bounds")
    try:
        result = json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise RegistryError("malformed registry response") from exc
    if not isinstance(result, dict):
        raise RegistryError("invalid registry response")
    return result


def registry_files(data: dict | None, package: str, version: str) -> list[dict]:
    if data is None:
        return []
    if not isinstance(data, dict):
        raise GateError("invalid registry metadata")
    info, urls = data.get("info"), data.get("urls")
    if (
        not isinstance(info, dict)
        or not isinstance(info.get("name"), str)
        or canonical(info["name"]) != package
        or info.get("version") != version
        or not isinstance(urls, list)
        or not urls
        or len(urls) > 128
    ):
        raise GateError("registry identity, version, or file inventory mismatch")
    seen = set()
    for item in urls:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("filename"), str)
            or item["filename"] in seen
            or item.get("yanked") is not False
            or not isinstance(item.get("digests"), dict)
            or not isinstance(item["digests"].get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", item["digests"]["sha256"])
        ):
            raise GateError("registry duplicate, yanked, or invalid file")
        seen.add(item["filename"])
    return urls


def load_inventory(path: Path) -> dict[str, str]:
    data = read_regular(path)
    if hashlib.sha256(data).hexdigest() != INVENTORY_SHA256:
        raise GateError("frozen inventory hash mismatch")
    try:
        inventory = json.loads(data)
    except ValueError as exc:
        raise GateError("invalid frozen inventory") from exc
    required = {
        "version": VERSION,
        "runtime": RUNTIME,
        "source_sha": SOURCE_SHA,
        "run_id": SOURCE_RUN,
        "artifact_id": SOURCE_ARTIFACT,
        "archive_sha256": ARCHIVE_SHA256,
    }
    if not isinstance(inventory, dict) or any(
        inventory.get(k) != v for k, v in required.items()
    ):
        raise GateError("frozen inventory source identity mismatch")
    files = inventory.get("files")
    if not isinstance(files, list) or len(files) != 22:
        raise GateError("frozen inventory file count mismatch")
    result = {}
    for item in files:
        if (
            not isinstance(item, dict)
            or item.get("filename") not in expected_files()
            or item["filename"] in result
            or not isinstance(item.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
            or type(item.get("bytes")) is not int
            or not 0 < item["bytes"] <= MAX_FILE_BYTES
        ):
            raise GateError("invalid frozen inventory entry")
        result[item["filename"]] = item["sha256"]
    if (
        set(result) != set(expected_files())
        or sum(item["bytes"] for item in files) > MAX_TOTAL_BYTES
    ):
        raise GateError("frozen inventory file set or size mismatch")
    return result


def directory_mode_safe(mode: int, platform: str = os.name) -> bool:
    """POSIX mode checks; Windows trusts the CI workspace ACL boundary."""
    return platform == "nt" or not (mode & 0o022)


def precheck(
    directory: Path, *, inventory=DEFAULT_INVENTORY, fetcher=fetch_json
) -> tuple[dict, dict[str, bytes]]:
    safe_path(directory)
    if not directory.is_dir() or not directory_mode_safe(directory.stat().st_mode):
        raise GateError("artifact directory must be private to its owner")
    expected = expected_files()
    frozen = load_inventory(inventory)
    if {p.name for p in directory.iterdir()} != set(expected):
        raise GateError("missing or unexpected release artifacts")
    contents = {}
    total = 0
    for filename, package in sorted(expected.items()):
        data = read_regular(directory / filename)
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise GateError("total artifact size exceeds bounds")
        digest = hashlib.sha256(data).hexdigest()
        if digest != frozen[filename]:
            raise GateError(f"frozen artifact hash mismatch: {filename}")
        try:
            validate_archive(directory / filename, package)
        except (OSError, zipfile.BadZipFile, tarfile.TarError, RuntimeError) as exc:
            raise GateError("invalid distribution archive") from exc
        contents[filename] = data
    native = registry_files(fetcher("ruvector", RUNTIME), "ruvector", RUNTIME)
    if not native:
        raise GateError("required native runtime is not published")
    published = []
    for package in PACKAGES:
        files = registry_files(fetcher(package, VERSION), package, VERSION)
        package_expected = {
            name for name, owner in expected.items() if owner == package
        }
        for item in files:
            filename = item["filename"]
            if filename not in package_expected:
                raise GateError("unknown registry artifact")
            if item["digests"]["sha256"] != frozen[filename]:
                raise GateError(f"published artifact hash mismatch: {filename}")
            published.append(filename)
    missing = sorted(set(expected) - set(published))
    plan = {
        "schema": 1,
        "operation": "precheck-only",
        "version": VERSION,
        "runtime": RUNTIME,
        "source_sha": SOURCE_SHA,
        "source_run": SOURCE_RUN,
        "source_artifact": SOURCE_ARTIFACT,
        "archive_sha256": ARCHIVE_SHA256,
        "published": sorted(published),
        "missing": missing,
        "files": {name: frozen[name] for name in sorted(expected)},
        "promotion": "independent-review-and-protected-environment-required",
    }
    return plan, {name: contents[name] for name in missing}


def stage(
    directory: Path, output: Path, *, inventory=DEFAULT_INVENTORY, fetcher=fetch_json
) -> dict:
    """Write a new directory only after every local and registry check passes."""
    safe_path(output)
    if output.exists():
        raise GateError("staging destination already exists")
    plan, files = precheck(directory, inventory=inventory, fetcher=fetcher)
    output.mkdir(mode=0o700)
    try:
        for name, data in files.items():
            with (output / name).open("xb") as stream:
                stream.write(data)
    except BaseException:
        shutil.rmtree(output)
        raise
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", required=True, type=Path)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument(
        "--missing-dir", type=Path, help="new directory for missing distributions only"
    )
    parser.add_argument("--receipt", type=Path, help="new JSON receipt file")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="require every original file published",
    )
    args = parser.parse_args()
    try:
        if args.verify_only and args.missing_dir:
            raise GateError("verify-only cannot select files")
        if args.receipt:
            safe_path(args.receipt)
            if args.receipt.exists():
                raise GateError("receipt destination already exists")
        result = (
            stage(args.dist, args.missing_dir, inventory=args.inventory)
            if args.missing_dir
            else precheck(args.dist, inventory=args.inventory)[0]
        )
        if args.verify_only and result["missing"]:
            raise GateError("original distributions remain unpublished")
        if args.receipt:
            with args.receipt.open("x", encoding="utf-8") as stream:
                json.dump(
                    {"status": "validated", **result}, stream, indent=2, sort_keys=True
                )
                stream.write("\n")
    except (GateError, RegistryError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}))
        return 1
    print(json.dumps({"status": "validated", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
