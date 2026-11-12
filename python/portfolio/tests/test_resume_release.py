import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import urllib.error
import zipfile

import pytest

SPEC = importlib.util.spec_from_file_location(
    "resume", Path(__file__).parents[1] / "resume_release.py"
)
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def distribution_metadata(package, runtime="0.1.3"):
    return f"Metadata-Version: 2.4\nName: {package}\nVersion: 0.2.0\nRequires-Dist: ruvector=={runtime}\n\n".encode()


def write_wheel(path, package, extra=None, runtime="0.1.3"):
    normalized = package.replace("-", "_")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{normalized}-0.2.0.dist-info/METADATA",
            distribution_metadata(package, runtime),
        )
        archive.writestr(f"{normalized}/__init__.py", "# synthetic fixture\n")
        if extra:
            archive.writestr(extra, "bad")


def write_sdist(path, package, extra=None, runtime="0.1.3"):
    normalized = package.replace("-", "_")
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(f"{normalized}-0.2.0/PKG-INFO")
        data = distribution_metadata(package, runtime)
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
        if extra:
            archive.addfile(extra)


@pytest.fixture
def release(tmp_path, monkeypatch):
    directory = tmp_path / "dist"
    directory.mkdir()
    inventory = tmp_path / "inventory.json"
    for filename, package in resume.expected_files().items():
        path = directory / filename
        (write_wheel if filename.endswith(".whl") else write_sdist)(path, package)

    def freeze():
        data = {
            "version": resume.VERSION,
            "runtime": resume.RUNTIME,
            "source_sha": resume.SOURCE_SHA,
            "run_id": resume.SOURCE_RUN,
            "artifact_id": resume.SOURCE_ARTIFACT,
            "archive_sha256": resume.ARCHIVE_SHA256,
            "files": [
                {
                    "filename": p.name,
                    "bytes": p.stat().st_size,
                    "sha256": digest(p.read_bytes()),
                }
                for p in sorted(directory.iterdir())
                if p.is_file()
            ],
        }
        inventory.write_text(json.dumps(data))
        monkeypatch.setattr(resume, "INVENTORY_SHA256", digest(inventory.read_bytes()))
        return data

    frozen = freeze()
    native = {
        "info": {"name": "ruvector", "version": "0.1.3"},
        "urls": [
            {"filename": "native.whl", "yanked": False, "digests": {"sha256": "a" * 64}}
        ],
    }
    registry = {"ruvector": native}
    for package in resume.PACKAGES:
        registry[package] = {
            "info": {"name": package, "version": "0.2.0"},
            "urls": [
                {
                    "filename": f["filename"],
                    "yanked": False,
                    "digests": {"sha256": f["sha256"]},
                }
                for f in frozen["files"]
                if resume.expected_files()[f["filename"]] == package
            ],
        }

    def fetcher(package, version):
        return registry.get(package)

    return directory, inventory, registry, fetcher, freeze


def test_complete_release_selects_nothing(release, tmp_path):
    directory, inventory, _, fetcher, _ = release
    output = tmp_path / "missing"
    plan = resume.stage(directory, output, inventory=inventory, fetcher=fetcher)
    assert not plan["missing"] and len(plan["published"]) == 22
    assert list(output.iterdir()) == []


def test_partial_release_selects_only_absent_files(release, tmp_path):
    directory, inventory, registry, fetcher, _ = release
    first = resume.PACKAGES[0]
    missing = registry[first]["urls"].pop()["filename"]
    del registry[resume.PACKAGES[1]]
    output = tmp_path / "missing"
    plan = resume.stage(directory, output, inventory=inventory, fetcher=fetcher)
    assert len(plan["missing"]) == 3 and missing in plan["missing"]
    assert {p.name for p in output.iterdir()} == set(plan["missing"])
    assert all(
        digest(p.read_bytes()) == plan["files"][p.name] for p in output.iterdir()
    )


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "unknown",
        "yanked",
        "yanked_missing",
        "wrong_version",
        "wrong_name",
        "duplicate",
        "digest_missing",
        "upper_digest",
        "empty_urls",
        "wrong_urls_type",
    ],
)
def test_registry_faults_write_no_selection(release, tmp_path, fault):
    directory, inventory, registry, fetcher, _ = release
    data = registry[resume.PACKAGES[-1]]
    if fault == "hash":
        data["urls"][0]["digests"]["sha256"] = "a" * 64
    elif fault == "unknown":
        data["urls"][0]["filename"] = "unknown.whl"
    elif fault == "yanked":
        data["urls"][0]["yanked"] = True
    elif fault == "yanked_missing":
        data["urls"][0].pop("yanked")
    elif fault in {"wrong_version", "wrong_name"}:
        data["info"]["version" if fault == "wrong_version" else "name"] = "unexpected"
    elif fault == "duplicate":
        data["urls"].append(copy.deepcopy(data["urls"][0]))
    elif fault == "digest_missing":
        data["urls"][0]["digests"] = {}
    elif fault == "upper_digest":
        data["urls"][0]["digests"]["sha256"] = "A" * 64
    elif fault == "empty_urls":
        data["urls"] = []
    else:
        data["urls"] = {}
    output = tmp_path / "missing"
    with pytest.raises(resume.GateError):
        resume.stage(directory, output, inventory=inventory, fetcher=fetcher)
    assert not output.exists()


def test_late_registry_outage_writes_nothing(release, tmp_path):
    directory, inventory, registry, fetcher, _ = release
    del registry[resume.PACKAGES[0]]

    def outage(package, version):
        if package == resume.PACKAGES[-1]:
            raise resume.RegistryError("registry HTTP 429")
        return fetcher(package, version)

    output = tmp_path / "missing"
    with pytest.raises(resume.RegistryError):
        resume.stage(directory, output, inventory=inventory, fetcher=outage)
    assert not output.exists()


def test_native_runtime_required(release):
    directory, inventory, registry, fetcher, _ = release
    del registry["ruvector"]
    with pytest.raises(resume.GateError, match="native"):
        resume.precheck(directory, inventory=inventory, fetcher=fetcher)


@pytest.mark.parametrize(
    "fault", ["extra", "absent", "hash", "symlink", "hardlink", "fifo", "oversize"]
)
def test_local_faults_fail_before_registry(release, tmp_path, fault, monkeypatch):
    directory, inventory, _, _, _ = release
    if fault == "fifo" and not hasattr(os, "mkfifo"):
        pytest.skip("POSIX FIFO files are unavailable on this platform")
    path = next(iter(directory.iterdir()))
    if fault == "extra":
        (directory / "unexpected.txt").write_text("extra")
    elif fault == "absent":
        path.unlink()
    elif fault == "hash":
        path.write_bytes(b"wrong")
    elif fault == "symlink":
        original = path.read_bytes()
        path.unlink()
        external = tmp_path / "outside"
        external.write_bytes(original)
        path.symlink_to(external)
    elif fault == "hardlink":
        os.link(path, tmp_path / "linked")
    elif fault == "fifo":
        path.unlink()
        os.mkfifo(path)
    else:
        monkeypatch.setattr(resume, "MAX_FILE_BYTES", 1)

    def should_not_fetch(*args):
        pytest.fail("invalid local artifacts must not reach registry")

    with pytest.raises(resume.GateError):
        resume.precheck(directory, inventory=inventory, fetcher=should_not_fetch)


def test_inventory_tampering(release):
    directory, inventory, _, fetcher, _ = release
    inventory.write_text(inventory.read_text() + " ")
    with pytest.raises(resume.GateError, match="inventory hash"):
        resume.precheck(directory, inventory=inventory, fetcher=fetcher)


@pytest.mark.parametrize(
    "field,value",
    [
        ("runtime", "0.1.4"),
        ("version", "0.3.0"),
        ("source_sha", "a" * 40),
        ("run_id", 1),
        ("artifact_id", 1),
        ("archive_sha256", "b" * 64),
    ],
)
def test_inventory_identity_bound(release, monkeypatch, field, value):
    directory, inventory, _, fetcher, _ = release
    content = json.loads(inventory.read_bytes())
    content[field] = value
    inventory.write_text(json.dumps(content))
    monkeypatch.setattr(resume, "INVENTORY_SHA256", digest(inventory.read_bytes()))
    with pytest.raises(resume.GateError, match="source identity"):
        resume.precheck(directory, inventory=inventory, fetcher=fetcher)


@pytest.mark.parametrize("suffix", [".whl", ".tar.gz"])
def test_wrong_runtime_rejected_even_when_fixture_hash_updated(release, suffix):
    directory, inventory, _, fetcher, freeze = release
    filename, package = next(
        (n, p) for n, p in resume.expected_files().items() if n.endswith(suffix)
    )
    path = directory / filename
    (write_wheel if suffix == ".whl" else write_sdist)(path, package, runtime="0.1.4")
    freeze()
    with pytest.raises(resume.GateError, match="runtime"):
        resume.precheck(directory, inventory=inventory, fetcher=fetcher)


@pytest.mark.parametrize(
    "path",
    [
        "../escape",
        "/absolute",
        "ruvector_attention/.secret",
        "ruvector_attention/key.pem",
    ],
)
def test_wheel_unsafe_paths(release, path):
    directory, inventory, _, fetcher, freeze = release
    wheel = directory / "ruvector_attention-0.2.0-py3-none-any.whl"
    write_wheel(wheel, "ruvector-attention", extra=path)
    freeze()
    with pytest.raises(resume.GateError):
        resume.precheck(directory, inventory=inventory, fetcher=fetcher)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "device", "fifo", "traversal"])
def test_sdist_unsafe_members(release, kind):
    directory, inventory, _, fetcher, freeze = release
    member = tarfile.TarInfo("ruvector_attention-0.2.0/src/bad")
    member.type = {
        "symlink": tarfile.SYMTYPE,
        "hardlink": tarfile.LNKTYPE,
        "device": tarfile.CHRTYPE,
        "fifo": tarfile.FIFOTYPE,
        "traversal": tarfile.REGTYPE,
    }[kind]
    if kind == "traversal":
        member.name = "ruvector_attention-0.2.0/../escape"
    write_sdist(
        directory / "ruvector_attention-0.2.0.tar.gz",
        "ruvector-attention",
        extra=member,
    )
    freeze()
    with pytest.raises(resume.GateError):
        resume.precheck(directory, inventory=inventory, fetcher=fetcher)


def test_archive_member_and_expansion_bounds(release, monkeypatch):
    directory, inventory, _, fetcher, _ = release
    monkeypatch.setattr(resume, "MAX_MEMBERS", 1)
    with pytest.raises(resume.GateError, match="members"):
        resume.precheck(directory, inventory=inventory, fetcher=fetcher)
    monkeypatch.setattr(resume, "MAX_MEMBERS", 256)
    monkeypatch.setattr(resume, "MAX_EXPANDED_BYTES", 1)
    with pytest.raises(resume.GateError, match="expanded"):
        resume.precheck(directory, inventory=inventory, fetcher=fetcher)


def test_existing_staging_never_overwritten(release, tmp_path):
    directory, inventory, _, fetcher, _ = release
    output = tmp_path / "missing"
    output.mkdir()
    sentinel = output / "keep"
    sentinel.write_text("retained")
    with pytest.raises(resume.GateError, match="already exists"):
        resume.stage(directory, output, inventory=inventory, fetcher=fetcher)
    assert sentinel.read_text() == "retained"


@pytest.mark.parametrize("status", [403, 429, 500])
def test_http_errors_are_unavailable_not_absent(monkeypatch, status):
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("https://pypi.org", status, "", {}, None)

    monkeypatch.setattr(resume.urllib.request, "urlopen", fail)
    with pytest.raises(resume.RegistryError, match=str(status)):
        resume.fetch_json("ruvector-core", "0.2.0")


def test_http_404_means_absent(monkeypatch):
    def absent(*args, **kwargs):
        raise urllib.error.HTTPError("https://pypi.org", 404, "", {}, None)

    monkeypatch.setattr(resume.urllib.request, "urlopen", absent)
    assert resume.fetch_json("ruvector-core", "0.2.0") is None


def test_duplicate_metadata_identity_rejected():
    data = distribution_metadata("ruvector-attention").replace(
        b"Version:", b"Name: ruvector-attention\nVersion:"
    )
    with pytest.raises(resume.GateError, match="identity"):
        resume.metadata(data, "ruvector-attention")


def test_runtime_environment_marker_and_extra_rejected():
    for requirement in ("ruvector==0.1.3; python_version<'3.13'", "ruvector[x]==0.1.3"):
        data = distribution_metadata("ruvector-attention").replace(
            b"ruvector==0.1.3", requirement.encode()
        )
        with pytest.raises(resume.GateError, match="runtime"):
            resume.metadata(data, "ruvector-attention")


@pytest.mark.parametrize(
    "verify,missing,expected", [(True, False, 0), (True, True, 1), (False, True, 0)]
)
def test_cli_verify_receipt(
    release, tmp_path, monkeypatch, capsys, verify, missing, expected
):
    directory, inventory, registry, fetcher, _ = release
    if missing:
        del registry[resume.PACKAGES[-1]]
    original = resume.precheck
    monkeypatch.setattr(
        resume,
        "precheck",
        lambda path, inventory: original(path, inventory=inventory, fetcher=fetcher),
    )
    receipt = tmp_path / "receipt.json"
    argv = [
        "resume",
        "--dist",
        str(directory),
        "--inventory",
        str(inventory),
        "--receipt",
        str(receipt),
    ]
    if verify:
        argv.append("--verify-only")
    monkeypatch.setattr(resume, "__name__", "resume")
    monkeypatch.setattr("sys.argv", argv)
    assert resume.main() == expected
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == ("validated" if expected == 0 else "blocked")
    assert receipt.exists() == (expected == 0)
    if expected == 0:
        assert json.loads(receipt.read_text()) == output


def test_cli_verify_cannot_select(tmp_path, monkeypatch, capsys):
    output = tmp_path / "missing"
    monkeypatch.setattr(
        "sys.argv",
        [
            "resume",
            "--dist",
            str(tmp_path),
            "--missing-dir",
            str(output),
            "--verify-only",
        ],
    )
    assert resume.main() == 1
    assert "cannot select" in capsys.readouterr().out
    assert not output.exists()


@pytest.mark.parametrize(
    "body", [b"[]", b"null", b"not-json", b"x" * (resume.MAX_RESPONSE_BYTES + 1)]
)
def test_registry_response_bounds_and_schema(monkeypatch, body):
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, limit):
            return body[:limit]

    monkeypatch.setattr(
        resume.urllib.request, "urlopen", lambda *args, **kwargs: Response()
    )
    with pytest.raises(resume.RegistryError):
        resume.fetch_json("ruvector-core", "0.2.0")


def test_metadata_allocation_bound(monkeypatch):
    monkeypatch.setattr(resume, "MAX_METADATA_BYTES", 5)
    with pytest.raises(resume.GateError, match="exceeds bounds"):
        resume.metadata(
            distribution_metadata("ruvector-attention"), "ruvector-attention"
        )


@pytest.mark.skipif(
    os.name == "nt", reason="POSIX chmod does not model Windows workspace ACLs"
)
def test_owned_directory_cannot_be_group_writable(release):
    directory, inventory, _, fetcher, _ = release
    directory.chmod(0o775)
    with pytest.raises(resume.GateError, match="private"):
        resume.precheck(directory, inventory=inventory, fetcher=fetcher)


def test_windows_synthetic_mode_0777_uses_trusted_workspace_acl(release, monkeypatch):
    directory, inventory, _, fetcher, _ = release
    directory.chmod(0o777)
    original = resume.directory_mode_safe
    assert original(0o777, "nt")
    assert not original(0o777, "posix")
    monkeypatch.setattr(
        resume, "directory_mode_safe", lambda mode: original(mode, "nt")
    )
    plan, files = resume.precheck(directory, inventory=inventory, fetcher=fetcher)
    assert len(plan["published"]) == 22 and not files


@pytest.mark.parametrize("mode", [0o775, 0o757, 0o777])
def test_posix_directory_write_mode_guard(mode):
    assert not resume.directory_mode_safe(mode, "posix")
    assert resume.directory_mode_safe(0o755, "posix")
