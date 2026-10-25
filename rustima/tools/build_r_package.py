"""Build a self-contained R source package from the shared Rust core."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def build(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rustima-r-source-") as tmp:
        stage = Path(tmp) / "rustima"
        ignore = shutil.ignore_patterns("target", ".cargo", "vendor", "*.o", "*.so", "*.dll", "Makevars", "Makevars.win", "vendor.tar.xz")
        shutil.copytree(ROOT / "packages/r/rustima", stage, ignore=ignore)
        rust = stage / "src/rust"
        shutil.copytree(ROOT / "crates/rustima-core", rust / "engine", ignore=ignore)
        shutil.copy2(ROOT / "LICENSE", rust / "engine/LICENSE")
        manifest = rust / "Cargo.toml"
        text = manifest.read_text()
        old = '../../../../../crates/rustima-core'
        if old not in text:
            raise RuntimeError("R adapter core dependency changed; update the source-package builder")
        manifest.write_text(text.replace(old, "engine"))
        vendor = stage / "src/vendor"
        command = ["cargo", "vendor", "--locked", "--versioned-dirs", "../vendor"]
        config = subprocess.check_output(command, cwd=rust, text=True)
        if 'directory = "../vendor"' not in config:
            raise RuntimeError("cargo vendor did not return the expected source replacement config")
        (rust / "vendor-config.toml").write_text(config.replace('directory = "../vendor"', 'directory = "vendor"'))
        with tarfile.open(rust / "vendor.tar.xz", "w:xz") as archive:
            archive.add(vendor, arcname="vendor")
        metadata = json.loads(subprocess.check_output(
            ["cargo", "metadata", "--locked", "--format-version", "1", "--manifest-path", str(manifest)], text=True))
        inst = stage / "inst"
        inst.mkdir(exist_ok=True)
        (inst / "COPYRIGHTS").write_text("\n".join(
            f"{p['name']} {p['version']}: {p.get('license') or 'see bundled source'}; " + ", ".join(p["authors"])
            for p in metadata["packages"]) + "\n")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        digest = hashlib.sha256()
        for source in sorted((rust / "engine").rglob("*")):
            if source.is_file():
                digest.update(source.relative_to(rust / "engine").as_posix().encode() + b"\0")
                digest.update(source.read_bytes())
        (inst / "engine-provenance").write_text(
            f"Repository HEAD: {commit}\nBundled core SHA-256: {digest.hexdigest()}\n"
            "The fingerprint identifies the bundled files, including any uncommitted edits.\n")
        subprocess.run(["R", "CMD", "build", "--no-build-vignettes", str(stage)], cwd=output, check=True)
    candidates = sorted(output.glob("rustima_*.tar.gz"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError("R CMD build produced no source archive")
    with tarfile.open(candidates[-1]) as archive:
        names = archive.getnames()
        assert any(n.endswith("/src/rust/engine/src/lib.rs") for n in names)
        assert any(n.endswith("/src/rust/vendor.tar.xz") for n in names)
        assert not any("lbfgsb_c" in n or "/target/" in n for n in names)
    print(candidates[-1])
    return candidates[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/r")
    build(parser.parse_args().output)
