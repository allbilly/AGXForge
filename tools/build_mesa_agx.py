#!/usr/bin/env python3
"""Build the optional, pinned Mesa AGX userspace compiler on macOS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VERSION = "26.2.4"
URL = f"https://archive.mesa3d.org/mesa-{VERSION}.tar.xz"
SHA256 = "bce5f7fbebb934373b86c999a064d52fb5065878dc57f287f95346648ec832e9"


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="existing isolated Mesa 26.2.4 checkout")
    parser.add_argument("--build", type=Path, default=ROOT / f"build/macos/mesa-{VERSION}")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if sys.platform != "darwin": parser.error("this adapter build is currently validated on macOS")
    if args.jobs < 1: parser.error("jobs must be positive")
    cache = ROOT / "build/macos/mesa-source"; cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"mesa-{VERSION}.tar.xz"
    source = args.source.resolve() if args.source else cache / f"mesa-{VERSION}"
    if not archive.is_file():
        temporary = archive.with_suffix(".download")
        urllib.request.urlretrieve(URL, temporary)
        if digest(temporary) != SHA256: raise RuntimeError("Mesa archive hash mismatch")
        temporary.rename(archive)
    if digest(archive) != SHA256: raise RuntimeError("Mesa archive hash mismatch")
    if not args.source:
        if not source.exists():
            with tarfile.open(archive) as tar: tar.extractall(cache, filter="data")
    if (source / "VERSION").read_text().strip() != VERSION:
        raise RuntimeError("Mesa adapter is pinned to 26.2.4")
    # Upstream's macOS IOKit branch revisits genxml when tools=asahi is set.
    meson = source / "src/asahi/meson.build"
    original = "elif dep_iokit.found()"
    patched = "elif dep_iokit.found() and not with_tools.contains('asahi')"
    text = meson.read_text()
    if patched not in text:
        if text.count(original) != 1: raise RuntimeError("unexpected Mesa build layout")
        meson.write_text(text.replace(original, patched))
    overlay = source / "agxforge-bridge"; overlay.mkdir(exist_ok=True)
    for name in ("bridge.c", "meson.build"):
        shutil.copyfile(ROOT / "tools/mesa_agx" / name, overlay / name)
    meson = source / "meson.build"; text = meson.read_text()
    marker = "\nsubdir('agxforge-bridge')\n"
    if marker not in text: meson.write_text(text + marker)
    build = args.build.resolve()
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + ":/opt/homebrew/opt/llvm/bin:" + env["PATH"]
    env["PYTHONPATH"] = str(ROOT / "build/macos/mesa-python") + os.pathsep + env.get("PYTHONPATH", "")
    options = ["-Dtools=asahi", "-Dgallium-drivers=", "-Dvulkan-drivers=", "-Dplatforms=",
               "-Dglx=disabled", "-Degl=disabled", "-Dgbm=disabled", "-Dgles1=disabled",
               "-Dgles2=disabled", "-Dopengl=false", "-Dllvm=enabled", "-Dbuild-tests=false",
               "-Dshader-cache=disabled", "-Dbuildtype=release"]
    command = ["meson", "setup", str(build), str(source), *options]
    if (build / "build.ninja").exists():
        info = json.loads((build / "meson-info/meson-info.json").read_text())
        command.append("--reconfigure" if Path(info["directories"]["source"]).resolve() == source.resolve() else "--wipe")
    subprocess.run(command, env=env, check=True)
    subprocess.run(["ninja", "-C", str(build), "-j", str(args.jobs),
                    "agxforge-bridge/agxforge_mesa"], env=env, check=True)
    executable = build / "agxforge-bridge/agxforge_mesa"
    # Record the actual sources, including external-checkout mode; don't infer
    # that an arbitrary supplied checkout equals the archive merely by version.
    files = []
    for directory in ("src/asahi", "src/compiler", "src/util", "include", "agxforge-bridge"):
        files += [p for p in (source / directory).rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts]
    files += [source / "meson.build", source / "meson.options", source / "VERSION"]
    sources = {str(p.relative_to(source)): digest(p) for p in sorted(files)}
    unchecked = set(sources) - {"agxforge-bridge/bridge.c", "agxforge-bridge/meson.build"}
    with tarfile.open(archive) as tar:
        for member in tar:
            relative = member.name.partition("/")[2]
            if relative not in unchecked or not (member.isfile() or member.issym() or member.islnk()): continue
            data = tar.extractfile(member).read()
            if relative == "meson.build": data += marker.encode()
            if relative == "src/asahi/meson.build": data = data.replace(original.encode(), patched.encode())
            if hashlib.sha256(data).hexdigest() != sources[relative]:
                raise RuntimeError(f"Mesa source differs from pinned archive: {relative}")
            unchecked.remove(relative)
    if unchecked: raise RuntimeError(f"unrecognized Mesa source files: {sorted(unchecked)}")
    identity = dict(schema_version=1, compiler=f"Mesa {VERSION} AGX", source_url=URL,
                    pinned_archive_sha256=SHA256, archive_verified=True, compiler_sources_verified=True,
                    compiler_sha256=digest(executable), build_options=options,
                    sources=sources,
                    adapter_sha256=digest(ROOT / "tools/mesa_agx/bridge.c"))
    (executable.parent / "build-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(f"Built {executable}")


if __name__ == "__main__": main()
