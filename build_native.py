"""Build the cycle bridge against the exact installed, pinned Ollama runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime",type=Path,default=Path(os.environ.get("LOCALAPPDATA",""))/"Programs/Ollama/lib/ollama")
    p.add_argument("--compiler",type=Path)
    args=p.parse_args()
    manifest=json.loads((ROOT/"native/runtime_manifest.json").read_text())
    for name,digest in manifest["runtime_sha256"].items():
        file=args.runtime/name
        if hashlib.sha256(file.read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f"Runtime {name} differs from pinned ABI. Use matching headers/runtime; do not build with a different struct layout.")
    candidates=[args.compiler,Path.home()/"Downloads/w64devkit/bin/g++.exe",Path("C:/msys64/mingw64/bin/g++.exe")]
    found=shutil.which("g++")
    if found:candidates.append(Path(found))
    compiler=None
    for candidate in candidates:
        if candidate and candidate.is_file():
            architecture=subprocess.check_output([str(candidate),"-dumpmachine"],text=True).strip()
            if architecture.startswith("x86_64"):
                compiler=candidate.resolve();break
    if compiler is None:
        raise RuntimeError("A 64-bit MinGW-w64 C++17 compiler is required. Pass --compiler PATH.")
    env=dict(os.environ);env["PATH"]=str(compiler.parent)+os.pathsep+env["PATH"]
    command=[str(compiler),"-std=c++17","-O2","-static-libgcc","-static-libstdc++",
             "-I",str(ROOT/"native/include"),str(ROOT/"native/bridge.cpp")]
    command += [str(args.runtime/name) for name in ("libllama.dll","ggml.dll","ggml-base.dll")]
    command += ["-o",str(ROOT/"native/bridge.exe")]
    subprocess.run(command,env=env,check=True)
    print("Built",ROOT/"native/bridge.exe")


if __name__=="__main__":main()
