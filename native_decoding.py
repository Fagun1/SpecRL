"""Python side of the native llama.cpp bridge (one JSON line per command)."""
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def gpu_used_mib():
    try:
        data = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                        "--format=csv,noheader,nounits"], text=True)
        return int(data.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class NativeDecoder:
    """Both models resident on the GPU; `standalone=True` never loads the draft."""

    def __init__(self, target, draft, context=256, threads=6, target_gpu_layers=-1,
                 draft_gpu_layers=-1, runtime=None, bridge=None, log_path=None,
                 standalone=False, cuda_graphs=True):
        runtime = Path(runtime or Path(os.environ["LOCALAPPDATA"]) / "Programs/Ollama/lib/ollama").resolve()
        bridge = Path(bridge or ROOT / "native/bridge.exe").resolve()
        cuda = runtime / "cuda_v12"
        target, draft = Path(target).resolve(), Path(draft).resolve()
        for file in (bridge, runtime / "libllama.dll", target, draft):
            if not file.is_file():
                raise FileNotFoundError(file)
        expected = json.loads((ROOT / "native/runtime_manifest.json").read_text())
        for filename, digest in expected["runtime_sha256"].items():
            if fingerprint(runtime / filename) != digest:
                raise RuntimeError(f"Runtime ABI changed ({filename}); install Ollama "
                                   f"{expected['ollama_version_observed']} or rebuild against matching headers")
        self.log_path = Path(log_path or ROOT / "native-runtime.log").resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("w", encoding="utf-8")
        env = dict(os.environ)
        env["PATH"] = str(runtime) + os.pathsep + str(cuda) + os.pathsep + env["PATH"]
        if cuda_graphs:
            env.pop("GGML_CUDA_DISABLE_GRAPHS", None)
        else:
            env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
        env.pop("GGML_CUDA_ENABLE_UNIFIED_MEMORY", None)
        env["LTD_STANDALONE_TARGET_ONLY"] = "1" if standalone else "0"
        self.standalone = standalone
        before = gpu_used_mib()
        start = time.perf_counter()
        self.proc = subprocess.Popen(
            [str(bridge), str(target), str(draft), str(target_gpu_layers), str(draft_gpu_layers),
             str(context), str(threads), str(runtime), str(cuda)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            ready = self._read()
        except Exception:
            self.close()
            raise
        after = gpu_used_mib()
        runtime_log = self.log_path.read_text(encoding="utf-8", errors="replace")
        offloads = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", runtime_log)
        if (target_gpu_layers or draft_gpu_layers) and "CUDA0" not in runtime_log:
            self.close()
            raise RuntimeError("GPU was requested but CUDA allocation was not confirmed in the runtime log")
        self.metadata = {
            "backend": "native llama.cpp cycle bridge", "backend_version": "74ade5274",
            "target_file": target.name, "draft_file": None if standalone else draft.name,
            "target_sha256": fingerprint(target), "draft_sha256": None if standalone else fingerprint(draft),
            "context": context, "threads": threads, "standalone": standalone, "cuda_graphs": cuda_graphs,
            "gpu_used_mib_after_load": after,
            "gpu_memory_delta_mib": after - before if after is not None and before is not None else None,
            "setup_seconds": time.perf_counter() - start, **ready,
            "offloaded_layers": [f"{a}/{b}" for a, b in offloads],
        }
        self.max_draft = None

    def _read(self):
        line = self.proc.stdout.readline()
        if not line:
            self.log.flush()
            tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
            raise RuntimeError(f"Native bridge exited ({self.proc.poll()}):\n{tail}")
        value = json.loads(line)
        if "error_hex" in value:
            raise RuntimeError(bytes.fromhex(value["error_hex"]).decode(errors="replace"))
        return value

    def _command(self, text):
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()
        return self._read()

    def configure(self, max_draft, thresholds):
        self._command(f"CONFIG {max_draft} {len(thresholds)} " + " ".join(map(str, thresholds)))
        self.max_draft = max_draft
        self.buckets = len(thresholds) + 1

    def reset(self, prompt, token_limit):
        text = ("<|im_start|>user\n" + prompt + "<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        result = self._command(f"RESET {token_limit} {text.encode().hex()}")
        self.output = []
        self.done = False
        self.lag = result["lag"]
        return result["prefill_seconds"]

    def cycle(self, draft_first, bits):
        """Run one verification cycle; `bits` is the per-(k, bucket) continue table."""
        if self.done:
            raise RuntimeError("Terminal episode")
        start = time.perf_counter()
        result = self._command(f"CYCLE {int(draft_first)} {bits or '-'}")
        result["wall_seconds"] = time.perf_counter() - start
        self.output.extend(result["emitted_ids"])
        self.done = result["done"]
        self.lag = result["lag_after"]
        return result

    def text(self):
        return bytes.fromhex(self._command("TEXT")["text_hex"]).decode(errors="replace")

    def pieces(self, ids):
        """Text of each token id (used by the demo dashboard)."""
        if not ids:
            return []
        reply = self._command("PIECES " + " ".join(map(str, ids)))
        return [bytes.fromhex(h).decode(errors="replace") for h in reply["pieces_hex"]]

    def close(self):
        proc = getattr(self, "proc", None)
        if proc and proc.poll() is None:
            try:
                proc.stdin.write("QUIT\n")
                proc.stdin.flush()
                proc.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                proc.kill()
                proc.wait()
        log = getattr(self, "log", None)
        if log and not log.closed:
            log.close()
