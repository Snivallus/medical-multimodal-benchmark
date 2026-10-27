"""Serve one or more local VLMs with vLLM and run the benchmark against them.

The benchmark itself is unchanged: every model gets the same prompt, the same
images and the same scoring rule. This script only automates the plumbing:

    vllm serve <model> --port <p> ...   # one server per GPU
    evaluate.py --variants photo,text_b,text_a --max-tokens 8192 --resume
    evaluate.py --mode conflicts --max-tokens 8192 --resume

    python bench/run_vlm_eval.py --models qwen3-vl-8b-instruct
    python bench/run_vlm_eval.py --models qwen3-vl-8b-instruct,internvl3.5-8b --smoke
    python bench/run_vlm_eval.py --gpus 0,1,2

Model paths default to ``$MEDBENCH_MODELS`` (``~/models/medbench``) and can be
overridden with ``--model-root``.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH = REPO_ROOT / "bench"
VENV = BENCH / ".venv"
# vLLM lives in the venv, and so does the only copy of requests that
# evaluate.py can import; the system python is not enough.
PYTHON = VENV / "bin" / "python"
EVALUATE = BENCH / "evaluate.py"
RESULTS = BENCH / "results"

MODELS = {
    "qwen3-vl-8b-instruct": {
        "dir": "qwen3-vl-8b-instruct",
        "port": 8000,
        "trust": False,
        "max_len": 40960,
    },
    "internvl3.5-8b": {
        "dir": "internvl3.5-8b",
        "port": 8001,
        "trust": True,
        "max_len": 40960,
    },
    "glm-4.1v-9b-thinking": {
        "dir": "glm-4.1v-9b-thinking",
        "port": 8002,
        "trust": False,
        "max_len": 40960,
    },
}


def wait_for_server(port, process, log_path, timeout=1800):
    """Poll ``/v1/models`` until the server answers or the process dies."""

    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout

    while time.time() < deadline:

        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
            raise RuntimeError(f"vLLM exited with {process.returncode}:\n{tail}")

        try:
            request = urllib.request.Request(url, headers={"Authorization": "Bearer dummy"})

            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status == 200:
                    return

        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass

        time.sleep(5)

    raise RuntimeError(f"vLLM on port {port} did not become ready in {timeout}s")


def port_is_free(port):
    """Return ``False`` when something already answers on ``port``.

    A vLLM server left over from a killed run would answer ``/v1/models`` too,
    so without this check the new run would silently evaluate the old server.
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def start_server(
    name,
    model_root,
    gpu,
    log_dir,
    gpu_memory_utilization=0.90,
    max_num_seqs=None,
    max_model_len=None,
):
    spec = MODELS[name]
    model_path = Path(model_root) / spec["dir"]

    if not model_path.exists():
        raise SystemExit(f"model directory missing: {model_path}")

    index = model_path / "model.safetensors.index.json"

    if not index.exists():
        raise SystemExit(f"model index missing (download unfinished?): {model_path}")

    if list(model_path.glob("*.part")):
        raise SystemExit(f"model still has .part files (download unfinished?): {model_path}")

    # The index names every shard; a shard that never arrived fails only after
    # vLLM has spent minutes loading, so check it here instead.
    weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    missing = sorted(
        {name for name in weight_map.values() if not (model_path / name).exists()}
    )

    if missing:
        raise SystemExit(f"model shard(s) missing in {model_path}: {', '.join(missing[:3])}")

    if not port_is_free(spec["port"]):
        raise SystemExit(
            f"port {spec['port']} is already in use (stale vLLM?); "
            f"stop it before starting {name}"
        )

    log_path = log_dir / "server.log"
    handle = open(log_path, "w", encoding="utf-8")

    # One image slot per item is not enough: 9 items carry four images.
    command = [
        str(VENV / "bin" / "vllm"),
        "serve",
        str(model_path),
        "--served-model-name",
        name,
        "--host",
        "127.0.0.1",
        "--port",
        str(spec["port"]),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(max_model_len or spec["max_len"]),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--limit-mm-per-prompt",
        # This benchmark has no video, so the video slot is closed entirely.
        # That also matters for InternVL: with the default video limit its V1
        # dummy-video profiling hits a numpy-int64 -> PIL conversion bug in
        # vLLM 0.11.0, which kills the engine before it serves a request.
        '{"image": 4, "video": 0}',
        "--disable-log-requests",
    ]

    if spec["trust"]:
        command.append("--trust-remote-code")

    if max_num_seqs is not None:
        command += ["--max-num-seqs", str(max_num_seqs)]

    handle.write(" ".join(command) + "\n")
    handle.flush()

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )

    print(f"[{name}] server pid {process.pid} on GPU {gpu}, port {spec['port']}", flush=True)
    print(f"[{name}] log -> {log_path}", flush=True)

    return process, handle, log_path


def stop_server(name, process, handle):
    if process.poll() is not None:
        handle.close()
        return

    print(f"[{name}] stopping server", flush=True)

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass

    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait()

    handle.close()


def run_evaluate(name, output, extra, log_dir, label):
    env = dict(os.environ)
    env["LLM_API_KEY"] = "dummy"
    env["LLM_BASE_URL"] = f"http://127.0.0.1:{MODELS[name]['port']}/v1"
    env["LLM_MODEL"] = name
    # evaluate.py prints Chinese and the parent reads through a pipe; make the
    # encoding explicit instead of relying on the locale of this shell.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    python = str(PYTHON) if PYTHON.exists() else sys.executable

    command = [
        python,
        str(EVALUATE),
        "--model",
        name,
        "--output",
        str(output),
        "--resume",
        *extra,
    ]

    log_path = log_dir / f"{label}.log"
    print(f"[{name}] {label}: {' '.join(command)}", flush=True)

    # Append: a resumed run should not erase the log of the attempt it resumes.
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"\n### {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {' '.join(command)}\n")
        handle.flush()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=str(REPO_ROOT),
        )

        for line in process.stdout:
            handle.write(line)
            if line.startswith("["):
                print(line.rstrip(), flush=True)

        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"{label} failed with {return_code}; see {log_path}")

    return log_path


def write_manifest(
    root,
    names,
    model_root,
    max_tokens,
    limit=None,
    skip_conflicts=False,
    gpu_memory_utilization=None,
    max_num_seqs=None,
    max_model_len=None,
):
    """Record what was run: model revisions, package versions, git commit."""

    def version(package):
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return None

    models = {}

    for name in names:
        path = Path(model_root) / MODELS[name]["dir"]
        download = path / "_download_manifest.json"
        revision = None

        if download.exists():
            revision = json.loads(download.read_text(encoding="utf-8")).get("revision")

        models[name] = {
            "path": str(path),
            "revision": revision,
            "port": MODELS[name]["port"],
            "max_model_len": max_model_len or MODELS[name]["max_len"],
        }

    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": commit,
        # The harness scripts themselves can be modified while HEAD stays put;
        # without this flag the recorded commit looks like the code that ran.
        "git_dirty": dirty,
        "vllm": version("vllm"),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "max_tokens": max_tokens,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_seqs": max_num_seqs,
        # The effective cap, including --max-model-len overrides.
        "max_model_len": max_model_len,
        "variants": ["photo", "text_b", "text_a"],
        "limit": limit,
        "skip_conflicts": skip_conflicts,
        "models": models,
    }

    path = Path(root) / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"manifest -> {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run the benchmark against local vLLM servers.")
    parser.add_argument(
        "--models",
        default="qwen3-vl-8b-instruct,internvl3.5-8b,glm-4.1v-9b-thinking",
        help="comma separated model names from " + ", ".join(MODELS),
    )
    parser.add_argument(
        "--gpus",
        default="0,1,2",
        help="comma separated GPU ids, one per model",
    )
    parser.add_argument(
        "--model-root",
        default=os.environ.get("MEDBENCH_MODELS", str(Path.home() / "models" / "medbench")),
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="vLLM's per-server GPU memory cap; lower it to share a busy GPU",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="vLLM concurrency cap; 1 is enough for the sequential evaluation",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="override the per-model context cap (default: MODELS[...]['max_len'])",
    )
    parser.add_argument("--smoke", action="store_true", help="5 items and 5 conflicts only")
    parser.add_argument("--skip-conflicts", action="store_true")
    args = parser.parse_args()

    names = [name.strip() for name in args.models.split(",") if name.strip()]
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]

    unknown = [name for name in names if name not in MODELS]

    if unknown:
        raise SystemExit(f"unknown model(s): {unknown}")

    if len(gpus) < len(names):
        raise SystemExit(f"{len(names)} model(s) need {len(names)} GPU id(s)")

    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("--gpu-memory-utilization must be in (0, 1]")

    if args.max_num_seqs is not None and args.max_num_seqs < 1:
        raise SystemExit("--max-num-seqs must be at least 1")

    root = RESULTS / "_smoke" if args.smoke else RESULTS
    running = []

    try:
        for name, gpu in zip(names, gpus):
            log_dir = root / name
            log_dir.mkdir(parents=True, exist_ok=True)
            process, handle, log_path = start_server(
                name,
                args.model_root,
                gpu,
                log_dir,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_num_seqs=args.max_num_seqs,
                max_model_len=args.max_model_len,
            )
            running.append((name, process, handle, log_path))

        for name, process, handle, log_path in running:
            print(f"[{name}] waiting for server ...", flush=True)
            wait_for_server(MODELS[name]["port"], process, log_path)

        write_manifest(
            root,
            names,
            args.model_root,
            args.max_tokens,
            limit=5 if args.smoke else None,
            skip_conflicts=args.skip_conflicts,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
        )

        def run_model(name):
            log_dir = root / name
            limit = ["--limit", "5"] if args.smoke else []
            outputs = []

            run_evaluate(
                name,
                log_dir / "items.jsonl",
                [
                    "--variants",
                    "photo,text_b,text_a",
                    "--max-tokens",
                    str(args.max_tokens),
                    *limit,
                ],
                log_dir,
                "items",
            )
            outputs.append(log_dir / "items.jsonl")

            if not args.skip_conflicts:
                run_evaluate(
                    name,
                    log_dir / "conflicts.jsonl",
                    ["--mode", "conflicts", "--max-tokens", str(args.max_tokens), *limit],
                    log_dir,
                    "conflicts",
                )
                outputs.append(log_dir / "conflicts.jsonl")

            print(f"[{name}] done: {', '.join(str(path) for path in outputs)}", flush=True)

        # One server per GPU, so the models can be evaluated at the same time.
        # GLM's chain of thought is the long pole; running the short models
        # first and in sequence would add their wall-clock time on top of it.
        with ThreadPoolExecutor(max_workers=len(running)) as pool:
            futures = {pool.submit(run_model, name): name for name, _, _, _ in running}

            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as error:  # noqa: BLE001 - reported, then re-raised
                    print(f"[{name}] failed: {error}", flush=True)
                    raise

    finally:
        for name, process, handle, _ in running:
            stop_server(name, process, handle)


if __name__ == "__main__":
    main()
