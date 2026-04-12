from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "sft-qwen-3.5-4B"
NETRC_PATH = Path("~/.netrc").expanduser()
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/vol"
DEFAULT_GPU = "H100:4"
DEFAULT_PPO_GPU = "H100:1"
DEFAULT_CPU = 8.0
DEFAULT_MEMORY_MB = 65536
DEFAULT_TIMEOUT_SECONDS = 60 * 60 * 24
DEFAULT_VOLUME_COMMIT_INTERVAL_SECONDS = 300
volume = modal.Volume.from_name("sft-qwen-3.5-4B-volume", create_if_missing=True)


def load_gitignore_patterns() -> list[str]:
    """Translate .gitignore entries into Modal ignore globs."""
    if not modal.is_local():
        return []

    root = Path(__file__).resolve().parent
    gitignore_path = root / ".gitignore"
    if not gitignore_path.is_file():
        return []

    patterns: list[str] = []
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("!"):
            continue
        entry = entry.lstrip("/")
        if entry.endswith("/"):
            entry = entry.rstrip("/")
            patterns.append(f"**/{entry}/**")
        else:
            patterns.append(f"**/{entry}")
    return patterns


def _to_volume_path(path_value: str) -> str:
    p = Path(path_value)
    if p.is_absolute():
        p_str = str(p)
        if p_str != VOLUME_PATH and not p_str.startswith(f"{VOLUME_PATH}/"):
            print(
                f"[modal][warning] path '{p_str}' is outside '{VOLUME_PATH}'. "
                "Files written there may not persist after the run."
            )
        return path_value
    return str(Path(VOLUME_PATH) / p)


def _rewrite_path_flag(
    args: list[str], flag: str, *, default_relative_if_missing: str | None = None
) -> list[str]:
    out = list(args)
    found = False
    i = 0
    while i < len(out):
        token = out[i]
        if token == flag:
            found = True
            if i + 1 >= len(out):
                raise ValueError(f"Missing value for {flag}")
            out[i + 1] = _to_volume_path(out[i + 1])
            i += 2
            continue
        if token.startswith(f"{flag}="):
            found = True
            key, value = token.split("=", 1)
            out[i] = f"{key}={_to_volume_path(value)}"
        i += 1

    if not found and default_relative_if_missing is not None:
        out.extend([flag, _to_volume_path(default_relative_if_missing)])
    return out


def _to_project_path(path_value: str) -> str:
    p = Path(path_value)
    if p.is_absolute():
        return path_value
    return str(Path(PROJECT_DIR) / p)


def _rewrite_project_path_flag(
    args: list[str], flag: str, *, default_relative_if_missing: str | None = None
) -> list[str]:
    out = list(args)
    found = False
    i = 0
    while i < len(out):
        token = out[i]
        if token == flag:
            found = True
            if i + 1 >= len(out):
                raise ValueError(f"Missing value for {flag}")
            out[i + 1] = _to_project_path(out[i + 1])
            i += 2
            continue
        if token.startswith(f"{flag}="):
            found = True
            key, value = token.split("=", 1)
            out[i] = f"{key}={_to_project_path(value)}"
        i += 1

    if not found and default_relative_if_missing is not None:
        out.extend([flag, _to_project_path(default_relative_if_missing)])
    return out


def _normalize_modal_args(args: tuple[str, ...], *, is_eval: bool) -> list[str]:
    normalized = list(args)
    # Keep checkpoint outputs/adapters on the mounted Modal volume (/vol).
    normalized = _rewrite_path_flag(
        normalized,
        "--output_dir",
        default_relative_if_missing=None if is_eval else "runs/default",
    )
    return normalized


def _normalize_ppo_args(args: tuple[str, ...], *, is_eval: bool) -> list[str]:
    normalized = list(args)
    normalized = _rewrite_path_flag(
        normalized,
        "--output_dir",
        default_relative_if_missing=None if is_eval else "runs/ppo_from_sft",
    )
    # training_set.json uses relative ms/si paths, so paper_root should usually
    # point to a Modal volume location that contains "all paper ft data/...".
    normalized = _rewrite_path_flag(
        normalized,
        "--paper_root",
        default_relative_if_missing="" if not is_eval else None,
    )
    normalized = _rewrite_path_flag(normalized, "--cache_prompts_path")
    # training_qwen.jsonl is mounted with the project source tree, not in /vol.
    normalized = _rewrite_project_path_flag(normalized, "--training_qwen_path")
    return normalized


def _normalize_grpo_args(args: tuple[str, ...], *, is_eval: bool) -> list[str]:
    normalized = list(args)
    normalized = _rewrite_path_flag(
        normalized,
        "--output_dir",
        default_relative_if_missing=None if is_eval else "runs/grpo_from_sft",
    )
    # training_set.json uses relative ms/si paths, so paper_root should usually
    # point to a Modal volume location that contains "all paper ft data/...".
    normalized = _rewrite_path_flag(
        normalized,
        "--paper_root",
        default_relative_if_missing="" if not is_eval else None,
    )
    normalized = _rewrite_path_flag(normalized, "--cache_prompts_path")
    # training_qwen.jsonl is mounted with the project source tree, not in /vol.
    normalized = _rewrite_project_path_flag(normalized, "--training_qwen_path")
    return normalized


def _get_flag_value(args: tuple[str, ...] | list[str], flag: str, default: str) -> str:
    tokens = list(args)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == flag and i + 1 < len(tokens):
            return tokens[i + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
        i += 1
    return default


def _get_flag_int(args: tuple[str, ...] | list[str], flag: str, default: int) -> int:
    value = _get_flag_value(args, flag, str(default))
    try:
        return int(value)
    except ValueError:
        return default


def _normalize_bundle_args(args: tuple[str, ...]) -> list[str]:
    normalized = list(args)
    normalized = _rewrite_path_flag(normalized, "--run_dir")
    normalized = _rewrite_path_flag(
        normalized,
        "--output_dir",
        default_relative_if_missing="submissions/hw4_gradescope_submission",
    )
    return normalized


def _is_wandb_enabled_for_train_args(args: tuple[str, ...] | list[str]) -> bool:
    # Mirror hw4.train argparse semantics:
    # - default is enabled
    # - --no-wandb_enabled disables
    # - --wandb_enabled enables
    enabled = True
    for token in args:
        if token == "--no-wandb_enabled":
            enabled = False
        elif token == "--wandb_enabled":
            enabled = True
    return enabled


def _assert_wandb_credentials_available_if_needed(args: tuple[str, ...] | list[str]) -> None:
    if not _is_wandb_enabled_for_train_args(args):
        return
    has_netrc = Path("/root/.netrc").is_file()
    has_api_key_env = bool(os.environ.get("WANDB_API_KEY"))
    if not has_netrc and not has_api_key_env:
        raise RuntimeError(
            "W&B logging is enabled for training, but no credentials were found in the Modal container. "
            "Run `uvx wandb login` locally (so ~/.netrc is copied), or export WANDB_API_KEY before modal run, "
            "or pass `--no-wandb_enabled`."
        )


def _run_subprocess_with_periodic_volume_commits(
    cmd: list[str], *, extra_env: dict[str, str] | None = None
) -> None:
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(cmd, cwd=PROJECT_DIR, env=env)
    returncode: int | None = None
    try:
        while returncode is None:
            try:
                returncode = proc.wait(timeout=DEFAULT_VOLUME_COMMIT_INTERVAL_SECONDS)
            except subprocess.TimeoutExpired:
                volume.commit()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        volume.commit()

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .uv_sync(extras=["remote"])
)

# Ensure CUDA-enabled torch inside the uv venv for H100 runs.
image = image.run_commands(
    "/.uv/uv pip install --python /.uv/.venv/bin/python --index-url https://download.pytorch.org/whl/cu124 'torch>=2.5,<2.7'"
)

if NETRC_PATH.is_file():
    image = image.add_local_file(
        NETRC_PATH,
        remote_path="/root/.netrc",
        copy=True,
    )

image = image.add_local_dir(
    ".",
    remote_path=PROJECT_DIR,
    ignore=load_gitignore_patterns(),
)

# training_qwen.jsonl is intentionally gitignored, but GRPO needs it inside the
# Modal image, so copy it explicitly the same way we do for adapter weights.
training_qwen_path = Path("training_qwen.jsonl")
if training_qwen_path.is_file():
    image = image.add_local_file(
        training_qwen_path,
        remote_path=f"{PROJECT_DIR}/training_qwen.jsonl",
        copy=False,
    )

# .gitignore excludes *.safetensors (and tokenizer.json), but PPO may need these
# adapter assets under /root/project/final_adapter/final_adapter.
local_adapter_dir = Path("final_adapter/final_adapter")
adapter_weight = local_adapter_dir / "adapter_model.safetensors"
adapter_weight_bin = local_adapter_dir / "adapter_model.bin"
adapter_tokenizer_json = local_adapter_dir / "tokenizer.json"
if adapter_weight.is_file():
    image = image.add_local_file(
        adapter_weight,
        remote_path=f"{PROJECT_DIR}/final_adapter/final_adapter/adapter_model.safetensors",
        copy=False,
    )
if adapter_weight_bin.is_file():
    image = image.add_local_file(
        adapter_weight_bin,
        remote_path=f"{PROJECT_DIR}/final_adapter/final_adapter/adapter_model.bin",
        copy=False,
    )
if adapter_tokenizer_json.is_file():
    image = image.add_local_file(
        adapter_tokenizer_json,
        remote_path=f"{PROJECT_DIR}/final_adapter/final_adapter/tokenizer.json",
        copy=False,
    )

app = modal.App(APP_NAME)

function_secrets = []
if os.environ.get("WANDB_API_KEY"):
    function_secrets.append(modal.Secret.from_dict({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]}))

env = {
    "PYTHONPATH": PROJECT_DIR,
    "PYTHONUNBUFFERED": "1",
    "WANDB_DIR": f"{VOLUME_PATH}/wandb",
    "HF_HOME": f"{VOLUME_PATH}/hf",
    "HF_DATASETS_CACHE": f"{VOLUME_PATH}/hf/datasets",
}


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    env=env,
    image=image,
    secrets=function_secrets,
    gpu=DEFAULT_GPU,
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
)
def train_remote(*args: str) -> None:
    normalized_args = _normalize_modal_args(args, is_eval=False)
    _assert_wandb_credentials_available_if_needed(normalized_args)
    cmd = [
        "torchrun", "--nproc_per_node", "4",
        f"{PROJECT_DIR}/sft_train.py", *normalized_args,
    ]
    _run_subprocess_with_periodic_volume_commits(cmd)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    env=env,
    image=image,
    secrets=function_secrets,
    gpu=DEFAULT_GPU,
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
)
def eval_remote(*args: str) -> None:
    normalized_args = _normalize_modal_args(args, is_eval=True)
    cmd = ["python", "-u", f"{PROJECT_DIR}/sft_eval.py", *normalized_args]
    _run_subprocess_with_periodic_volume_commits(cmd)


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60 * 30,
    env=env,
    image=image,
    cpu=2.0,
    memory=4096,
)
def bundle_submission_remote(*args: str) -> None:
    normalized_args = _normalize_bundle_args(args)
    cmd = ["python", "-u", "-m", "hw4.gradescope_bundle", *normalized_args]
    _run_subprocess_with_periodic_volume_commits(cmd)

@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    env=env,
    image=image,
    secrets=function_secrets,
    gpu=DEFAULT_PPO_GPU,
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
)
def ppo_remote(*args: str) -> None:
    normalized_args = _normalize_ppo_args(args, is_eval=False)
    _assert_wandb_credentials_available_if_needed(normalized_args)
    cmd = ["python3", "-u", f"{PROJECT_DIR}/ppo_train.py", *normalized_args]
    _run_subprocess_with_periodic_volume_commits(
        cmd,
        extra_env={
            "CUDA_VISIBLE_DEVICES": "0",
            "CUDA_LAUNCH_BLOCKING": "0",
        },
    )


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    env=env,
    image=image,
    secrets=function_secrets,
    gpu=DEFAULT_PPO_GPU,
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
)
def grpo_remote(*args: str) -> None:
    normalized_args = _normalize_grpo_args(args, is_eval=False)
    _assert_wandb_credentials_available_if_needed(normalized_args)
    cmd = ["python3", "-u", f"{PROJECT_DIR}/grpo_train.py", *normalized_args]
    _run_subprocess_with_periodic_volume_commits(
        cmd,
        extra_env={
            "CUDA_VISIBLE_DEVICES": "0",
            "CUDA_LAUNCH_BLOCKING": "0",
        },
    )


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    env=env,
    image=image,
    secrets=function_secrets,
    gpu=DEFAULT_PPO_GPU,
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
)
def ppo_eval_remote(*args: str) -> None:
    normalized_args = _normalize_ppo_args(args, is_eval=True)
    cmd = ["python3", "-u", f"{PROJECT_DIR}/ppo_eval.py", *normalized_args]
    _run_subprocess_with_periodic_volume_commits(
        cmd,
        extra_env={
            "CUDA_VISIBLE_DEVICES": "0",
        },
    )


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=DEFAULT_TIMEOUT_SECONDS,
    env=env,
    image=image,
    secrets=function_secrets,
    gpu=DEFAULT_PPO_GPU,
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
)
def grpo_eval_remote(*args: str) -> None:
    normalized_args = _normalize_grpo_args(args, is_eval=True)
    cmd = ["python3", "-u", f"{PROJECT_DIR}/grpo_eval.py", *normalized_args]
    _run_subprocess_with_periodic_volume_commits(
        cmd,
        extra_env={
            "CUDA_VISIBLE_DEVICES": "0",
        },
    )


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60 * 30,
    env=env,
    image=image,
    cpu=2.0,
    memory=4096,
)
def ppo_data_check_remote(*args: str) -> None:
    import json

    normalized_args = _normalize_ppo_args(args, is_eval=False)
    training_qwen_path = Path(_get_flag_value(normalized_args, "--training_qwen_path", ""))
    paper_root = Path(_get_flag_value(normalized_args, "--paper_root", VOLUME_PATH))
    training_set_path = Path(_get_flag_value(normalized_args, "--training_set_path", f"{PROJECT_DIR}/training_set.json"))
    sample_papers = _get_flag_int(normalized_args, "--sample_papers", 50)

    if str(training_qwen_path):
        print(f"[ppo-data-check] training_qwen_path={training_qwen_path}")
        if not training_qwen_path.is_file():
            raise RuntimeError(f"training_qwen_path does not exist: {training_qwen_path}")
        checked_rows = 0
        bad_rows = 0
        with training_qwen_path.open("r", encoding="utf-8") as f:
            for line in f:
                if checked_rows >= sample_papers:
                    break
                if not line.strip():
                    continue
                checked_rows += 1
                row = json.loads(line)
                messages = row.get("messages", [])
                if not isinstance(messages, list) or len(messages) < 3:
                    bad_rows += 1
                    continue
                roles = {m.get("role") for m in messages if isinstance(m, dict)}
                if "user" not in roles or "assistant" not in roles:
                    bad_rows += 1
        print(f"[ppo-data-check] rows checked: {checked_rows}")
        print(f"[ppo-data-check] malformed rows in sample: {bad_rows}")
        return

    print(f"[ppo-data-check] training_set_path={training_set_path}")
    print(f"[ppo-data-check] paper_root={paper_root}")
    print(f"[ppo-data-check] sample_papers={sample_papers}")

    if not training_set_path.is_file():
        raise RuntimeError(f"training_set_path does not exist: {training_set_path}")

    papers = json.loads(training_set_path.read_text(encoding="utf-8"))
    checked = 0
    with_context = 0
    missing_examples: list[str] = []

    for paper in papers:
        if checked >= sample_papers:
            break
        checked += 1
        ms_raw = str(paper.get("ms", "")).strip()
        si_raw = str(paper.get("si", "")).strip()

        has_context = False
        if ms_raw:
            ms_path = Path(ms_raw)
            if not ms_path.is_absolute():
                ms_path = paper_root / ms_path
            if ms_path.is_file():
                has_context = True
            elif len(missing_examples) < 8:
                missing_examples.append(f"missing ms: {ms_path}")

        if not has_context and si_raw:
            for line in si_raw.splitlines():
                rp = line.strip()
                if not rp:
                    continue
                si_path = Path(rp)
                if not si_path.is_absolute():
                    si_path = paper_root / si_path
                if si_path.is_file():
                    has_context = True
                    break
                if len(missing_examples) < 8:
                    missing_examples.append(f"missing si: {si_path}")

        with_context += int(has_context)

    print(f"[ppo-data-check] papers checked: {checked}")
    print(f"[ppo-data-check] papers with readable ms/si: {with_context}")
    if missing_examples:
        print("[ppo-data-check] sample missing paths:")
        for x in missing_examples:
            print(f"  - {x}")
    if with_context == 0:
        raise RuntimeError(
            "No readable ms/si files under paper_root. Put corpus under that root (typically /vol/all paper ft data/...) or change --paper_root."
        )


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60 * 30,
    env=env,
    image=image,
    cpu=2.0,
    memory=4096,
)
def grpo_data_check_remote(*args: str) -> None:
    import json

    normalized_args = _normalize_grpo_args(args, is_eval=False)
    training_qwen_path = Path(_get_flag_value(normalized_args, "--training_qwen_path", ""))
    paper_root = Path(_get_flag_value(normalized_args, "--paper_root", VOLUME_PATH))
    training_set_path = Path(_get_flag_value(normalized_args, "--training_set_path", f"{PROJECT_DIR}/training_set.json"))
    sample_papers = _get_flag_int(normalized_args, "--sample_papers", 50)

    if str(training_qwen_path):
        print(f"[grpo-data-check] training_qwen_path={training_qwen_path}")
        if not training_qwen_path.is_file():
            raise RuntimeError(f"training_qwen_path does not exist: {training_qwen_path}")
        checked_rows = 0
        bad_rows = 0
        with training_qwen_path.open("r", encoding="utf-8") as f:
            for line in f:
                if checked_rows >= sample_papers:
                    break
                if not line.strip():
                    continue
                checked_rows += 1
                row = json.loads(line)
                messages = row.get("messages", [])
                if not isinstance(messages, list) or len(messages) < 3:
                    bad_rows += 1
                    continue
                roles = {m.get("role") for m in messages if isinstance(m, dict)}
                if "user" not in roles or "assistant" not in roles:
                    bad_rows += 1
        print(f"[grpo-data-check] rows checked: {checked_rows}")
        print(f"[grpo-data-check] malformed rows in sample: {bad_rows}")
        return

    print(f"[grpo-data-check] training_set_path={training_set_path}")
    print(f"[grpo-data-check] paper_root={paper_root}")
    print(f"[grpo-data-check] sample_papers={sample_papers}")

    if not training_set_path.is_file():
        raise RuntimeError(f"training_set_path does not exist: {training_set_path}")

    papers = json.loads(training_set_path.read_text(encoding="utf-8"))
    checked = 0
    with_context = 0
    missing_examples: list[str] = []

    for paper in papers:
        if checked >= sample_papers:
            break
        checked += 1
        ms_raw = str(paper.get("ms", "")).strip()
        si_raw = str(paper.get("si", "")).strip()

        has_context = False
        if ms_raw:
            ms_path = Path(ms_raw)
            if not ms_path.is_absolute():
                ms_path = paper_root / ms_path
            if ms_path.is_file():
                has_context = True
            elif len(missing_examples) < 8:
                missing_examples.append(f"missing ms: {ms_path}")

        if not has_context and si_raw:
            for line in si_raw.splitlines():
                rp = line.strip()
                if not rp:
                    continue
                si_path = Path(rp)
                if not si_path.is_absolute():
                    si_path = paper_root / si_path
                if si_path.is_file():
                    has_context = True
                    break
                if len(missing_examples) < 8:
                    missing_examples.append(f"missing si: {si_path}")

        with_context += int(has_context)

    print(f"[grpo-data-check] papers checked: {checked}")
    print(f"[grpo-data-check] papers with readable ms/si: {with_context}")
    if missing_examples:
        print("[grpo-data-check] sample missing paths:")
        for x in missing_examples:
            print(f"  - {x}")
    if with_context == 0:
        raise RuntimeError(
            "No readable ms/si files under paper_root. Put corpus under that root (typically /vol/all paper ft data/...) or change --paper_root."
        )


@app.local_entrypoint()
def main(*args: str) -> None:
    """Default entrypoint: forward args to train_remote."""
    if _is_wandb_enabled_for_train_args(args) and not NETRC_PATH.is_file() and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "W&B logging is enabled (default), but no credentials were detected locally. "
            "Run `uvx wandb login` (creates ~/.netrc), or export WANDB_API_KEY before modal run, "
            "or pass `--no-wandb_enabled`."
        )
    train_remote.remote(*args)
