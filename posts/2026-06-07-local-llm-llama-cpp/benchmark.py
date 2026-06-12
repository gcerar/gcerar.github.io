from __future__ import annotations

import contextlib
import itertools as it
import json
import logging
import os
import re
import shlex
import signal
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Final

WORKDIR = Path(__file__).resolve().parent


BASE_URL: Final[str] = "http://127.0.0.1:8080"
CONTAINER_NAME: Final[str] = "llama-bench-server"
DEFAULT_IMAGE_VERSION: Final[str] = "b9544"
DEFAULT_IMAGE_CUDA_VERSION: Final[str] = "cuda"
DEFAULT_IMAGE_TAG: Final[str] = f"full-{DEFAULT_IMAGE_CUDA_VERSION}-{DEFAULT_IMAGE_VERSION}"
# DEFAULT_OUTPUT: Final[Path] = Path(f"llamacpp_context_benchmark.{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl")
LOGGER: Final[logging.Logger] = logging.getLogger("llamacpp_benchmark")


DEFAULT_DEPTHS: tuple[int, ...] = tuple(2**i for i in range(12, 19))

BENCH_ONLY_FLAGS: Final[set[str]] = {
    "--spec-type",
    "--spec-draft-n-max",
    "-ctkd",
    "-ctvd",
}

FIT_ONLY_FLAGS: Final[set[str]] = {
    # auto fit parameters
    "-fit",
    "--fit",
    # offloading all expert router to CPU
    "-cmoe",
    "--cpu-moe",
    # predefine context size
    "-c",
    "--ctx-size",
    # reserve space during fit;
    # also causes bug in llama-bench and ignores `-ot`
    "-fitt",
    # KV offload
    "-nkvo",
    "-kvo",
}

FLAG_ALIASES: Final[dict[str, str]] = {
    "--flash-attn": "-fa",
    "--cache-type-k": "-ctk",
    "--cache-type-v": "-ctv",
    "--ctx-size": "-c",
    "--gpu-layers": "-ngl",
    "--n-gpu-layers": "-ngl",
    "--override-tensor": "-ot",
    "--cpu-moe": "-cmoe",
    "--fit": "-fit",
    "--fit-target": "-fitt",
    "--kv-offload": "-kvo",
    "--no-kv-offload": "-nkvo",
    "--n-cpu-moe": "-ncmoe",
}


ArgValue = str | int | float | bool | None
CacheKey = tuple[int, str, tuple[tuple[str, str | None], ...]]


@dataclass(frozen=True)
class LlamaArg:
    flag: str
    value: ArgValue = None

    @property
    def canonical_flag(self) -> str:
        return FLAG_ALIASES.get(self.flag, self.flag)

    def tokens(self) -> list[str]:
        if self.value is None:
            return [self.flag]
        if isinstance(self.value, bool):
            return [self.flag] if self.value else []
        return [self.flag, str(self.value)]


@dataclass(frozen=True)
class LlamaArgs:
    args: tuple[LlamaArg, ...] = ()

    def to_list(self) -> list[list[str]]:
        return [arg.tokens() for arg in self.args]

    def canonical_items(self) -> tuple[tuple[str, str | None], ...]:
        return tuple((arg.canonical_flag, None if arg.value is None else str(arg.value)) for arg in self.args)

    def cache_items(self) -> tuple[tuple[str, str | None], ...]:
        items: dict[str, str | None] = {}

        for arg in self.args:
            items[arg.canonical_flag] = None if arg.value is None else str(arg.value)

        return tuple(sorted(items.items()))

    def tokens(self) -> list[str]:
        return [token for arg in self.args for token in arg.tokens()]

    @classmethod
    def from_str(cls, text: str) -> LlamaArgs:
        tokens = text.strip().split()
        return cls.parse_tokens(tokens)

    @classmethod
    def parse_tokens(cls, params: Iterable[str]) -> LlamaArgs:
        tokens = list(params)
        args: list[LlamaArg] = []
        i: int = 0

        while i < len(tokens):
            flag = tokens[i]

            if not flag.startswith("-"):
                i += 1
                continue

            next_token = tokens[i + 1] if i + 1 < len(tokens) else None

            has_value = next_token is not None and (
                not next_token.startswith("-") or re.fullmatch(r"-?\d+(\.\d+)?", next_token) is not None
            )

            if has_value:
                args.append(LlamaArg(flag, next_token))
                i += 2
            else:
                args.append(LlamaArg(flag))
                i += 1

        return cls(tuple(args))

    def merge(self, *others: LlamaArgs) -> LlamaArgs:
        merged: dict[str, LlamaArg] = {}

        for source in (self, *others):
            for arg in source.args:
                merged[arg.canonical_flag] = arg

        return LlamaArgs(args=tuple(merged.values()))

    def for_bench(self) -> LlamaArgs:
        return LlamaArgs(tuple(adapted for arg in self.args if (adapted := adapt_arg_for_bench(arg)) is not None))

    def for_fit(self) -> LlamaArgs:
        return LlamaArgs(tuple(adapted for arg in self.args if (adapted := adapt_arg_for_fit(arg)) is not None))


def normalize_name(case: BaseCase | FitCase | BenchCase) -> str:
    def normalize_params(args: LlamaArgs) -> str:
        tokens = []
        for arg in args.args:
            token = f"{arg.flag}={arg.value}" if arg.value else arg.flag
            tokens.append(token)

        tokens = sorted(tokens)
        tokens = " ".join(tokens)
        return tokens

    if isinstance(case, BaseCase):
        return f"{case.hf_model_ref} {normalize_params(case.llama_args)}"

    if isinstance(case, FitCase):
        return f"{case.base_case.hf_model_ref} {normalize_params(case.args)}"

    if isinstance(case, BenchCase):
        return f"{case.base_case.hf_model_ref} {normalize_params(case.args)}"


@dataclass(frozen=True)
class BaseCase:
    hf_model: str
    max_context: int
    quantization: str | None = None
    notes: str = ""
    llama_args: LlamaArgs = field(default_factory=LlamaArgs)
    env: dict[str, str] = field(default_factory=dict)
    image_tag: str = DEFAULT_IMAGE_TAG

    @property
    def hf_model_ref(self) -> str:
        if self.quantization is None or ":" in self.hf_model:
            return self.hf_model

        return f"{self.hf_model}:{self.quantization}"


@dataclass(frozen=True)
class FitCase:
    base_case: BaseCase
    args: LlamaArgs = field(default_factory=LlamaArgs)

    @property
    def context_size(self) -> int:
        ctx_len = self.base_case.max_context
        for arg in self.args.args:
            if arg.canonical_flag == "-c":
                value = str(arg.value)
                if re.fullmatch(r"-?\d+", value):
                    parsed_ctx_len = int(value)
                    if parsed_ctx_len > 0:
                        ctx_len = parsed_ctx_len

        return ctx_len


@dataclass(frozen=True)
class BenchCase:
    fit_case: FitCase
    args: LlamaArgs = field(default_factory=LlamaArgs)
    benchmarks: dict[str, Any] = field(default_factory=dict)

    @property
    def base_case(self) -> BaseCase:
        return self.fit_case.base_case

    @property
    def context_size(self) -> int:
        return self.fit_case.context_size


def build_number_from_image_version(version: str) -> int:
    match = re.fullmatch(r"b?(\d+)", version)
    if match is None:
        msg = f"Cannot parse build number from image version: {version!r}"
        raise ValueError(msg)

    return int(match.group(1))


def build_number_from_image_tag(image_tag: str) -> int:
    version = image_tag.rsplit("-", maxsplit=1)[-1]
    return build_number_from_image_version(version)


def llama_args_from_lists(items: Iterable[Iterable[str]]) -> LlamaArgs:
    return LlamaArgs(
        tuple(LlamaArg(tokens[0], tokens[1] if len(tokens) > 1 else None) for item in items if (tokens := list(item)))
    )


def make_cache_key(base_case: BaseCase) -> CacheKey:
    return (
        build_number_from_image_tag(base_case.image_tag),
        base_case.hf_model_ref,
        base_case.llama_args.cache_items(),
    )


def cache_key_from_row(row: dict[str, Any]) -> CacheKey | None:
    perf = row.get("perf")
    if not isinstance(perf, list) or not perf:
        return None

    first_result = perf[0]
    if not isinstance(first_result, dict):
        return None

    build_number = first_result.get("build_number")
    hf_model = row.get("hf_model")
    fit_params = row.get("params", {}).get("fit")

    if not isinstance(build_number, int) or not isinstance(hf_model, str) or not isinstance(fit_params, list):
        return None

    return (build_number, hf_model, llama_args_from_lists(fit_params).cache_items())


def make_result_row(
    base_case: BaseCase, fit_case: FitCase, bench_case: BenchCase, results: list[dict[str, Any]]
) -> dict[str, Any]:
    fit_input_args = base_case.llama_args.for_fit()

    return {
        "hf_model": base_case.hf_model_ref,
        "hf_repo": base_case.hf_model,
        "quantization": base_case.quantization,
        "notes": base_case.notes,
        "context_size": fit_case.context_size,
        "base_params": shlex.join(base_case.llama_args.tokens()),
        "fit_params": f"-hf {base_case.hf_model_ref} {shlex.join(fit_input_args.tokens())}",
        "fit_result_params": shlex.join(fit_case.args.tokens()),
        "bench_params": f"-hf {base_case.hf_model_ref} {shlex.join(bench_case.args.tokens())}",
        "params": {
            "fit": fit_input_args.to_list(),
            "bench": bench_case.args.to_list(),
        },
        "perf": results,
    }


@dataclass
class BenchmarkCache:
    path: Path
    rows: list[dict[str, Any]] = field(default_factory=list)
    index: dict[CacheKey, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> BenchmarkCache:
        if not path.exists():
            return cls(path=path)

        with path.open() as fp:
            loaded_rows = json.load(fp)

        if not isinstance(loaded_rows, list):
            msg = f"Expected benchmark cache to be a JSON array: {path}"
            raise ValueError(msg)

        cache = cls(path=path, rows=loaded_rows)
        cache.rebuild_index()
        return cache

    def rebuild_index(self) -> None:
        self.index.clear()
        for row in self.rows:
            key = cache_key_from_row(row)
            if key is not None:
                self.index.setdefault(key, row)

    def find(self, key: CacheKey) -> dict[str, Any] | None:
        return self.index.get(key)

    def add(self, key: CacheKey, row: dict[str, Any]) -> None:
        if key in self.index:
            return

        self.rows.append(row)
        self.index[key] = row

    def save(self) -> None:
        with self.path.open(mode="w") as fp:
            json.dump(self.rows, fp, indent=2)


def safe_killpg(proc: subprocess.Popen, sig: signal.Signals) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except ProcessLookupError as exc:
        print(exc)
        pass

    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, sig)


def _run_shell_command(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    try:
        res = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=True,
            timeout=timeout,
            start_new_session=True,
        )

        return res

    except subprocess.TimeoutExpired as err:
        # subprocess.run already killed/waited for the direct child.
        raise RuntimeError(
            f"Command timed out after {timeout}s\n\nSTDOUT:\n{err.stdout}\n\nSTDERR:\n{err.stderr}"
        ) from err

    except KeyboardInterrupt:
        # subprocess.run does not give you proc.pid here,
        # so you cannot easily do custom process-group cleanup.
        raise

    except subprocess.CalledProcessError as err:
        LOGGER.error(err.cmd)
        LOGGER.error(err.stdout)
        LOGGER.error(err.stderr)
        raise

    finally:
        _stop_container()


def _stop_container(timeout: int = 10) -> None:
    cmd_stop: Final[list[str]] = ["docker", "stop", "--timeout", str(timeout), CONTAINER_NAME]
    subprocess.run(cmd_stop, check=False)


def _run_text_command(cmd: list[str], timeout: float | None = None) -> str:
    proc = _run_shell_command(cmd, timeout=timeout)
    return proc.stdout


def _run_json_command(cmd: list[str], timeout: float | None = None) -> list[dict[str, Any]]:
    proc = _run_shell_command(cmd, timeout=timeout)
    return json.loads(proc.stdout)


def run_llama_fit_params(case: BaseCase) -> FitCase:
    """This runs `llama-fit-params` with given case's parameters and finds out"""
    docker_envs = [item for key, value in case.env.items() for item in ("-e", f"{key}={value}")]
    fit_input_args = case.llama_args.for_fit()

    cmd: list[str] = [
        "docker",
        "run",
        "--name",
        CONTAINER_NAME,
        "--rm",
        "--memory=32G",
        "--gpus",
        "all",
        "-v",
        "coding-agents_models:/models",
        "-e",
        "LLAMA_CACHE=/models",
        *docker_envs,
        "--entrypoint",
        "/app/llama-fit-params",
        f"ghcr.io/ggml-org/llama.cpp:{case.image_tag}",
        "-hf",
        case.hf_model_ref,
        *fit_input_args.tokens(),
    ]

    LOGGER.debug(f"COMMAND: `{shlex.join(cmd)}`")

    text = _run_text_command(cmd)
    params_line = text.strip()
    tokens = shlex.split(params_line)
    fit_args = LlamaArgs.parse_tokens(tokens)

    return FitCase(base_case=case, args=fit_args)


def run_context_benchmark(case: BenchCase, lengths: Iterable[int] = DEFAULT_DEPTHS) -> list[dict[str, Any]]:
    docker_envs = [item for key, value in case.base_case.env.items() for item in ("-e", f"{key}={value}")]

    max_context = case.fit_case.context_size
    usable_lengths: list[int] = [i for i in lengths if i < max_context]
    usable_lengths += [max_context]

    cmd: Final[list[str]] = [
        "docker",
        "run",
        "--name",
        CONTAINER_NAME,
        "--rm",
        "--gpus",
        "all",
        "-v",
        "coding-agents_models:/models",
        "-e",
        "LLAMA_CACHE=/models",
        *docker_envs,
        "--entrypoint",
        "/app/llama-bench",
        f"ghcr.io/ggml-org/llama.cpp:{case.base_case.image_tag}",
        "-o",
        "json",
        "-hf",
        case.base_case.hf_model_ref,
        *case.args.tokens(),
        "-d",
        ",".join(map(str, usable_lengths)),
        "-p",
        "0",
        "--mmap",
        "0",
        "--repetitions",
        "1",
        "--delay",
        "5",
        "--numa",
        "numactl",
    ]

    return _run_json_command(cmd)


def adapt_arg_for_fit(arg: LlamaArg) -> LlamaArg | None:
    canonical_flag = arg.canonical_flag

    # flags not present in bench
    if canonical_flag in BENCH_ONLY_FLAGS:
        return None

    return arg


def adapt_arg_for_bench(arg: LlamaArg) -> LlamaArg | None:
    canonical_flag = arg.canonical_flag

    # flags not present in bench
    if canonical_flag in FIT_ONLY_FLAGS:
        return None

    # bench: -1 is invalid; replace with high number
    if canonical_flag == "-ngl" and str(arg.value) == "-1":
        return LlamaArg(arg.flag, 999)

    if canonical_flag == "-ot" and isinstance(arg.value, str):
        # fit-params uses `,` to specify layers,
        # but in bench, `,` uses hparam separator.
        return LlamaArg(arg.flag, arg.value.replace(",", ";"))

    if canonical_flag == "-nkvo":
        arg = LlamaArg(arg.flag, "1")

    # TODO: Not optimal
    value = str(arg.value).lower() if arg.value is not None else "1"
    return LlamaArg(arg.flag, {"on": "1", "off": "0", "auto": "1"}.get(value, arg.value))


_64K: Final[int] = 65_536
_128K: Final[int] = 131_072
_256K: Final[int] = 262_144

# model matrix with cherry picked models for RTX 3090, 24GB VRAM
MODEL_MATRIX: Final[list[BaseCase]] = [
    *[
        # Hybrid; Q6 won't fit into memory
        BaseCase(
            hf_model="unsloth/Qwen3.6-35B-A3B-GGUF",
            quantization="UD-Q6_K_XL",
            max_context=_256K,
            llama_args=LlamaArgs.from_str(f"-fa on -ctk {q} -ctv {q} -fit on -c {c}"),
        )
        for q, c in it.product({"f16", "bf16", "q8_0"}, {_128K, _256K})
    ],
    *[
        # Hybrid; Q5 would fit, but with a small context
        BaseCase(
            hf_model="unsloth/Qwen3.6-35B-A3B-GGUF",
            quantization="UD-Q5_K_XL",
            max_context=_256K,
            llama_args=LlamaArgs.from_str(f"-fa on -ctk {q} -ctv {q} -fit on -c {c}"),
        )
        for q, c in it.product({"f16", "bf16", "q8_0"}, {_128K, _256K})
    ],
    *[
        # Q4, will fit; let's see to where it goes
        BaseCase(
            hf_model="unsloth/Qwen3.6-35B-A3B-GGUF",
            quantization="UD-Q4_K_XL",
            max_context=_256K,
            llama_args=LlamaArgs.from_str(f"-fa on -ctk {q} -ctv {q} -fit off"),
        )
        for q in {"f16", "bf16", "q8_0"}
    ],
    *[
        # hybrid offloads
        BaseCase(
            hf_model="unsloth/Qwen3.6-35B-A3B-GGUF",
            quantization="UD-Q4_K_XL",
            max_context=_256K,
            llama_args=LlamaArgs.from_str(f"-fa on -ctk {q} -ctv {q} -fit on -c {c}"),
        )
        for q, ctxs in {("f16", (_128K, _256K)), ("bf16", (_128K, _256K)), ("q8_0", (_256K,))}
        for c in ctxs
    ],
    *[
        # Unload all MoE to the CPU.
        BaseCase(
            hf_model="unsloth/Qwen3.6-35B-A3B-GGUF",
            quantization="UD-Q4_K_XL",
            max_context=_256K,
            llama_args=LlamaArgs.from_str(f"-fa on -ctk {q} -ctv {q} -fit off -cmoe"),
        )
        for q in {"f16", "bf16", "q8_0"}
    ],
    # Qwen3.6 27B
    *[
        # Unload all MoE to the CPU.
        BaseCase(
            hf_model="unsloth/Qwen3.6-27B-GGUF",
            quantization="UD-Q4_K_XL",
            max_context=_256K,
            llama_args=LlamaArgs.from_str(f"-fa on -ctk {q} -ctv {q} -fit off"),
        )
        for q in {"f16", "bf16", "q8_0"}
    ],
    # Gemma4 MoE
    *[
        BaseCase(
            hf_model="unsloth/gemma-4-26B-A4B-it-GGUF",
            quantization=quant,
            max_context=_256K,
            llama_args=LlamaArgs.from_str(f"-fa on -ctk {qkv} -ctv {qkv} -fit off"),
        )
        for quant, qkv in it.product({"UD-Q4_K_XL", "UD-Q5_K_XL"}, {"f16", "bf16", "q8_0"})
    ],
    *[
        BaseCase(
            hf_model="unsloth/gemma-4-26B-A4B-it-GGUF",
            quantization=quant,
            max_context=_256K,
            llama_args=LlamaArgs.from_str(f"-fa on -ctk {qkv} -ctv {qkv} -fit on -c {c}"),
        )
        for quant, qkv, c in it.product({"UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL"}, {"f16", "bf16", "q8_0"}, {_256K})
    ],
]


def main() -> None:
    cache_path = WORKDIR / f"results-{DEFAULT_IMAGE_VERSION}.json"
    cache = BenchmarkCache.load(cache_path)

    for base_case in MODEL_MATRIX:
        cache_key = make_cache_key(base_case)
        if cache.find(cache_key):
            LOGGER.info(f"SKIP cached: `-hf {base_case.hf_model_ref} {shlex.join(base_case.llama_args.tokens())}`")
            continue

        # resolve parameters
        LOGGER.info(f"MODEL: `-hf {base_case.hf_model_ref} {shlex.join(base_case.llama_args.tokens())}`")
        fit_case = run_llama_fit_params(base_case)

        bench_args = base_case.llama_args.merge(fit_case.args).for_bench()

        LOGGER.info(f"MODEL: `-hf {fit_case.base_case.hf_model_ref} {shlex.join(bench_args.tokens())}`")
        bench_case = BenchCase(fit_case=fit_case, args=bench_args)
        results = run_context_benchmark(case=bench_case)

        row = make_result_row(base_case=base_case, fit_case=fit_case, bench_case=bench_case, results=results)
        cache.add(cache_key, row)
        cache.save()

    cache.save()


def format_benchmark(result: dict[str, Any]):
    return {"mean": result["avg_ts"], "stddev": result["stddev_ts"]}
    # return f"{result['avg_ts']:.1f} ± {result['stddev_ts']:.1f}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
