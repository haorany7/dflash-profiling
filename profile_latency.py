"""Profile target model and draft head latency across different sequence lengths.

Usage:
    python profile_latency.py \
        --model-name-or-path Qwen/Qwen3-8B \
        --draft-name-or-path z-lab/Qwen3-8B-DFlash-b16 \
        --context-lengths 128 256 512 1024 2048 4096 \
        --block-sizes 1 2 4 8 16 32 64 \
        --num-warmup 10 \
        --num-runs 50 \
        --output profile_results.json
"""

import argparse
import json
import time
from typing import Optional

import torch
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, DynamicCache

from dflash.model import DFlashDraftModel, extract_context_feature

console = Console(width=300)


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def profile_fn(fn, num_warmup: int = 10, num_runs: int = 50) -> float:
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    start = cuda_time()
    for _ in range(num_runs):
        fn()
    elapsed = cuda_time() - start
    return elapsed / num_runs * 1000  # ms


@torch.inference_mode()
def profile_target_decode(
    target: AutoModelForCausalLM,
    context_len: int,
    block_size: int,
    vocab_size: int,
    device: torch.device,
    num_warmup: int = 10,
    num_runs: int = 50,
) -> float:
    """Profile target model decode: verify `block_size` tokens with `context_len` KV cache."""
    input_ids = torch.randint(0, vocab_size, (1, context_len), device=device)
    kv_cache = DynamicCache()

    # Prefill to build KV cache
    target(
        input_ids,
        past_key_values=kv_cache,
        use_cache=True,
        logits_to_keep=1,
    )

    # Profile decode step: verify block_size tokens
    decode_ids = torch.randint(0, vocab_size, (1, block_size), device=device)
    position_ids = torch.arange(
        context_len, context_len + block_size, device=device
    ).unsqueeze(0)

    def run():
        # Clone cache to avoid growing it
        cache_copy = DynamicCache.from_legacy_cache(kv_cache.to_legacy_cache())
        target(
            decode_ids,
            position_ids=position_ids,
            past_key_values=cache_copy,
            use_cache=True,
            logits_to_keep=1,
        )

    return profile_fn(run, num_warmup, num_runs)


@torch.inference_mode()
def profile_draft_forward(
    draft_model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    context_len: int,
    block_size: int,
    vocab_size: int,
    device: torch.device,
    num_warmup: int = 10,
    num_runs: int = 50,
) -> float:
    """Profile draft head forward: generate `block_size` candidates with `context_len` context."""
    input_ids = torch.randint(0, vocab_size, (1, context_len), device=device)

    # Run target to get hidden states
    output = target(
        input_ids,
        use_cache=False,
        output_hidden_states=True,
    )
    target_hidden = extract_context_feature(
        output.hidden_states, draft_model.target_layer_ids
    )

    # Noise embedding (block_size tokens)
    noise_ids = torch.randint(0, vocab_size, (1, block_size), device=device)
    noise_embedding = target.model.embed_tokens(noise_ids)
    position_ids = torch.arange(
        context_len, context_len + block_size, device=device
    ).unsqueeze(0)

    # For draft, target_hidden is the full context, noise_embedding is the block
    # In actual inference, draft sees full context via target_hidden K/V
    # Here we simulate: position_ids covers [0, context_len + block_size)
    full_position_ids = torch.arange(
        context_len + block_size, device=device
    ).unsqueeze(0)

    def run():
        draft_output = draft_model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=full_position_ids,
            is_causal=False,
        )
        target.lm_head(draft_output[:, -block_size + 1 :, :])

    return profile_fn(run, num_warmup, num_runs)


def main():
    parser = argparse.ArgumentParser(description="Profile target vs draft latency")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096],
    )
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
    )
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=50)
    parser.add_argument("--output", type=str, default="profile_results.json")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    def has_flash_attn():
        try:
            import flash_attn
            return True
        except ImportError:
            return False

    attn_impl = "flash_attention_2" if has_flash_attn() else "sdpa"
    console.print(f"[bold]Attention implementation:[/bold] {attn_impl}")

    console.print("[bold]Loading target model...[/bold]")
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    vocab_size = target.config.vocab_size

    console.print("[bold]Loading draft model...[/bold]")
    draft_model = (
        DFlashDraftModel.from_pretrained(
            args.draft_name_or_path,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )

    console.print(
        f"\n[bold green]Profiling config:[/bold green] "
        f"context_lengths={args.context_lengths}, block_sizes={args.block_sizes}, "
        f"warmup={args.num_warmup}, runs={args.num_runs}\n"
    )

    results = []

    # --- Profile target decode latency (sweep context_len x block_size) ---
    table_target = Table(title="Target Model Decode Latency (ms)")
    table_target.add_column("context_len", style="cyan")
    for bs in args.block_sizes:
        table_target.add_column(f"bs={bs}", justify="right")

    for ctx_len in args.context_lengths:
        row = [str(ctx_len)]
        for bs in args.block_sizes:
            try:
                latency = profile_target_decode(
                    target, ctx_len, bs, vocab_size, device,
                    args.num_warmup, args.num_runs,
                )
                results.append({
                    "type": "target",
                    "context_len": ctx_len,
                    "block_size": bs,
                    "latency_ms": round(latency, 3),
                })
                row.append(f"{latency:.2f}")
            except torch.cuda.OutOfMemoryError:
                row.append("OOM")
                torch.cuda.empty_cache()
        table_target.add_row(*row)

    console.print(table_target)

    # --- Profile draft head latency (sweep context_len x block_size) ---
    table_draft = Table(title="Draft Head Forward Latency (ms)")
    table_draft.add_column("context_len", style="cyan")
    for bs in args.block_sizes:
        table_draft.add_column(f"bs={bs}", justify="right")

    for ctx_len in args.context_lengths:
        row = [str(ctx_len)]
        for bs in args.block_sizes:
            try:
                latency = profile_draft_forward(
                    draft_model, target, ctx_len, bs, vocab_size, device,
                    args.num_warmup, args.num_runs,
                )
                results.append({
                    "type": "draft",
                    "context_len": ctx_len,
                    "block_size": bs,
                    "latency_ms": round(latency, 3),
                })
                row.append(f"{latency:.2f}")
            except torch.cuda.OutOfMemoryError:
                row.append("OOM")
                torch.cuda.empty_cache()
        table_draft.add_row(*row)

    console.print(table_draft)

    # --- Print ratio table ---
    table_ratio = Table(title="Ratio (Target / Draft)")
    table_ratio.add_column("context_len", style="cyan")
    for bs in args.block_sizes:
        table_ratio.add_column(f"bs={bs}", justify="right")

    target_map = {
        (r["context_len"], r["block_size"]): r["latency_ms"]
        for r in results if r["type"] == "target"
    }
    draft_map = {
        (r["context_len"], r["block_size"]): r["latency_ms"]
        for r in results if r["type"] == "draft"
    }

    for ctx_len in args.context_lengths:
        row = [str(ctx_len)]
        for bs in args.block_sizes:
            t = target_map.get((ctx_len, bs))
            d = draft_map.get((ctx_len, bs))
            if t and d and d > 0:
                ratio = t / d
                style = "green" if ratio > 5 else "yellow" if ratio > 2 else "red"
                row.append(f"[{style}]{ratio:.1f}x[/{style}]")
            else:
                row.append("-")
        table_ratio.add_row(*row)

    console.print(table_ratio)

    # --- Save results ---
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[bold]Results saved to {args.output}[/bold]")


if __name__ == "__main__":
    main()
