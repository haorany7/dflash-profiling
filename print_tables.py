"""Reprocess saved profile JSON to emit PTD/PTV tables (rich + LaTeX).

Usage:
    python print_tables.py results/h200nvl.json
    python print_tables.py --latex results/h200nvl.json
    # Per-depth normalisation (paper Fig.~2 alignment): divide PTD by target depth.
    python print_tables.py --depth 32 --latex results/h200nvl.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from rich.console import Console
from rich.table import Table

console = Console(width=300)


def load_results(path: str) -> Tuple[Dict, Dict, List[int], List[int]]:
    with open(path) as f:
        results = json.load(f)
    target_map = {
        (r["context_len"], r["block_size"]): r["latency_ms"]
        for r in results if r["type"] == "target"
    }
    draft_map = {
        (r["context_len"], r["block_size"]): r["latency_ms"]
        for r in results if r["type"] == "draft"
    }
    ctx_lens = sorted({r["context_len"] for r in results})
    block_sizes = sorted({r["block_size"] for r in results})
    return target_map, draft_map, ctx_lens, block_sizes


def print_rich(target_map, draft_map, ctx_lens, block_sizes, depth: int = 1):
    # Target latency
    table_t = Table(title="Target Decode Latency (ms)")
    table_t.add_column("context_len", style="cyan")
    for bs in block_sizes:
        table_t.add_column(f"bs={bs}", justify="right")
    for ctx in ctx_lens:
        row = [str(ctx)]
        for bs in block_sizes:
            t = target_map.get((ctx, bs))
            row.append(f"{t:.2f}" if t else "-")
        table_t.add_row(*row)
    console.print(table_t)

    # Draft latency
    table_d = Table(title="Draft Forward Latency (ms)")
    table_d.add_column("context_len", style="cyan")
    for bs in block_sizes:
        table_d.add_column(f"bs={bs}", justify="right")
    for ctx in ctx_lens:
        row = [str(ctx)]
        for bs in block_sizes:
            d = draft_map.get((ctx, bs))
            row.append(f"{d:.2f}" if d else "-")
        table_d.add_row(*row)
    console.print(table_d)

    # Speedup ratio (target / draft)
    table_r = Table(title="Speedup Ratio (Target / Draft)")
    table_r.add_column("context_len", style="cyan")
    for bs in block_sizes:
        table_r.add_column(f"bs={bs}", justify="right")
    for ctx in ctx_lens:
        row = [str(ctx)]
        for bs in block_sizes:
            t = target_map.get((ctx, bs))
            d = draft_map.get((ctx, bs))
            if t and d and d > 0:
                ratio = t / d
                style = "green" if ratio > 5 else "yellow" if ratio > 2 else "red"
                row.append(f"[{style}]{ratio:.1f}x[/{style}]")
            else:
                row.append("-")
        table_r.add_row(*row)
    console.print(table_r)

    # PTD / PTV percentage (paper appendix style, block-level)
    table_p = Table(title="c_block = PTD / PTV (%) -- whole-block drafting cost / verification cost")
    table_p.add_column("context_len", style="cyan")
    for bs in block_sizes:
        table_p.add_column(f"bs={bs}", justify="right")
    for ctx in ctx_lens:
        row = [str(ctx)]
        for bs in block_sizes:
            t = target_map.get((ctx, bs))
            d = draft_map.get((ctx, bs))
            if t and d and t > 0:
                pct = d / t * 100
                style = "green" if pct < 20 else "yellow" if pct < 30 else "red"
                row.append(f"[{style}]{pct:.1f}%[/{style}]")
            else:
                row.append("-")
        table_p.add_row(*row)
    console.print(table_p)

    # Per-depth normalised c (Fig.~2 alignment): c_block / depth
    if depth and depth > 1:
        table_n = Table(title=f"c = c_block / depth (%, depth={depth}) -- per-depth drafting cost ratio (Fig.~2)")
        table_n.add_column("context_len", style="cyan")
        for bs in block_sizes:
            table_n.add_column(f"bs={bs}", justify="right")
        for ctx in ctx_lens:
            row = [str(ctx)]
            for bs in block_sizes:
                t = target_map.get((ctx, bs))
                d = draft_map.get((ctx, bs))
                if t and d and t > 0:
                    pct = (d / t * 100) / depth
                    style = "green" if pct < 1 else "yellow" if pct < 2 else "red"
                    row.append(f"[{style}]{pct:.3f}%[/{style}]")
                else:
                    row.append("-")
            table_n.add_row(*row)
        console.print(table_n)


def print_latex(target_map, draft_map, ctx_lens, block_sizes, source_path: str, depth: int = 1):
    print("% Auto-generated from", source_path)
    print(r"\begin{table}[h]")
    print(r"    \centering")
    if depth and depth > 1:
        print(rf"    \caption{{Per-depth drafting cost ratio $c = (\mathrm{{PTD}}/\mathrm{{PTV}}) / D$ (\%) on a single H200 NVL, with depth $D{{=}}{depth}$ (target layer count). Aligns with the $c$ used in Fig.~2's analytical speedup model. Lower is better.}}")
        print(r"    \label{tab:c-per-depth}")
    else:
        print(r"    \caption{Whole-block drafting cost ratio $c_{block} = \mathrm{PTD}/\mathrm{PTV}$ (\%) on a single H200 NVL. Lower is better.}")
        print(r"    \label{tab:c-block}")
    print(r"    \small")
    cols = "r|" + "r" * len(block_sizes)
    print(r"    \begin{tabular}{" + cols + r"}")
    print(r"        \toprule")
    header = " & ".join([r"$L \backslash \gamma$"] + [rf"${bs}$" for bs in block_sizes])
    print("        " + header + r" \\")
    print(r"        \midrule")
    for ctx in ctx_lens:
        cells = [str(ctx)]
        for bs in block_sizes:
            t = target_map.get((ctx, bs))
            d = draft_map.get((ctx, bs))
            if t and d and t > 0:
                pct = d / t * 100
                if depth and depth > 1:
                    pct = pct / depth
                    cells.append(f"{pct:.3f}")
                else:
                    cells.append(f"{pct:.1f}")
            else:
                cells.append("-")
        print("        " + " & ".join(cells) + r" \\")
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table}")


def main():
    parser = argparse.ArgumentParser(description="Reprocess profile JSON")
    parser.add_argument("json_path", type=str, help="Path to profile JSON output")
    parser.add_argument("--latex", action="store_true",
                        help="Emit LaTeX table (in addition to rich tables)")
    parser.add_argument("--depth", type=int, default=1,
                        help="Target depth (number of target layers) used to "
                             "normalise c into per-depth cost matching Fig.~2's "
                             "analytical model. Defaults to 1 (no normalisation). "
                             "Llama-3-8B = 32, Qwen2.5-7B = 28, etc.")
    args = parser.parse_args()

    if not Path(args.json_path).exists():
        raise FileNotFoundError(f"JSON not found: {args.json_path}")

    target_map, draft_map, ctx_lens, block_sizes = load_results(args.json_path)
    print_rich(target_map, draft_map, ctx_lens, block_sizes, depth=args.depth)
    if args.latex:
        console.print("\n[bold]LaTeX table:[/bold]")
        print_latex(target_map, draft_map, ctx_lens, block_sizes, args.json_path, depth=args.depth)


if __name__ == "__main__":
    main()
