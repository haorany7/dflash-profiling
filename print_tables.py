"""Reprocess saved profile JSON to emit PTD/PTV tables (rich + LaTeX).

Usage:
    python print_tables.py results/h200nvl.json
    python print_tables.py --latex results/h200nvl.json

Reports two cost ratios:
    c_block = draft_latency / target_latency               (whole-block)
    c       = c_block / gamma                              (per-draft-token,
                                                            aligns with Fig.~2)
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


def print_rich(target_map, draft_map, ctx_lens, block_sizes):
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

    # Per-draft-token c (Fig.~2 alignment): c_block / gamma (column-wise)
    table_n = Table(title="c = c_block / gamma (%) -- per-draft-token cost ratio (aligns with Fig.~2)")
    table_n.add_column("context_len", style="cyan")
    for bs in block_sizes:
        table_n.add_column(f"bs={bs}", justify="right")
    for ctx in ctx_lens:
        row = [str(ctx)]
        for bs in block_sizes:
            t = target_map.get((ctx, bs))
            d = draft_map.get((ctx, bs))
            if t and d and t > 0 and bs > 0:
                pct = (d / t * 100) / bs
                style = "green" if pct < 0.5 else "yellow" if pct < 2 else "red"
                row.append(f"[{style}]{pct:.3f}%[/{style}]")
            else:
                row.append("-")
        table_n.add_row(*row)
    console.print(table_n)


def print_latex(target_map, draft_map, ctx_lens, block_sizes, source_path: str):
    print("% Auto-generated from", source_path)
    print(r"\begin{table}[h]")
    print(r"    \centering")
    print(r"    \caption{Per-draft-token drafting cost ratio "
          r"$c = (\mathrm{PTD}/\mathrm{PTV}) / \gamma$ (\%) on a single H200 NVL, "
          r"sweeping context length $L$ and draft length $\gamma$. "
          r"This is the $c$ used in Fig.~2's analytical speedup model. "
          r"Lower is better.}")
    print(r"    \label{tab:c-per-token}")
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
            if t and d and t > 0 and bs > 0:
                pct = (d / t * 100) / bs
                cells.append(f"{pct:.3f}")
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
    args = parser.parse_args()

    if not Path(args.json_path).exists():
        raise FileNotFoundError(f"JSON not found: {args.json_path}")

    target_map, draft_map, ctx_lens, block_sizes = load_results(args.json_path)
    print_rich(target_map, draft_map, ctx_lens, block_sizes)
    if args.latex:
        console.print("\n[bold]LaTeX table:[/bold]")
        print_latex(target_map, draft_map, ctx_lens, block_sizes, args.json_path)


if __name__ == "__main__":
    main()
