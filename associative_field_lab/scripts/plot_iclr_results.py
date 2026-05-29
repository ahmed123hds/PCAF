#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Set academic style parameters for publication-quality plots
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'savefig.bbox': 'tight',
    'savefig.dpi': 300
})

# Curated publication color palette
COLORS = {
    'transformer_dense':                '#434343',  # Dark Charcoal
    'linear_attention':                 '#5f5f5f',  # Kernel Attention Gray
    'local_transformer_w128':           '#7f7f7f',  # Medium Gray
    'global_local_transformer_w128_g16':'#adadad',  # Light Gray
    'local_conv':                       '#cc3333',  # Strong Red
    'pcaf_no_gate':                     '#e69138',  # Warm Orange
    'pcaf_semantic':                    '#3d85c8',  # Steel Blue
    'pcaf_hybrid':                      '#7b5ea7',  # Purple
    'pcaf_context':                     '#2e7d32',  # Deep Green (Proposed)
}

LABELS = {
    'transformer_dense':                'Dense Transformer',
    'linear_attention':                 'Linear Attention',
    'local_transformer_w128':           'Local Transformer (w=128)',
    'global_local_transformer_w128_g16':'Global-Local Trans. (w=128, g=16)',
    'local_conv':                       'Local Conv (No Cache)',
    'pcaf_no_gate':                     'PCAF No Gate (Fixed 0.5)',
    'pcaf_semantic':                    'PCAF Semantic Hash',
    'pcaf_hybrid':                      'PCAF Hybrid Hash',
    'pcaf_context':                     'PCAF Context Hash (Proposed)',
}

LINESTYLES = {
    'transformer_dense':                (0, (5, 2)),
    'linear_attention':                 (0, (4, 1, 1, 1)),
    'local_transformer_w128':           (0, (3, 2)),
    'global_local_transformer_w128_g16':(0, (1, 2)),
    'local_conv':                       '-',
    'pcaf_no_gate':                     '-',
    'pcaf_semantic':                    '-',
    'pcaf_hybrid':                      '-',
    'pcaf_context':                     '-',
}

LINEWIDTHS = {
    'transformer_dense':                1.5,
    'linear_attention':                 1.5,
    'local_transformer_w128':           1.5,
    'global_local_transformer_w128_g16':1.5,
    'local_conv':                       1.8,
    'pcaf_no_gate':                     1.8,
    'pcaf_semantic':                    1.8,
    'pcaf_hybrid':                      1.8,
    'pcaf_context':                     2.5,  # bold proposed
}

MARKERS = {
    'transformer_dense':                '^',
    'linear_attention':                 'P',
    'local_transformer_w128':           'v',
    'global_local_transformer_w128_g16':'>',
    'local_conv':                       'x',
    'pcaf_no_gate':                     'd',
    'pcaf_semantic':                    'o',
    'pcaf_hybrid':                      '*',
    'pcaf_context':                     's',
}

# Smoothing windows: higher for noisy small-batch attention models
SMOOTH_WINDOW = {
    'transformer_dense':                5,
    'linear_attention':                 5,
    'local_transformer_w128':           5,
    'global_local_transformer_w128_g16':5,
    'local_conv':                       3,
    'pcaf_no_gate':                     3,
    'pcaf_semantic':                    3,
    'pcaf_hybrid':                      3,
    'pcaf_context':                     3,
}

# Burn-in: skip early steps where PPL is extremely high and uninformative.
# Per-model, per-seqlen: (seq_len -> min_step_to_plot)
BURN_IN = {
    'transformer_dense':                {1024: 1000, 2048: 2000},
    'linear_attention':                 {1024: 1000, 2048: 2000},
    'local_transformer_w128':           {1024: 1000, 2048: 2000},
    'global_local_transformer_w128_g16':{1024: 1000, 2048: 2000},
    'local_conv':                       {1024: 1000, 2048: 1000},
    'pcaf_no_gate':                     {1024: 1000, 2048: 1000},
    'pcaf_semantic':                    {1024: 1000, 2048: 1000},
    'pcaf_hybrid':                      {1024: 1000, 2048: 1000},
    'pcaf_context':                     {1024: 1000, 2048: 1000},
}


def parse_jsonl(path: Path, min_step: int = 0):
    steps, ppl = [], []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = row.get('step', 0)
            if s < min_step:
                continue
            steps.append(s)
            ppl.append(row['eval_ppl'])
    return steps, ppl


def smooth(values, window: int = 3):
    """Centred moving average."""
    if len(values) < window:
        return values
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode='same')
    half = window // 2
    smoothed[:half] = arr[:half]
    smoothed[-half:] = arr[-half:]
    return smoothed.tolist()


def main():
    if len(sys.argv) != 3:
        print("Usage: plot_iclr_results.py <log_dir> <output_dir>")
        sys.exit(1)

    log_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [
        'transformer_dense',
        'linear_attention',
        'local_transformer_w128',
        'global_local_transformer_w128_g16',
        'local_conv',
        'pcaf_no_gate',
        'pcaf_semantic',
        'pcaf_hybrid',
        'pcaf_context',
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    for ax_idx, seq_len in enumerate([1024, 2048]):
        ax = axes[ax_idx]
        ax.set_title(f"Sequence Length $T = {seq_len}$", fontweight='bold', pad=10)

        for config_name in models:
            file_path = log_dir / f"{config_name}_seq{seq_len}.jsonl"
            if not file_path.exists():
                continue

            min_step = BURN_IN[config_name].get(seq_len, 1000)
            steps, ppl = parse_jsonl(file_path, min_step=min_step)
            if not steps:
                continue

            ppl_smooth = smooth(ppl, SMOOTH_WINDOW[config_name])

            # Marker every ~2000 training steps
            step_gap = steps[1] - steps[0] if len(steps) > 1 else 1000
            markevery = max(1, 2000 // step_gap)

            ax.plot(
                steps,
                ppl_smooth,
                color=COLORS[config_name],
                label=LABELS[config_name],
                linestyle=LINESTYLES[config_name],
                linewidth=LINEWIDTHS[config_name],
                marker=MARKERS[config_name],
                markersize=5 if config_name == 'pcaf_context' else 4,
                markevery=markevery,
                zorder=10 if config_name == 'pcaf_context' else 5,
                alpha=0.95,
            )

        ax.set_xlabel("Training Step", labelpad=8)
        if ax_idx == 0:
            ax.set_ylabel("Validation Perplexity (WikiText-103)", labelpad=8)

        ax.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(60, 700)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))

    # Shared legend — deduplicated across both subplots
    seen_labels: dict = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in seen_labels:
                seen_labels[label] = handle

    fig.legend(
        list(seen_labels.values()),
        list(seen_labels.keys()),
        loc='upper center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=True,
        facecolor='white',
        edgecolor='#cccccc',
        framealpha=0.95,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.87])

    pdf_path = output_dir / "iclr_baselines_curves.pdf"
    png_path = output_dir / "iclr_baselines_curves.png"

    plt.savefig(pdf_path, format='pdf')
    plt.savefig(png_path, format='png')
    plt.close()

    print(f"Successfully generated ICLR baseline plot at:\n - {pdf_path}\n - {png_path}")


if __name__ == "__main__":
    main()
