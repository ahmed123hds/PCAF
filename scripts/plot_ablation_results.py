#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Set academic style parameters for publication-quality plots
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'savefig.bbox': 'tight',
    'savefig.dpi': 300
})

# Curated harmonious color palette matching premium aesthetics
COLORS = {
    'local_conv': '#e06666',      # Sleek soft red
    'pcaf_no_gate': '#f6b26b',    # Soft warm orange
    'pcaf_semantic': '#9fc5e8',   # Elegant soft blue
    'pcaf_context': '#6aa84f'     # Robust forest green
}

LABELS = {
    'local_conv': 'Local Conv (No Cache)',
    'pcaf_no_gate': 'PCAF No Gate (Fixed 0.5)',
    'pcaf_semantic': 'PCAF Semantic Hash',
    'pcaf_context': 'PCAF Context Hash (Proposed)'
}

MARKERS = {
    'local_conv': 'x',
    'pcaf_no_gate': 'd',
    'pcaf_semantic': 'o',
    'pcaf_context': 's'
}

def parse_jsonl(path: Path):
    steps = []
    ppl = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Skip step 1 as it has a very high initial perplexity that skews the plot range
            if row.get('step', 1) == 1:
                continue
            steps.append(row.get('step'))
            ppl.append(row.get('eval_ppl'))
    return steps, ppl

def main():
    if len(sys.argv) != 3:
        print("Usage: plot_ablation_results.py <log_dir> <output_dir>")
        sys.exit(1)
        
    log_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    seq_lengths = [1024, 2048]
    
    for ax_idx, seq_len in enumerate(seq_lengths):
        ax = axes[ax_idx]
        ax.set_title(f"Sequence Length $T = {seq_len}$", fontweight='bold', pad=10)
        
        # Plot each ablation model configuration
        for config_name in ['local_conv', 'pcaf_no_gate', 'pcaf_semantic', 'pcaf_context']:
            file_path = log_dir / f"{config_name}_seq{seq_len}.jsonl"
            if not file_path.exists():
                print(f"Warning: file {file_path} not found. Skipping...")
                continue
                
            steps, ppl = parse_jsonl(file_path)
            
            ax.plot(
                steps, 
                ppl, 
                color=COLORS[config_name], 
                label=LABELS[config_name],
                marker=MARKERS[config_name],
                markersize=6,
                linewidth=2,
                markevery=2
            )
            
        ax.set_xlabel("Training Step", labelpad=8)
        if ax_idx == 0:
            ax.set_ylabel("Validation Perplexity (WikiText-2)", labelpad=8)
            
        ax.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    # Place a single shared legend at the top of the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, 
        labels, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 0.98), 
        ncol=4, 
        frameon=False
    )
    
    # Adjust layout to make room for titles and shared legend
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    
    # Save both vector PDF (for publication) and high-res PNG (for viewing)
    pdf_path = output_dir / "ablation_curves.pdf"
    png_path = output_dir / "ablation_curves.png"
    
    plt.savefig(pdf_path, format='pdf')
    plt.savefig(png_path, format='png')
    plt.close()
    
    print(f"Successfully generated ablation plot at:\n - {pdf_path}\n - {png_path}")

if __name__ == "__main__":
    main()
