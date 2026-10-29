"""
Report charts from $RESULTS_DIR (default ./results). Runs on a login node, no GPU needed:
  source common.sh && python plot_results.py
Writes $RESULTS_DIR/plots/*.png and summary.txt. A chart whose runs don't exist yet is skipped.

  scaling.png               strong (global batch 64) + weak (32/GPU) scaling vs ideal, 400-step profiling runs
  batch_sweep.png           1-GPU throughput vs per-GPU batch size, plus the TF32 point
  throughput_over_time.png  throughput across the long runs (02/03): dips vs steady state
  heldout.png               held-out top-1 / NLL vs epoch for 02/03, paper SmallIL as a reference line

Throughput points are the median over 100-step windows; error bars span min-max, because
data-loader stalls make single windows noisy. Training loss curves are in wandb, not here.
"""
import csv
import glob
import os
import statistics

import matplotlib

matplotlib.use("Agg")  # no display on Perlmutter
import matplotlib.pyplot as plt

RESULTS = os.environ.get("RESULTS_DIR", "results")
OUT = os.path.join(RESULTS, "plots")

# light reference palette (fixed slot order); text stays in ink colors, never the series color
SURFACE, INK, INK2, GRID, IDEAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#9a9893"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RUN_STYLE = {"bc1_gen1ou": ("1 GPU (job 02)", BLUE), "bc4_gen1ou": ("4 GPUs (job 03)", ORANGE)}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False,
})


def load_throughput():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "throughput_*.csv"))):
        run = os.path.basename(path)[len("throughput_"):-len(".csv")]
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["step"] == "step":  # header repeated if a file was appended to after a crash
                    continue
                rows.append(dict(
                    run=run, t=float(r["unix_time"]), n_gpus=int(r["n_gpus"]),
                    batch=int(r["batch_per_gpu"]), global_batch=int(r["global_batch"]),
                    sps=float(r["steps_per_sec"]), seqs=float(r["seqs_per_sec"]),
                    mem=float(r["peak_mem_gb_rank0"]), tf32=r["tf32"] == "True",
                ))
    return rows


def load_heldout():
    path = os.path.join(RESULTS, "heldout_eval.csv")
    if not os.path.exists(path):
        return []
    latest = {}  # reruns append duplicate rows; keep the last one per (model, epoch, min_date)
    with open(path) as f:
        for r in csv.DictReader(f):
            latest[(r["model"], int(r["epoch"]), r["format"], r["min_date"])] = r
    return [dict(model=m, epoch=e, min_date=d, top1=float(r["top1"]), top2=float(r["top2"]),
                 nll=float(r["nll"]), n=int(r["n_actions"])) for (m, e, _, d), r in latest.items()]


def med(rows, key):
    v = [r[key] for r in rows]
    return statistics.median(v), min(v), max(v)


def errorbar(ax, xs, groups, key, color, label, **kw):
    stats = [med(groups[x], key) for x in xs]
    m = [s[0] for s in stats]
    yerr = [[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]]
    ax.errorbar(xs, m, yerr=yerr, color=color, label=label, marker="o", capsize=3,
                elinewidth=1, markeredgecolor=SURFACE, markeredgewidth=1.5, **kw)
    return m


def scaling_panel(ax, groups, title, lines):
    ns = sorted(n for n, g in groups.items() if g)
    if 1 not in ns or len(ns) < 2:
        ax.text(0.5, 0.5, "needs a 1-GPU and a multi-GPU run", ha="center", va="center",
                color=INK2, transform=ax.transAxes)
        ax.set_title(title, loc="left")
        return
    m = errorbar(ax, ns, groups, "seqs", BLUE, "Measured (median)")
    base = m[0]
    ax.plot(ns, [base * n for n in ns], color=IDEAL, linestyle="--", linewidth=1.5, label="Ideal (linear)")
    for n, v in zip(ns[1:], m[1:]):
        ax.annotate(f"{v / base:.1f}x, {v / (base * n):.0%} eff.", (n, v), xytext=(6, -14),
                    textcoords="offset points", color=INK2, fontsize=9)
        lines.append(f"  {title}: {n} GPUs = {v / base:.2f}x speedup, {v / (base * n):.0%} efficiency")
    ax.set_xticks(ns)
    ax.set_xlabel("GPUs")
    ax.set_ylabel("sequences / sec")
    ax.set_ylim(bottom=0)
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left")


def fig_scaling(T, lines):
    prof = [r for r in T if r["run"].startswith("prof") and not r["tf32"]]
    counts = sorted({r["n_gpus"] for r in prof})
    strong = {n: [r for r in prof if r["n_gpus"] == n and r["global_batch"] == 64] for n in counts}
    weak = {n: [r for r in prof if r["n_gpus"] == n and r["batch"] == 32 and r["global_batch"] == 32 * n]
            for n in counts}
    if sum(1 for g in strong.values() if g) < 2 and sum(1 for g in weak.values() if g) < 2:
        return "scaling.png: skipped (need profiling runs from jobs 00 + 01)"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    lines.append("Scaling (400-step profiling runs, median of windows):")
    scaling_panel(axes[0], strong, "Strong scaling: global batch 64", lines)
    scaling_panel(axes[1], weak, "Weak scaling: 32 per GPU", lines)
    fig.text(0.01, 0.01, "Points: median of 100-step windows. Bars: min-max across windows.",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "scaling.png"), dpi=200)
    plt.close(fig)
    return "scaling.png"


def fig_batch_sweep(T):
    one = [r for r in T if r["run"].startswith("prof1")]
    fp32 = {b: [r for r in one if r["batch"] == b and not r["tf32"]] for b in sorted({r["batch"] for r in one})}
    fp32 = {b: g for b, g in fp32.items() if g}
    if len(fp32) < 2:
        return "batch_sweep.png: skipped (need job 00)"
    fig, ax = plt.subplots(figsize=(6, 4))
    bs = sorted(fp32)
    errorbar(ax, bs, fp32, "seqs", BLUE, "FP32 (paper setting)")
    tf = {b: [r for r in one if r["batch"] == b and r["tf32"]] for b in bs}
    tf = {b: g for b, g in tf.items() if g}
    if tf:  # nudged right so its error bar doesn't sit on the FP32 one
        tf = {b * 1.08: g for b, g in tf.items()}
        errorbar(ax, sorted(tf), tf, "seqs", ORANGE, "TF32 matmuls", linestyle="none")
    ax.set_xscale("log", base=2)
    ax.set_xticks(bs, [f"{b}\n{max(r['mem'] for r in fp32[b]):.1f} GB" for b in bs])
    ax.minorticks_off()
    ax.set_xlabel("batch size per GPU, with peak GPU memory (1 GPU)")
    ax.set_ylabel("sequences / sec")
    ax.set_ylim(bottom=0)
    ax.set_title("Throughput vs batch size", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "batch_sweep.png"), dpi=200)
    plt.close(fig)
    return "batch_sweep.png"


def fig_over_time(T, lines):
    runs = sorted({r["run"] for r in T if not r["run"].startswith("prof")})
    runs = [run for run in runs if sum(r["run"] == run for r in T) >= 5]
    if not runs:
        return "throughput_over_time.png: skipped (need long runs from jobs 02/03)"
    fig, ax = plt.subplots(figsize=(8, 4))
    lines.append("Long runs (median over all windows):")
    for i, run in enumerate(runs):
        rows = sorted((r for r in T if r["run"] == run), key=lambda r: r["t"])
        label, color = RUN_STYLE.get(run, (run, [BLUE, ORANGE, AQUA][i % 3]))
        xs = [100 * (j + 1) for j in range(len(rows))]  # resumed jobs restart the step counter
        ax.plot(xs, [r["seqs"] for r in rows], color=color, label=label, linewidth=1.5)
        m = med(rows, "seqs")[0]
        lines.append(f"  {run}: {m:.1f} seqs/s, {med(rows, 'sps')[0]:.3f} steps/s over {len(rows)} windows")
    ax.set_xlabel("training steps measured (cumulative across resubmits)")
    ax.set_ylabel("sequences / sec (100-step windows)")
    ax.set_ylim(bottom=0)
    ax.set_title("Throughput over the long runs (100-step windows)", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "throughput_over_time.png"), dpi=200)
    plt.close(fig)
    return "throughput_over_time.png"


def fig_heldout(H, lines):
    full = [h for h in H if not h["min_date"]]
    ref = [h for h in full if not h["model"].startswith("bc")]
    runs = sorted({h["model"] for h in full if h["model"].startswith("bc")})
    if H:
        lines.append("Held-out eval (latest row per model/epoch/date filter):")
        for h in sorted(H, key=lambda h: (h["model"], h["min_date"], h["epoch"])):
            lines.append(f"  {h['model']} ep{h['epoch']} {h['min_date'] or 'all dates'}: "
                         f"top1 {h['top1']:.4f}  top2 {h['top2']:.4f}  nll {h['nll']:.4f}  (n={h['n']})")
    if not runs:
        return "heldout.png: skipped (need checkpoints from jobs 02/03 + job 05)"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, name in [(axes[0], "top1", "Top-1 accuracy (higher is better)"),
                          (axes[1], "nll", "NLL (lower is better)")]:
        for i, run in enumerate(runs):
            pts = sorted((h for h in full if h["model"] == run), key=lambda h: h["epoch"])
            label, color = RUN_STYLE.get(run, (run, [BLUE, ORANGE, AQUA][i % 3]))
            ax.plot([h["epoch"] for h in pts], [h[key] for h in pts], color=color, label=label,
                    marker="o", markeredgecolor=SURFACE, markeredgewidth=1.5)
        for h in ref:
            ax.axhline(h[key], color=IDEAL, linestyle="--", linewidth=1.5, label=f"Paper {h['model']}")
        ax.set_xlabel("epoch (5,000 steps each)")
        ax.set_title(name, loc="left")
        ax.legend(loc="lower right" if key == "top1" else "upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "heldout.png"), dpi=200)
    plt.close(fig)
    return "heldout.png"


def summary_table(T):
    out = ["Throughput by run (median over 100-step windows):",
           f"  {'run':34s} {'GPUs':>4s} {'B/GPU':>5s} {'global':>6s} {'tf32':>5s} {'win':>3s} "
           f"{'steps/s':>8s} {'seqs/s':>8s} {'mem GB':>6s}"]
    for run in sorted({r["run"] for r in T}):
        g = [r for r in T if r["run"] == run]
        out.append(f"  {run:34s} {g[0]['n_gpus']:4d} {g[0]['batch']:5d} {g[0]['global_batch']:6d} "
                   f"{str(g[0]['tf32']):>5s} {len(g):3d} {med(g, 'sps')[0]:8.3f} {med(g, 'seqs')[0]:8.1f} "
                   f"{max(r['mem'] for r in g):6.2f}")
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    T, H = load_throughput(), load_heldout()
    lines = summary_table(T) if T else ["No throughput CSVs found in " + RESULTS]
    made = [fig_scaling(T, lines), fig_batch_sweep(T), fig_over_time(T, lines), fig_heldout(H, lines)]
    text = "\n".join(lines)
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    print("\nCharts in " + OUT + ":")
    for m in made:
        print("  " + m)
