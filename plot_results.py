#!/usr/bin/env python3
"""
plot_results.py  —  produces two figures from benchmark_all CSV output.
Usage:  python3 plot_results.py bench_<JOBID>.csv
Output: bench_<JOBID>_graph1.png  (CPU vs GPU kernels, power-of-2)
        bench_<JOBID>_graph2.png  (Bluestein performance, non-pow2 neighborhood)
"""

import sys, collections, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── load ──────────────────────────────────────────────────────────────────────
def load(path):
    g1 = collections.defaultdict(lambda: {"logn":[],"N":[],"mean":[],"std":[],"gf":[]})
    g2 = collections.defaultdict(lambda: {"ref_logN":[],"N":[],"is_pow2":[],"mean":[],"std":[],"gf":[]})

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split(",")

            if parts[0] == "graph1" and len(parts) == 7:
                _, kernel, logn, N, mean, std, gf = parts
                if kernel == "gpu_global": continue
                d = g1[kernel]
                d["logn"].append(int(logn))
                d["N"].append(int(N))
                d["mean"].append(float(mean))
                d["std"].append(float(std))
                d["gf"].append(float(gf))

            elif parts[0] == "graph2" and len(parts) == 8:
                _, kernel, ref_logN, N, is_pow2, mean, std, gf = parts
                if kernel == "gpu_global": continue
                d = g2[kernel]
                d["ref_logN"].append(int(ref_logN))
                d["N"].append(int(N))
                d["is_pow2"].append(int(is_pow2))
                d["mean"].append(float(mean))
                d["std"].append(float(std))
                d["gf"].append(float(gf))

    return g1, g2

# ── style ─────────────────────────────────────────────────────────────────────
STYLE = {
    "cpu_radix2":  {"color":"#1f77b4", "marker":"o", "ls":"-",  "label":"CPU radix-2"},
    "cpu_radix4":  {"color":"#aec7e8", "marker":"s", "ls":"--", "label":"CPU radix-4"},
    "gpu_hybrid":  {"color":"#ff7f0e", "marker":"D", "ls":"-",  "label":"GPU hybrid"},
    "gpu_bailey":  {"color":"#2ca02c", "marker":"*", "ls":"-",  "label":"GPU Bailey 4-step"},
    "bluestein":   {"color":"#9467bd", "marker":"x", "ls":"--", "label":"Bluestein (any N)"},
}

def style(k):
    return STYLE.get(k, {"color":"gray","marker":"o","ls":"-","label":k})

# ── Graph 1 ───────────────────────────────────────────────────────────────────
def plot_graph1(g1, out_path):
    fig, (ax_t, ax_g) = plt.subplots(1, 2, figsize=(13, 5))

    for kernel, d in sorted(g1.items()):
        st = style(kernel)
        xs = d["logn"]
        ax_t.errorbar(xs, d["mean"], yerr=d["std"],
                      color=st["color"], marker=st["marker"], ls=st["ls"],
                      label=st["label"], ms=5, capsize=3, lw=1.5)
        ax_g.plot(xs, d["gf"],
                  color=st["color"], marker=st["marker"], ls=st["ls"],
                  label=st["label"], ms=5, lw=1.5)

    for ax in (ax_t, ax_g):
        ax.set_xlabel("log₂(N)")
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.35)
        ax.legend(fontsize=8)

    ax_t.set_ylabel("Mean latency (µs, log scale)")
    ax_t.set_title("Graph 1 — Runtime: CPU vs GPU kernels (power-of-2)")

    ax_g.set_ylabel("GFLOPS (log scale)")
    ax_g.set_title("Graph 1 — Throughput (5·N·log₂N / time)")

    plt.suptitle("GPU FFT Benchmark — power-of-2 sizes", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

# ── Graph 2 ───────────────────────────────────────────────────────────────────
def plot_graph2(g2, out_path):
    """
    X-axis = actual N (linear).
    Bluestein: scatter for all N, distinguishing pow2 vs non-pow2.
    Annotate the power-of-2 positions with logN labels.
    """
    fig, (ax_t, ax_g) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Bluestein ----
    if "bluestein" in g2:
        d = g2["bluestein"]
        # split pow2 vs non-pow2
        Np2 = [d["N"][i] for i in range(len(d["N"])) if d["is_pow2"][i]]
        mp2 = [d["mean"][i] for i in range(len(d["N"])) if d["is_pow2"][i]]
        sp2 = [d["std"][i]  for i in range(len(d["N"])) if d["is_pow2"][i]]
        gp2 = [d["gf"][i]   for i in range(len(d["N"])) if d["is_pow2"][i]]

        Nnp = [d["N"][i] for i in range(len(d["N"])) if not d["is_pow2"][i]]
        mnp = [d["mean"][i] for i in range(len(d["N"])) if not d["is_pow2"][i]]
        snp = [d["std"][i]  for i in range(len(d["N"])) if not d["is_pow2"][i]]
        gnp = [d["gf"][i]   for i in range(len(d["N"])) if not d["is_pow2"][i]]

        ax_t.errorbar(Nnp, mnp, yerr=snp, fmt="x", color="#9467bd",
                      label="Bluestein (non-pow2)", ms=5, capsize=2, lw=1, alpha=0.8)
        ax_t.errorbar(Np2, mp2, yerr=sp2, fmt="o", color="#9467bd",
                      label="Bluestein (pow2)", ms=6, capsize=2, lw=1.5)

        ax_g.scatter(Nnp, gnp, marker="x", color="#9467bd", s=30,
                     label="Bluestein (non-pow2)", alpha=0.8)
        ax_g.scatter(Np2, gp2, marker="o", color="#9467bd", s=50,
                     label="Bluestein (pow2)")


    for ax in (ax_t, ax_g):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("N")
        ax.grid(True, which="both", ls="--", alpha=0.35)
        ax.legend(fontsize=8)
        # label the power-of-2 x-ticks with logN
        pow2_ticks = [1<<k for k in range(1,23)]
        ax.set_xticks(pow2_ticks)
        ax.set_xticklabels([f"2^{k}" for k in range(1,23)], rotation=45, fontsize=7)

    ax_t.set_ylabel("Mean latency (µs, log scale)")
    ax_t.set_title("Graph 2 — Runtime: Bluestein performance")

    ax_g.set_ylabel("GFLOPS (log scale)")
    ax_g.set_title("Graph 2 — Throughput: Bluestein performance")

    plt.suptitle("GPU FFT Benchmark — arbitrary-N neighborhood (N±1, N±3 around each 2^k)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "bench.csv"
    g1, g2 = load(path)
    
    base = path.replace(".csv","").replace(".out","")
    plot_graph1(g1, base + "_graph1.png")
    plot_graph2(g2, base + "_graph2.png")
