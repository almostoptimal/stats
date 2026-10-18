# -*- coding: utf-8 -*-
"""
《赢家诅咒》统一版配图生成脚本（可重跑）

用法：
    D:/anaconda/python.exe make_winfig.py            # 输出 SVG
    WC_OUT_FMT=png D:/anaconda/python.exe make_winfig.py   # 输出 PNG 预览

产出（图号与正文一一对应）：
    figw1_auction.{FMT}       图 1：同一个读数 115，在四种处境下的后验
    figw2_cloud.{FMT}         图 2：只有越过阈值的点被看见
    figw3_overfit.{FMT}       图 3：过拟合是连续版的赢家诅咒
    figw4_ruler.{FMT}         图 4：一把尺子 σ√(2log M)，与四个场景
    figw5_split.{FMT}         图 5：把选择与评价分开
    figw6_tails.{FMT}         图 6（补充）：换分布，倍数变，机制不变

所有数值现算，不写死。
"""
import os
import numpy as np
from scipy.stats import (norm, t, logistic, laplace, expon, lognorm, pareto,
                         uniform)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from numpy.polynomial import chebyshev as Ch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"

C_BLUE, C_ORANGE, C_GREEN, C_GREY, C_RED = "#2E5C8A", "#C75B39", "#4E8C72", "#8A8A8A", "#B33A3A"
C_TEAL, C_PURPLE = "#3A7C8C", "#6B5B95"
OUT = os.path.dirname(os.path.abspath(__file__))
FMT = os.environ.get("WC_OUT_FMT", "svg").lower()

MU_V, SIG_V, SIG_E = 100.0, 20.0, 10.0
LAMBDA = SIG_V ** 2 / (SIG_V ** 2 + SIG_E ** 2)
X_STAR = 115.0
C_GWAS = 5.4517          # 双侧 p < 5e-8

GAMMA_EM = 0.5772156649015329


# ============================================================================
# 数值工具
# ============================================================================
def posterior_V(x, n=1, grid=None):
    """共同价值拍卖的后验 p(v | x, win)，n 个竞标者（含自己）。"""
    if grid is None:
        grid = np.linspace(30, 200, 4001)
    lik = norm.pdf((x - grid) / SIG_E)
    if n > 1:
        lik = lik * norm.cdf((x - grid) / SIG_E) ** (n - 1)
    w = norm.pdf((grid - MU_V) / SIG_V) * lik
    w /= np.trapz(w, grid)
    mu = np.trapz(grid * w, grid)
    return grid, w, mu


def posterior_sd(grid, w):
    m = np.trapz(grid * w, grid)
    v = np.trapz((grid - m) ** 2 * w, grid)
    return np.sqrt(v)


def power_two_tail(c, delta):
    return norm.cdf(delta - c) + norm.cdf(-c - delta)


def E_absZ_two_tail(c, delta):
    pA = norm.cdf(delta - c) + norm.cdf(-c - delta)
    num = (norm.pdf(c - delta) + norm.pdf(c + delta)
           + delta * (norm.cdf(delta - c) - norm.cdf(-c - delta)))
    return num / pA


_XG = np.linspace(-6.0, 8.5, 9001)
_PHI = norm.pdf(_XG)
_PHI_CDF = norm.cdf(_XG)


def E_max_exact(M):
    """iid 标准正态最大值的期望：∫ z·M·φ(z)Φ(z)^(M-1) dz"""
    M = int(M)
    return float(np.trapz(_XG * M * _PHI * _PHI_CDF ** (M - 1), _XG))


def E_max_extreme(N):
    """Bailey–López de Prado 极值近似，N>=10 时误差 <2%"""
    N = np.maximum(np.asarray(N, dtype=float), 1.0001)
    return ((1 - GAMMA_EM) * norm.ppf(1 - 1 / N)
            + GAMMA_EM * norm.ppf(1 - 1 / (N * np.e)))


def savefig(fig, name):
    p = os.path.join(OUT, f"{name}.{FMT}")
    fig.savefig(p, format=FMT)
    plt.close(fig)
    return p


# ============================================================================
# 图 1：拍卖 —— 同一个读数，四种处境
# ============================================================================
def fig_auction(x_star=X_STAR):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.7),
                                   gridspec_kw={"width_ratios": [1.18, 1.0]})

    # ---- 左：后验密度 ----
    grid = np.linspace(55, 160, 4001)
    g, w0, mu0 = posterior_V(x_star, 1, grid)
    ax1.plot(g, w0, color=C_GREY, lw=2.3, ls="--", label="不条件于输赢（n=1）")
    for n, col in [(5, C_BLUE), (15, C_RED)]:
        _, w, mu = posterior_V(x_star, n, grid)
        ax1.plot(g, w, color=col, lw=2.0, label=f"条件于赢，n={n}")
        ax1.axvline(mu, color=col, lw=0.9, alpha=0.5, ymax=0.62)

    ax1.axvline(112, color=C_ORANGE, lw=1.8, ls="-.", ymax=0.93)
    ax1.annotate("你打算出的价 112", xy=(112, 0.0455), xytext=(120, 0.0605),
                 color=C_ORANGE, fontsize=9.6,
                 arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=1.0))
    ax1.axvline(MU_V, color="#333333", lw=1.0, alpha=0.28)

    ax1.set_xlim(62, 152)
    ax1.set_ylim(0, 0.086)
    ax1.set_xlabel("这块地的真实价值（百万）")
    ax1.set_ylabel("后验密度")
    ax1.set_title("读数是 115：条件于“我赢了”，答案整体左移", fontsize=11.2)
    ax1.legend(fontsize=9.0, loc="upper left", framealpha=0.95)
    ax1.grid(alpha=0.15, lw=0.6)

    # ---- 右：高估幅度 vs 人数 ----
    ns = np.arange(1, 31)
    gaps, gap_err = [], []
    for n in ns:
        _, w, mu = posterior_V(x_star, n, grid)
        gaps.append(112.0 - mu)
        gap_err.append(0.6 * posterior_sd(grid, w) / np.sqrt(n))
    gaps = np.array(gaps)

    ax2.plot(ns, gaps, color=C_RED, lw=2.4)
    ax2.fill_between(ns, 0, gaps, color=C_RED, alpha=0.09)
    ax2.axhline(0, color="#333333", lw=1.0, ls="--")
    ax2.text(6.0, 0.75, "没有竞争时，一分不差", fontsize=9.2, color="#333333")
    for n in (2, 5, 15, 30):
        y = gaps[n - 1]
        ax2.plot([n], [y], "o", color=C_RED, ms=5)
        ax2.annotate(f"{y:.1f}", (n, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=9.0, color=C_RED,
                     bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))
    ax2.set_xlim(0.5, 31.5)
    ax2.set_ylim(-0.6, 20.5)
    ax2.set_xlabel("竞标人数 n")
    ax2.set_ylabel("出价 112 平均高估多少（百万）")
    ax2.set_title("人越多，赢家的读数越“靠运气”", fontsize=11.2)
    ax2.grid(alpha=0.15, lw=0.6)

    fig.suptitle("图 1　同一个读数 115，两种问法，两个答案", fontsize=12.8, y=1.02)
    return savefig(fig, "figw1_auction"), gaps


# ============================================================================
# 图 2：筛选的机制 —— 只有越过阈值的点被看见
# ============================================================================
def fig_cloud(n_cand=2200, seed=20261018, c=C_GWAS):
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.0, 5.0, n_cand)
    z = d + rng.normal(0, 1, n_cand)
    seen = z > c

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.scatter(d[~seen], z[~seen], s=14, color="#B8B8B8", alpha=0.55, lw=0,
               label="做出来了，但没过线（看不到）")
    ax.scatter(d[seen], z[seen], s=26, color=C_RED, alpha=0.85, lw=0,
               label=f"过线，被写进论文（{seen.sum()} 个）")

    ax.axhline(c, color="#333333", lw=1.3, ls="-.")
    ax.text(0.08, c + 0.34, "报告门槛：双侧 p < 5e-8（Z = 5.45）",
            fontsize=9.6, color="#333333",
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.9))

    if seen.sum():
        mz, md = z[seen].mean(), d[seen].mean()
        ax.axhline(mz, color=C_RED, lw=1.4, ls=":")
        ax.plot([md], [c], "o", color="#333333", ms=5, zorder=5)
        ax.annotate("", xy=(md, c), xytext=(md, mz),
                    arrowprops=dict(arrowstyle="<->", color=C_RED, lw=1.4))
        ax.annotate(f"虚增 {mz - md:.2f}", xy=(md, 0.5 * (c + mz)),
                    xytext=(4.86, 7.50), fontsize=9.4, color=C_RED, ha="right",
                    arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.9))
        ax.annotate(f"报告值平均 {mz:.2f}", xy=(3.05, mz), xytext=(2.75, 7.55),
                    fontsize=9.4, color=C_RED,
                    arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.9))
        ax.annotate(f"这些位点的真值平均 {md:.2f}", xy=(md, c - 0.10),
                    xytext=(3.05, 4.35), fontsize=9.4, color="#333333",
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9))

    ax.set_xlim(0, 5.05)
    ax.set_ylim(-3.0, 8.4)
    ax.set_xlabel("位点的真实效应（以标准误为单位）")
    ax.set_ylabel("算出来的 Z 值")
    ax.set_title("红色点排成一条近水平的带：报告值由“过没过线”决定，不由真值决定",
                 fontsize=10.4, pad=10)
    ax.legend(fontsize=9.2, loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.14, lw=0.6)
    fig.suptitle("图 2　筛选的机制：被看见的那一条带，几乎是平的", fontsize=12.6, y=1.02)
    return savefig(fig, "figw2_cloud"), (float(d[seen].mean()), float(z[seen].mean()))


# ============================================================================
# 图 3：一把尺子 —— 偏差 σ√(2log M)，与五个场景
# ============================================================================
def fig_ruler():
    Ms = np.logspace(np.log10(2), 7, 220)
    ex = np.array([E_max_exact(m) for m in np.unique(np.round(Ms).astype(int))])
    Mu = np.unique(np.round(Ms).astype(int))
    asy = np.sqrt(2 * np.log(Mu))

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.plot(Mu, asy, color=C_BLUE, lw=1.6, ls="--",
            label="教材式 √(2log M)：偏保守，是个上界")
    ax.plot(Mu, ex, color=C_RED, lw=2.4, label="精确值（数值积分）")
    ax.fill_between(Mu, ex, asy, color=C_BLUE, alpha=0.10)

    marks = [(5, "5 家竞标油田", (7, 20)),
             (100, "100 组超参 / 100 条回测", (10, 16)),
             (1e6, "GWAS：100 万个位点", (6, -26))]
    lab_y = []
    for m, name, off in marks:
        y = E_max_exact(int(m))
        lab_y.append((m, y))
        ax.plot([m], [y], "o", color=C_GREEN, ms=8, mec="white", mew=1.4, zorder=6)
        ax.annotate(f"{name}\n{y:.2f}σ", (m, y), textcoords="offset points",
                    xytext=off, fontsize=9.0, color=C_GREEN,
                    ha="right" if m > 1e5 else "left",
                    va="top" if off[1] < 0 else "bottom",
                    bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.9))

    ax.plot([2], [E_max_exact(2)], "o", color=C_PURPLE, ms=7, zorder=6)
    ax.annotate("只有 2 个候选也有 0.56σ：\n“两个就够，不是做多了才有”",
                (2, 0.56), xytext=(7.2, 0.24), textcoords="data",
                fontsize=9.0, color=C_PURPLE, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=0.9,
                                connectionstyle="arc3,rad=0.18"))

    ax.set_xscale("log")
    ax.set_xlim(2, 1.2e7)
    ax.set_ylim(0, 6.4)
    ax.set_xlabel("有效候选数 M（取对数）")
    ax.set_ylabel("被选中者的平均高估（以噪声 σ 为单位）")
    ax.set_title("门槛的高度只取决于两件事：噪声有多大、试过多少次", fontsize=11.0)
    ax.legend(fontsize=9.4, loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.15, lw=0.6, which="both")
    fig.suptitle("图 4　一把尺子：偏差 ≈ σ × √(2log M)", fontsize=12.8, y=1.02)
    return savefig(fig, "figw4_ruler"), lab_y


# ============================================================================
# 图 4：过拟合 —— 连续版的赢家诅咒
# ============================================================================
def fig_overfit(n=30, sigma=0.50, dmax=10, reps=600, seed=11):
    rng = np.random.default_rng(seed)
    f = lambda x: np.sin(5 * x) + 0.5 * x
    x = np.linspace(-1, 1, n)
    xt = np.linspace(-1, 1, 1200)
    fx, ft = f(x), f(xt)
    degs = np.arange(1, dmax + 1)
    tr = np.zeros((reps, len(degs)))
    te = np.zeros((reps, len(degs)))

    for r in range(reps):
        y = fx + rng.normal(0, sigma, n)
        yt = ft + rng.normal(0, sigma, xt.size)
        for k, d in enumerate(degs):
            c = Ch.chebfit(x, y, d)
            tr[r, k] = np.mean((Ch.chebval(x, c) - y) ** 2)
            te[r, k] = np.mean((Ch.chebval(xt, c) - yt) ** 2)

    mtr, mte = tr.mean(axis=0), te.mean(axis=0)
    kbest = int(np.argmin(mte))

    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    ax.plot(degs, mtr, color=C_BLUE, lw=2.2, marker="o", ms=4.6,
            label="训练误差（同一批数据，自评）")
    ax.plot(degs, mte, color=C_RED, lw=2.2, marker="s", ms=4.6,
            label="测试误差（新数据，他评）")
    ax.fill_between(degs, mtr, mte, color=C_RED, alpha=0.10)

    ax.axhline(sigma ** 2, color="#333333", lw=1.0, ls=":")
    ax.text(1.12, sigma ** 2 + 0.014, "噪声底 0.25（训练误差低于它，一定在记噪声）",
            fontsize=8.9, color="#333333", ha="left")

    ax.axvline(degs[kbest], color=C_GREEN, lw=1.1, ls="--")
    ax.annotate(f"真正最优 d={degs[kbest]}", xy=(degs[kbest], mte[kbest]),
                xytext=(degs[kbest] + 0.45, 0.66), fontsize=9.6, color=C_GREEN,
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=0.9))

    ax.annotate("", xy=(degs[-1], mte[-1]), xytext=(degs[-1], mtr[-1]),
                arrowprops=dict(arrowstyle="<->", color=C_ORANGE, lw=1.6))
    ax.text(degs[-1] - 0.28, 0.5 * (mte[-1] + mtr[-1]) - 0.02,
            "乐观量\n%.2f" % (mte[-1] - mtr[-1]), fontsize=9.6,
            color=C_ORANGE, ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.95))

    ax.set_xlim(0.6, dmax + 0.35)
    ax.set_ylim(0.09, 0.90)
    ax.set_xticks(degs)
    ax.set_xlabel("模型复杂度 d（自由度 d+1）")
    ax.set_ylabel("均方误差")
    ax.set_title("训练误差一路降，测试误差 d=5 之后持续回升", fontsize=10.8)
    ax.legend(fontsize=9.4, loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.15, lw=0.6)
    fig.suptitle("图 3　过拟合：训练误差里有一块是「拟合噪声」", fontsize=12.6, y=1.02)
    return savefig(fig, "figw3_overfit"), [(int(d), float(a), float(b))
                                           for d, a, b in zip(degs, mtr, mte)]


# ============================================================================
# 图 5：把选择与评价分开
# ============================================================================
def fig_split():
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=9.4, tc="#222222"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.6,rounding_size=1.6",
                                    fc=fc, ec=ec, lw=1.4, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=4)

    def arrow(x1, y1, x2, y2, col="#666666"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color=col, lw=1.3, zorder=2))

    # 左列：复用
    ax.text(26, 96, "做法一：同一批数据选，也用它评", ha="center", fontsize=10.6,
            color=C_RED, fontweight="bold")
    box(26, 82, 34, 9, "一批数据 D", "#EEF3F8", C_BLUE)
    box(26, 61, 34, 9, "在 D 里选出最好的那个", "#EEF3F8", C_BLUE)
    box(26, 40, 34, 9, "用 D 报告它有多好", "#FDECEC", C_RED)
    box(26, 17, 34, 11, "报告值偏高\n（D 里的噪声被用了两次）", "#FDECEC", C_RED, fs=9.2)
    for y1, y2 in [(77.4, 65.6), (56.4, 44.6), (35.4, 22.7)]:
        arrow(26, y1, 26, y2)
    ax.text(47.0, 40, "偏", fontsize=12, color=C_RED, ha="center", va="center")

    # 右列：分开
    ax.text(74, 96, "做法二：一半选，另一半评", ha="center", fontsize=10.6,
            color=C_GREEN, fontweight="bold")
    box(74, 82, 34, 9, "数据 D 拆成 D1 和 D2", "#EEF3F8", C_BLUE)
    box(60, 61, 22, 9, "D1：选出最好的", "#EEF7F2", C_GREEN, fs=9.0)
    box(88, 61, 22, 9, "D2：报告它多好", "#EEF7F2", C_GREEN, fs=9.0)
    box(74, 40, 34, 9, "两边用的是不同的噪声", "#EEF7F2", C_GREEN, fs=9.2)
    box(74, 17, 34, 11, "报告值是对的\n（代价：各自的数据变少）", "#EEF7F2", C_GREEN, fs=9.2)
    arrow(68, 77.4, 62, 65.6)
    arrow(80, 77.4, 86, 65.6)
    arrow(60, 56.4, 70, 44.6)
    arrow(88, 56.4, 78, 44.6)
    arrow(74, 35.4, 74, 22.7)
    ax.text(96.6, 40, "准", fontsize=12, color=C_GREEN, ha="center", va="center")

    ax.text(50, 4, "同一句话：让「挑」和「量」用的不是同一份噪声",
            ha="center", fontsize=9.6, color="#444444", style="italic")

    fig.suptitle("图 5　所有的解药都是这一件事：把选择与评价分开", fontsize=12.8, y=1.0)
    return savefig(fig, "figw5_split")


# ============================================================================
# 图 6（补充）：换分布 —— 倍数变，机制不变
# ============================================================================
def dists():
    return [
        ("uniform", "均匀（有界）", uniform(-np.sqrt(3), 2 * np.sqrt(3))),
        ("norm", "正态", norm(0, 1)),
        ("t30", "t(30)", t(30)),
        ("logistic", "Logistic", logistic(0, np.sqrt(3) / np.pi)),
        ("laplace", "Laplace", laplace(0, 1 / np.sqrt(2))),
        ("exp", "指数", expon(scale=1.0)),
        ("t10", "t(10)", t(10)),
        ("t5", "t(5)", t(5)),
        ("lognorm", "对数正态", lognorm(s=1.0)),
        ("pareto", "Pareto(3)", pareto(3, loc=1)),
        ("t3", "t(3)", t(3)),
    ]


def fig_tails(M=1000000):
    rows = []
    for key, lab, d in dists():
        sg = float(d.std())
        xs = float(d.isf(1.0 / M))
        r = max(float(d.sf(xs) / d.pdf(xs)), 1e-6)
        e = E_max_exact(M) if key == "norm" else None
        rows.append([lab, r, sg])
    # 统一用数值积分算 E[max]
    out = []
    for key, lab, d in dists():
        sg = float(d.std())
        xs = float(d.isf(1.0 / M))
        r = max(float(d.sf(xs) / d.pdf(xs)), 1e-6)
        out.append((lab, r, _Emax_generic(d, M) / sg))

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    thin = [(l, r, e) for l, r, e in out if r < 0.9]
    heavy = [(l, r, e) for l, r, e in out if r >= 0.9]
    ax.scatter([r for _, r, _ in thin], [e for _, _, e in thin], s=74,
               color=C_BLUE, edgecolor="white", lw=1.2, zorder=4,
               label="薄尾 / 指数型尾（倍数 1.7–14）")
    ax.scatter([r for _, r, _ in heavy], [e for _, _, e in heavy], s=74,
               color=C_RED, edgecolor="white", lw=1.2, zorder=4,
               label="幂律尾（倍数 22–157）")

    off = {"正态": (4, -16), "Logistic": (-13, 3), "Laplace": (-11, 9),
           "指数": (7, 11), "均匀": (9, -6), "t(30)": (-9, -12), "t(10)": (11, -11),
           "t(5)": (11, 1), "t(3)": (12, -2), "对数正态": (-10, 12),
           "Pareto(3)": (-12, 12)}
    for l, r, e in out:
        dx, dy = off.get(l, (6, 4))
        ax.annotate(l, (r, e), textcoords="offset points", xytext=(dx, dy),
                    fontsize=8.6, color=C_BLUE if r < 0.9 else C_RED,
                    ha="left" if dx >= 0 else "right",
                    va="bottom" if dy >= 0 else "top")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-6, 300)
    ax.set_ylim(1.2, 400)
    ax.set_xlabel("工作点的局部尾厚度 r = 尾部概率 / 密度")
    ax.set_ylabel("被选中者的平均高估（以各自 σ 为单位）")
    ax.axhline(4.88, color="#333333", lw=1.0, ls=":")
    ax.text(1.6e-6, 5.1, "正态 4.88", fontsize=8.6, color="#333333", va="bottom")
    ax.legend(fontsize=9.0, loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.15, lw=0.6, which="both")
    ax.set_title("横轴跨 7 个数量级：尾巴越厚，倍数越大（M=100 万）", fontsize=10.6)
    fig.suptitle("图 6（补充）换分布：倍数会变，机制不变", fontsize=12.6, y=1.02)
    return savefig(fig, "figw6_tails"), out


def _Emax_generic(dist, M, lo=-1e6, hi=1e9, npts=400001):
    """E[max of M] = ∫_0^∞ F^{-1}(e^{-s/n}) e^{-s} ds（中点法，无蒙特卡洛误差）"""
    DS = 60.0 / npts
    s = (np.arange(npts) + 0.5) * DS
    q = np.clip(-np.expm1(-s / M), 1e-14, 1 - 1e-12)
    return float(np.sum(dist.isf(q) * np.exp(-s)) * DS)


# ============================================================================
def main():
    print("=" * 78)
    p1, gaps = fig_auction()
    print("图1 高估幅度：n=2 %.2f, n=5 %.2f, n=15 %.2f, n=30 %.2f"
          % (gaps[1], gaps[4], gaps[14], gaps[29]))

    p2, mm = fig_cloud()
    print("图2 被报告者：真值均值 %.2f，报告值均值 %.2f，虚增 %.2f"
          % (mm[0], mm[1], mm[1] - mm[0]))

    p3, tb = fig_overfit()
    print("图3 训练/测试（d=5,10）：%.3f/%.3f  %.3f/%.3f  乐观量 %.3f"
          % (tb[4][1], tb[4][2], tb[9][1], tb[9][2], tb[9][2] - tb[9][1]))

    p4, marks = fig_ruler()
    print("图4 关键点：" + "  ".join("M=%d→%.2f" % (m, v) for m, v in marks))

    p5 = fig_split()
    p6, rows = fig_tails()
    print("图6 各分布 r 与倍数：")
    for l, r, e in rows:
        print("    %-10s r=%8.3f  ×%6.2f" % (l, r, e))

    print("-" * 78)
    for p in (p1, p2, p3, p4, p5, p6):
        print("[saved] %s  (%.1f KB)" % (p, os.path.getsize(p) / 1024))


if __name__ == "__main__":
    main()
