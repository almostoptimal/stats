# -*- coding: utf-8 -*-
"""
Barra CNE5 风格风险模型 —— 可直接运行的完整实现
依赖: numpy, pandas

跑法:
    python barra_cne5.py

流程:
    descriptors -> 预处理(缩尾/填充/标准化/合成/正交化)
              -> 截面WLS -> {f_t} 与 {u_t}
              -> F: EWMA(波动42/相关200) + Newey-West + Eigenfactor调整 + VRA
              -> D: EWMA(42) + 自由度修正 + Vasicek收缩 + VRA
              -> Sigma = X F X' + D

参数全部按 MSCI 官方口径 (CNE5 Empirical Notes / USE4 Methodology Notes /
GEMTR Datasheet)，可直接改 P 类里的常量做敏感性测试。
"""
import numpy as np
import pandas as pd

np.random.seed(7)


# ============================================================
# 0. 全局参数 —— 全部来自 MSCI 公开文档，单位统一为「交易日」
# ============================================================
class P:
    # ---- 因子协方差 F ----
    HL_VOL      = 42    # 因子波动率半衰期        (GEMTR datasheet)
    NW_LAG_VOL  = 2     # 波动率 Newey-West 滞后阶
    HL_CORR     = 200   # 因子相关性半衰期
    NW_LAG_CORR = 1     # 相关性 Newey-West 滞后阶
    HL_VRA_F    = 42    # 因子 VRA 半衰期  (USE4S)
    #   USE4L 的对应值: 波动252 / 相关504 / VRA 168 —— 长周期模型用这一组

    # ---- 特异方差 D ----
    HL_SPEC     = 42    # 特异波动半衰期
    NW_LAG_SPEC = 0     # 特异方差 Newey-West 滞后阶
    HL_VRA_SPEC = 4     # 特异 VRA 半衰期 (比因子侧短很多, 反应更快)
    SHRINK      = 0.10  # Vasicek 贝叶斯收缩强度 (GEMTR: 0.1)

    # ---- descriptor 窗口 ----
    BETA_WIN,   BETA_HL   = 252, 63    # Beta: 252日窗口, 半衰期63
    RSTR_WIN,   RSTR_HL   = 504, 126   # Momentum: 504日窗口, 半衰期126
    RSTR_SKIP             = 21         # 动量跳过最近21日(防短期反转)
    DASTD_WIN,  DASTD_HL  = 252, 42    # 日波动率
    HSIGMA_WIN, HSIGMA_HL = 252, 63    # Beta回归残差波动
    CMRA_MONTHS           = 12         # 累积收益振幅, 12个月
    STOM_WIN              = 21         # 月换手率, 21个交易日
    MONTH                 = 21         # 1个月 = 21个交易日

    # ---- 预处理 ----
    WINSOR_K    = 5.0   # MAD 缩尾倍数 (±5倍)
    MAD_SCALE   = 1.4826  # 使 MAD 与标准差可比的系数

    # ---- Eigenfactor 调整 ----
    N_SIM       = 200   # Monte Carlo 次数 (正式用 1000+)


# ============================================================
# 1. 通用工具
# ============================================================
def ewma(x, hl, axis=0):
    """指数加权移动平均，hl = 半衰期(期数)。返回与 x 同形状。"""
    x = np.asarray(x, dtype=float)
    lam = 0.5 ** (1.0 / hl)
    n = x.shape[axis]
    w = lam ** np.arange(n - 1, -1, -1)          # 越近期权重越大
    w = w / w.sum()
    return np.apply_along_axis(lambda v: np.dot(v, w), axis, x)


def ewma_series(x, hl):
    """逐点 EWMA(递推版)，返回每个时点的加权值序列，用于 VRA。"""
    x = np.asarray(x, dtype=float)
    lam = 0.5 ** (1.0 / hl)
    out = np.empty_like(x)
    acc = np.nan
    for t, v in enumerate(x):
        acc = v if (t == 0 or np.isnan(acc)) else lam * acc + (1 - lam) * v
        out[t] = acc
    return out


def winsorize_mad(s, k=P.WINSOR_K):
    """MAD 缩尾。不要用 ±3σ —— 厚尾会把 σ 撑大，极值根本去不掉。"""
    s = np.asarray(s, dtype=float)
    med = np.nanmedian(s)
    mad = np.nanmedian(np.abs(s - med)) * P.MAD_SCALE
    if mad <= 0 or np.isnan(mad):
        return s - med
    return np.clip(s, med - k * mad, med + k * mad)


def wmean_std(s, w):
    """市值加权均值与标准差（与 WLS 口径一致，不能用等权）。"""
    s = np.asarray(s, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = ~np.isnan(s)
    if ok.sum() < 2:
        return np.nan, np.nan
    s, w = s[ok], w[ok]
    sw = w.sum()
    mu = np.dot(w, s) / sw
    var = np.dot(w, (s - mu) ** 2) / sw
    return mu, np.sqrt(max(var, 1e-16))


def standardize(s, w):
    mu, sd = wmean_std(s, w)
    if np.isnan(sd) or sd <= 0:
        return np.zeros_like(np.asarray(s, dtype=float))
    return (s - mu) / sd


def wls_orthogonalize(y, X, w):
    """把 y 对 X 做加权回归，返回残差（正交化）。用于 RESVOL~Beta、NLSIZE~Size。"""
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones_like(y), np.asarray(X, dtype=float)])
    W = np.sqrt(np.asarray(w, dtype=float))
    Xw, yw = X * W[:, None], y * W
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return y - X @ beta


# ============================================================
# 2. 截面回归（阶段 A）
# ============================================================
def build_X(style_df, industry_code, w):
    """
    构造 N×K 暴露矩阵。列顺序固定:
        [市场(全1)] + [行业哑变量(已纯化)] + [10个风格]
    """
    N = len(style_df)
    market = np.ones((N, 1))

    # 行业哑变量纯化:  X̃_ij = D_ij - s_j,  s_j = 行业j的 sqrt(市值) 占比
    dummies = pd.get_dummies(industry_code).values.astype(float)
    s = (dummies * w[:, None]).sum(axis=0) / w.sum()   # 注意用 sqrt(cap) 权重
    ind = dummies - s[None, :]
    ind = ind[:, 1:]          # 丢一列已无必要(纯化后已正交), 但保留秩更直观 -> 丢掉第一列
    # 说明: 纯化后 Σ_j X̃_ij = 0, 与市场列不共线, 无需再丢列

    style = style_df.values.astype(float)
    return np.column_stack([market, ind, style]), ["MKT"] + \
           [f"IND_{j}" for j in range(ind.shape[1])] + list(style_df.columns)


def cross_section_regression(X, r, w):
    """
    一期截面 WLS:  f_hat = (X'WX)^{-1} X'W r
    数值上不要显式求逆 —— 用加权后的最小二乘。
    返回 (f, resid)。秩不足时自动加 ridge 兜底。
    """
    X = np.asarray(X, dtype=float)
    r = np.asarray(r, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = ~np.isnan(r) & ~np.isnan(X).any(axis=1)
    X, r, w = X[ok], r[ok], w[ok]

    sw = np.sqrt(w)
    Xw, rw = X * sw[:, None], r * sw
    try:
        f, *_ = np.linalg.lstsq(Xw, rw, rcond=None)
    except np.linalg.LinAlgError:
        ridge = 1e-8 * np.eye(X.shape[1])
        f = np.linalg.solve(Xw.T @ Xw + ridge, Xw.T @ rw)
    resid = r - X @ f
    return f, resid, ok


# ============================================================
# 3. 因子协方差 F（阶段 B）
# ============================================================
def newey_west(m0, cross_terms, lags):
    """
    HAC 修正: Var = Γ_0 + Σ_{l=1}^{L} (1 - l/(L+1)) (Γ_l + Γ_l')
    cross_terms: list, 第 l 项 = 滞后 l 阶的(加权)交叉矩矩阵
    """
    out = m0.copy()
    for l, g in enumerate(cross_terms, start=1):
        bartlett = 1.0 - l / (lags + 1.0)
        out += bartlett * (g + g.T)
    return out


def factor_covariance(F_ret, dates=None):
    """
    输入 F_ret: T×K 的因子收益矩阵(日频)
    输出: (F, 诊断字典)

    四步: EWMA(波动/相关分开半衰期) -> Newey-West -> Eigenfactor -> VRA
    """
    F_ret = np.asarray(F_ret, dtype=float)
    T, K = F_ret.shape
    Fr = F_ret - ewma(F_ret, P.HL_CORR, axis=0)[None, :]   # 去均值(EWMA)

    # ---- Step 1: 波动率(短半衰期) ----
    lam_v = 0.5 ** (1.0 / P.HL_VOL)
    wv = lam_v ** np.arange(T - 1, -1, -1); wv /= wv.sum()
    M0_v = np.einsum('t,tk,kl->kl', wv, Fr * Fr, np.eye(K))   # 对角
    g_v = []
    for l in range(1, P.NW_LAG_VOL + 1):
        cross = np.zeros((K, K))
        for i in range(K):
            cross[i, i] = np.dot(wv[l:], Fr[l:, i] * Fr[:-l, i])
        g_v.append(cross)
    var_v = np.diag(newey_west(M0_v, g_v, P.NW_LAG_VOL)).copy()
    vol = np.sqrt(np.maximum(var_v, 1e-16))

    # ---- Step 2: 相关性(长半衰期), 用标准化后的收益 ----
    Z = Fr / vol[None, :]
    lam_c = 0.5 ** (1.0 / P.HL_CORR)
    wc = lam_c ** np.arange(T - 1, -1, -1); wc /= wc.sum()
    M0_c = np.einsum('t,ti,tj->ij', wc, Z, Z)
    g_c = []
    for l in range(1, P.NW_LAG_CORR + 1):
        g_c.append(np.einsum('t,ti,tj->ij', wc[l:], Z[l:], Z[:-l]))
    R = newey_west(M0_c, g_c, P.NW_LAG_CORR)
    d = 1.0 / np.sqrt(np.maximum(np.diag(R), 1e-16))
    R = (R * d[:, None]) * d[None, :]                  # 归一到 diag=1
    np.fill_diagonal(R, 1.0)

    # ---- Step 2b: 拼回协方差 F_ij = ρ_ij σ_i σ_j ----
    F = (R * vol[:, None]) * vol[None, :]

    # ---- Step 3: Eigenfactor Risk Adjustment (Monte Carlo) ----
    F_adj, bias_ratio = eigenfactor_adjust(F, T, K)

    # ---- Step 4: VRA ----
    B_f, F_vra = vra_adjust_factor(F_ret, F_adj, P.HL_VRA_F)

    diag = {"vol": vol, "corr": R, "eigen_bias": bias_ratio, "B_factor": B_f}
    return F_vra, diag


def eigenfactor_adjust(F, T, K, n_sim=P.N_SIM):
    """
    Shepard(2009): 优化组合的真实波动 ≈ 预测波动 / (1 - K/T)。
    估计误差使「小特征根方向」的风险被系统性低估 —— 而优化器最爱加仓这些方向。
    做法: 在特征因子上做 Monte Carlo, 比较「预测波动」与「模拟实现波动」。
    """
    lam, U = np.linalg.eigh(F)
    lam = np.maximum(lam, 1e-12)
    sqrt_lam = np.sqrt(lam)

    # 特征因子收益 g_t = U' f_t, 理论上协方差 = diag(lam)
    est_ratio = np.zeros(K)
    for _ in range(n_sim):
        g = np.random.randn(T, K) * sqrt_lam[None, :]      # 真实协方差 = diag(lam)
        # 用与真实流程相同的 EWMA 估计
        lam_s = 0.5 ** (1.0 / P.HL_VOL)
        wv = lam_s ** np.arange(T - 1, -1, -1); wv /= wv.sum()
        lam_hat = np.einsum('t,tk->k', wv, g * g)          # 估计的方差
        est_ratio += (lam_hat / lam) / n_sim

    # est_ratio[k] < 1 表示该方向被低估 -> 放大
    adj = np.where(est_ratio > 0, 1.0 / np.maximum(est_ratio, 1e-6), 1.0)
    lam_adj = lam * adj
    return (U * lam_adj[None, :]) @ U.T, est_ratio


def vra_adjust_factor(F_ret, F, hl_vra):
    """
    VRA: B = EWMA加权 mean( (f_kt / σ_kt^pred)^2 ), 理想值 1。
    注意: B 作用在方差上 —— 标准差要乘 sqrt(B), 所以协方差矩阵乘 B。
    """
    pred_var = np.diag(F).copy()
    pred_var = np.maximum(pred_var, 1e-18)
    ratio = (F_ret ** 2) / pred_var[None, :]         # T×K
    B = ewma_series(np.nanmean(ratio, axis=1), hl_vra)[-1]
    B = float(np.clip(B, 1e-6, 100.0))
    return B, F * B


# ============================================================
# 4. 特异方差 D（阶段 B）
# ============================================================
def specific_variance(U_res, w_cap, K, group=None):
    """
    U_res: T×N 残差矩阵;  K = 因子个数
    步骤: EWMA(42) -> 自由度修正 N/(N-K) -> Vasicek 收缩 -> VRA
    """
    U_res = np.asarray(U_res, dtype=float)
    T, N = U_res.shape

    # Step 1: EWMA 方差
    lam = 0.5 ** (1.0 / P.HL_SPEC)
    wv = lam ** np.arange(T - 1, -1, -1); wv /= wv.sum()
    var_raw = np.einsum('t,tn->n', wv, U_res ** 2)
    var_raw = np.nan_to_num(var_raw, nan=np.nanmean(var_raw))

    # Step 2: 自由度修正 —— 样本内残差被 K 个因子吸走了噪声
    dof = N / max(N - K, 1)
    var_dof = var_raw * dof

    # Step 3: Vasicek 收缩 —— 向「市值分位 × 行业」组内均值收缩
    if group is None:
        # 默认按市值分 10 组
        q = pd.qcut(pd.Series(w_cap).rank(method="first"), 10, labels=False)
        group = q.values
    prior = np.zeros(N)
    shrink = np.full(N, P.SHRINK)
    for g in np.unique(group):
        m = group == g
        prior[m] = np.mean(var_dof[m])
    var_shrunk = (1 - shrink) * var_dof + shrink * prior

    return np.maximum(var_shrunk, 1e-12), {"dof": dof, "prior": prior}


def vra_adjust_specific(U_res, D, hl_vra=P.HL_VRA_SPEC):
    """特异方差侧的 VRA，半衰期比因子侧短(4天)，反应更快。"""
    pred = np.maximum(D, 1e-18)
    ratio = (U_res ** 2) / pred[None, :]
    B = ewma_series(np.nanmean(ratio, axis=1), hl_vra)[-1]
    B = float(np.clip(B, 1e-6, 100.0))
    return B, D * B


# ============================================================
# 5. 风险分解与自检
# ============================================================
def risk_decomposition(w_p, X, F, D):
    """把组合方差拆成因子部分与特异部分。"""
    xp = w_p @ X                       # K×1 组合因子暴露
    var_factor = float(xp @ F @ xp)
    var_spec = float(np.dot(w_p ** 2, D))
    return var_factor, var_spec, np.sqrt(var_factor + var_spec)


def bias_statistic(port_ret, pred_sigma, hl=250):
    """
    唯一的验收标准: B = mean((r_t / σ_{t|t-1})^2) ≈ 1
    B > 1 -> 低估波动; B < 1 -> 高估
    """
    r = np.asarray(port_ret, dtype=float)
    s = np.asarray(pred_sigma, dtype=float)
    ok = (s > 1e-12) & ~np.isnan(r)
    return float(np.nanmean((r[ok] / s[ok]) ** 2))


# ============================================================
# 6. descriptor 计算（CNE5 口径）
# ============================================================
def desc_size(mktcap):
    """LNCAP = ln(总市值)"""
    return np.log(mktcap)


def desc_beta(ret, mkt_ret):
    """BETA: 252日窗口, 半衰期63, 对市值加权市场超额收益的时序回归斜率"""
    T, N = ret.shape
    out = np.full(N, np.nan)
    lam = 0.5 ** (1.0 / P.BETA_HL)
    for i in range(N):
        y = ret[-P.BETA_WIN:, i]
        x = mkt_ret[-P.BETA_WIN:]
        ok = ~np.isnan(y) & ~np.isnan(x)
        if ok.sum() < 60:
            continue
        y, x = y[ok], x[ok]
        w = lam ** np.arange(len(y) - 1, -1, -1)
        Xd = np.column_stack([np.ones_like(x), x])
        W = np.sqrt(w)
        b, *_ = np.linalg.lstsq(Xd * W[:, None], y * W, rcond=None)
        out[i] = b[1]
    return out


def desc_momentum(ret):
    """RSTR: 504日窗口, 跳过最近21日, 半衰期126, 超额对数收益之和"""
    T, N = ret.shape
    lo = max(0, T - P.RSTR_WIN)
    hi = T - P.RSTR_SKIP
    if hi <= lo:
        return np.full(N, np.nan)
    lr = np.log1p(np.clip(ret[lo:hi], -0.99, None))    # 对数收益, 可加
    lam = 0.5 ** (1.0 / P.RSTR_HL)
    w = lam ** np.arange(hi - lo - 1, -1, -1); w /= w.sum()
    return np.einsum('t,tn->n', w, lr)


def desc_dastd(ret):
    """DASTD: 252日日超额收益波动, 半衰期42"""
    T, N = ret.shape
    r = ret[-P.DASTD_WIN:]
    lam = 0.5 ** (1.0 / P.DASTD_HL)
    w = lam ** np.arange(len(r) - 1, -1, -1); w /= w.sum()
    mu = np.einsum('t,tn->n', w, np.nan_to_num(r))
    var = np.einsum('t,tn->n', w, (np.nan_to_num(r) - mu[None, :]) ** 2)
    return np.sqrt(var)


def desc_cmra(ret):
    """CMRA: 过去12个月累积收益的(最大 - 最小)"""
    T, N = ret.shape
    months = P.CMRA_MONTHS * P.MONTH
    r = np.nan_to_num(ret[-months:])
    cum = np.cumsum(r, axis=0)
    return cum.max(axis=0) - cum.min(axis=0)


def desc_hsigma(ret, mkt_ret, beta):
    """HSIGMA: Beta 回归残差的标准差, 252日, 半衰期63"""
    T, N = ret.shape
    r = ret[-P.HSIGMA_WIN:]
    x = mkt_ret[-P.HSIGMA_WIN:]
    lam = 0.5 ** (1.0 / P.HSIGMA_HL)
    w = lam ** np.arange(len(x) - 1, -1, -1); w /= w.sum()
    out = np.full(N, np.nan)
    for i in range(N):
        y = r[:, i]
        ok = ~np.isnan(y)
        if ok.sum() < 60:
            continue
        resid = y[ok] - (beta[i] * x[ok] if not np.isnan(beta[i]) else 0)
        out[i] = np.sqrt(np.dot(w[ok], resid ** 2) / w[ok].sum())
    return out


def desc_liquidity(volume, shares):
    """STOM / STOQ / STOA —— 一个月 = 21 个交易日"""
    T, N = volume.shape
    turn = volume / shares[None, :]                       # 日换手率

    def stom_at(end):
        s = turn[max(0, end - P.STOM_WIN):end]
        return np.log(np.nansum(s, axis=0) + 1e-12)

    stom = stom_at(T)
    # STOQ/STOA = 月度的几何平均
    def avg_over(n_months):
        vals = [np.exp(stom_at(T - m * P.MONTH)) for m in range(n_months)
                if T - (m + 1) * P.MONTH >= 0]
        if not vals:
            return np.full(N, np.nan)
        return np.log(np.mean(vals, axis=0) + 1e-12)

    return stom, avg_over(3), avg_over(12)


# ============================================================
# 7. 主流程
# ============================================================
def run_demo():
    print("=" * 66)
    print("Barra CNE5 最小完整实现 —— 模拟数据端到端自检")
    print("=" * 66)

    # ---- 模拟数据 ----
    N, T = 400, 500
    n_ind, n_style = 8, 10
    K = 1 + n_ind + n_style

    ind_code = np.random.randint(0, n_ind, N)
    cap = np.exp(np.random.normal(20, 1.2, N))
    w_reg = np.sqrt(cap)                       # 回归权重 = sqrt(市值)

    # 行业哑变量纯化: X̃_ij = D_ij - s_j,  s_j = 行业j的 sqrt(市值) 占比
    # 纯化后 Σ_j X̃_ij = 0 -> 秩为 n_ind-1 -> 必须丢掉一列, 否则矩阵奇异
    dum = np.eye(n_ind)[ind_code]
    s_j = (dum * w_reg[:, None]).sum(axis=0) / w_reg.sum()
    ind_pur = (dum - s_j[None, :])[:, 1:]

    # 暴露矩阵(固定)
    X_static = np.column_stack([
        np.ones(N),
        ind_pur,
        np.random.randn(N, n_style),
    ])
    K = X_static.shape[1]                      # = 1 + (n_ind-1) + n_style

    # 真实因子协方差: 用「相关矩阵 ⊙ 波动外积」构造, 保证半正定
    C = np.corrcoef(np.random.randn(K, K + 20))
    vol_true = np.abs(np.random.randn(K)) * 0.004 + 0.002
    vol_true[0] = 0.010                        # 市场因子日波动 1%
    F_true = C * np.outer(vol_true, vol_true) + 1e-10 * np.eye(K)

    f_true = np.random.multivariate_normal(np.zeros(K), F_true, size=T)
    sigma_true = 0.008 + 0.02 * (cap.max() / cap) ** 0.5   # 小市值残差大
    sigma_true = sigma_true / sigma_true.mean() * 0.028    # 调到 A 股个股日波动量级

    r = f_true @ X_static.T + np.random.randn(T, N) * sigma_true[None, :]

    # ---- 逐期截面回归 ----
    K_mat = X_static.shape[1]
    F_ret = np.full((T, K_mat), np.nan)
    U_res = np.full((T, N), np.nan)
    for t in range(T):
        f, resid, _ = cross_section_regression(X_static, r[t], w_reg)
        F_ret[t] = f
        U_res[t] = resid

    print(f"\n[阶段A] 截面回归完成: N={N}, T={T}, K={K_mat}")
    print(f"  因子收益均值(年化%): {np.round(np.nanmean(F_ret, 0)*252*100, 2)[:5]}")
    R2 = 1 - np.nansum(U_res**2) / np.nansum((r - np.nanmean(r, axis=1)[:, None]) ** 2)
    print(f"  个股层面日频截面 R² = {R2:.3f}   <- 70%左右的波动在特异性里, 设计使然")

    # ---- 因子协方差 ----
    F_hat, diag = factor_covariance(F_ret)
    print(f"\n[阶段B] 因子协方差")
    print(f"  Eigenfactor 偏差比 (应≈1, <1 表示被低估): "
          f"{np.round(diag['eigen_bias'][:5], 3)}")
    print(f"  VRA 偏差统计量 B = {diag['B_factor']:.3f}   (目标 ≈ 1)")

    # ---- 特异方差 ----
    D_hat, sd = specific_variance(U_res, cap, K_mat)
    B_s, D_hat = vra_adjust_specific(U_res, D_hat)
    print(f"\n[阶段B] 特异方差")
    print(f"  自由度修正因子 = {sd['dof']:.4f}")
    print(f"  VRA 偏差统计量 B = {B_s:.3f}   (目标 ≈ 1)")

    # ---- 组装并做组合级自检 ----
    Sigma = X_static @ F_hat @ X_static.T + np.diag(D_hat)
    print(f"\n[组装] Sigma = X F X' + D,  形状 {Sigma.shape}, "
          f"对称误差 {np.abs(Sigma - Sigma.T).max():.2e}")

    # 随机组合 + 最小方差组合的 bias statistic
    rng = np.random.default_rng(0)
    rows = []
    for name, wp in [
        ("等权组合", np.ones(N) / N),
        ("市值加权", cap / cap.sum()),
        ("随机组合", rng.dirichlet(np.ones(N))),
    ]:
        pred_var = float(wp @ Sigma @ wp)
        pred_sd = np.sqrt(pred_var)
        port_r = r @ wp
        B = bias_statistic(port_r, np.full(T, pred_sd))
        vf, vs, _ = risk_decomposition(wp, X_static, F_hat, D_hat)
        rows.append((name, pred_sd * np.sqrt(252) * 100,
                     port_r.std() * np.sqrt(252) * 100, B, vf / (vf + vs)))
    print(f"\n[自检] {'组合':<10}{'预测年化波动%':>14}{'实现年化波动%':>14}"
          f"{'B':>8}{'因子风险占比':>12}")
    for nm, pv, rv, B, fr in rows:
        print(f"  {nm:<10}{pv:>14.2f}{rv:>14.2f}{B:>8.2f}{fr:>12.1%}")

    print("\n判定: B 应落在 0.8 ~ 1.25 之间。偏离大就回去调半衰期 / VRA / 收缩强度。")
    print("注意: 分散化组合的『因子风险占比』必然很高 —— 特异风险早被分散掉了,")
    print("      这不代表模型有问题; 单看一只个股时特异性通常占 60%~80%。")
    print("=" * 66)


if __name__ == "__main__":
    import io, sys, os
    _buf, _old = io.StringIO(), sys.stdout
    sys.stdout = _buf
    try:
        run_demo()
    finally:
        sys.stdout = _old
    _txt = _buf.getvalue()
    _out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "_tmp", "_barra_out.txt")
    with io.open(_out, "w", encoding="utf-8") as fh:
        fh.write(_txt)
    print(_txt)
