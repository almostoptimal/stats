# 互联网新用户 LTV 长尾建模：理论与实务

> 整理自“新用户 LTV 建模分析”对话，并补全公式、适用条件与工程边界。  
> 核心问题：在零值多、收入右偏、尾部稀疏、标签未成熟且分布漂移的条件下，估计业务真正需要的长期价值。  
> 本文是方法笔记，不包含真实业务数据上的实验结论。

## 目录

1. [定义目标与统计本质](#1-定义目标与统计本质)
2. [Estimand：均值、中位数与排序](#2-estimand均值中位数与排序)
3. [Hurdle model：零值与正值分开建模](#3-hurdle-model零值与正值分开建模)
4. [Gamma deviance：推导、性质与梯度](#4-gamma-deviance推导性质与梯度)
5. [MSE、log-MSE、Gamma 与 Tweedie](#5-mselog-msegamma-与-tweedie)
6. [Shrinkage、partial pooling 与 Empirical Bayes](#6-shrinkagepartial-pooling-与-empirical-bayes)
7. [Tail decomposition：保留尾部的期望贡献](#7-tail-decomposition保留尾部的期望贡献)
8. [Censoring、survival 与收入过程](#8-censoringsurvival-与收入过程)
9. [多时间尺度与多任务学习](#9-多时间尺度与多任务学习)
10. [评估：校准、排序、总量与不确定性](#10-评估校准排序总量与不确定性)
11. [工程落地路线](#11-工程落地路线)
12. [常见误区与参考资料](#12-常见误区与参考资料)

## 1. 定义目标与统计本质

### 1.1 先定义预测时点和收入口径

设注册后第 $`a`$ 天作预测，$`\mathcal H_a`$ 是当时实际可获得的信息。以注册后 $`T`$ 天为固定终点：

```math
Y_T=\sum_{t=1}^{T}d_tR_t,\qquad
m_{a,T}(\mathcal H_a)=E[Y_T\mid\mathcal H_a].
```

其中 $`R_t`$ 为日收入或贡献利润，$`d_t`$ 为可选折现系数。已发生部分无需预测：

```math
E[Y_T\mid\mathcal H_a]
=
Y_a+E[Y_T-Y_a\mid\mathcal H_a].
```

必须区分“注册后 180 天总价值”和“从今天起未来 180 天价值”。还要固定币种、税费、平台分成、退款、广告归因窗口与入账延迟。GMV、收入、毛利、净利润不是同一标签。

下文主要假设 $`Y\ge0`$。若退款使净收入为负，可分别预测正向收入与退款金额再相减；不能直接把负数送入 Gamma 或 $`1<p<2`$ 的 Tweedie。

### 1.2 业务长尾不等于严格的重尾分布

LTV 的难点往往来自以下因素叠加：

- 大量零值或低值，少数 whale 用户贡献较多收入。
- 不同人群混合：付费倾向、频次、客单价与生命周期不同。
- 条件异方差：高均值群体通常也有更大的收入波动。
- 新用户特征弱，未来行为含有大量不可预测随机性。
- 长期标签尚未成熟，成熟样本又可能偏旧。
- 渠道、产品、价格和投放政策变化造成分布漂移。

“直方图右偏”不足以证明幂律或无限方差。Gamma 是右偏但指数衰减的轻尾分布；用 Gamma loss 并不等于假设真实数据是 Pareto 重尾。

若确有尾部近似 $`P(Y>y)\sim Cy^{-\alpha}`$，则通常 $`\alpha>1`$ 才有有限均值，$`\alpha>2`$ 才有有限方差。业务上有限上界也可能使矩存在，但有限样本仍极不稳定。

### 1.3 条件噪声与可学习信号

在二阶矩存在时：

```math
\mathrm{Var}(Y)
=
\mathrm{Var}(E[Y\mid X])
+
E[\mathrm{Var}(Y\mid X)].
```

第一项反映特征可解释的差异；第二项包含现有信息下的随机性。增加模型容量不能凭空恢复未被 $`X`$ 记录的信息。

例如某类用户有 1% 概率产生 50,000 元收入，否则为零，那么条件均值为 500 元。对后来实际消费 50,000 元的用户预测 500，不足以单独证明模型错误。

大样本下均值可能更稳定，但用户间共同冲击、极重尾和渠道漂移会削弱聚合带来的好处。收入可加性不要求独立；“误差会相互抵消”则需要额外条件。

## 2. Estimand：均值、中位数与排序

**Loss 的选择改变了估计对象，不只是改变优化难度。**

| 业务问题 | 目标 estimand | 常用方法 | 主要边界 |
|---|---|---|---|
| 收入预算、期望价值、风险中性出价 | $`E[Y\mid X]`$ | MSE、Gamma/Tweedie deviance、Hurdle | 要做金额校准 |
| 典型用户会消费多少 | 条件中位数 | MAE | 不能代替均值汇总 |
| 上下行情景 | 条件分位数 $`Q_\tau(Y\mid X)`$ | Pinball loss | 分位数不能直接相加成组合分位数 |
| 找出超过阈值的用户 | $`P(Y>c\mid X)`$ | 二分类 | 不描述超过阈值后的金额 |
| 固定名额下捕获最多期望收入 | 按 $`E[Y\mid X]`$ 排序 | 均值模型或适当排序目标 | 成本不同或存在干预效应时需改目标 |

平方损失在有限二阶矩条件下满足：

```math
E[(Y-a)^2\mid X]
=
\mathrm{Var}(Y\mid X)
+
(a-E[Y\mid X])^2.
```

因此最优解为条件均值。MAE 最优解是条件中位数；pinball loss

```math
\rho_\tau(u)=u\{\tau-\mathbf1(u<0)\}
```

的最优解为条件 $`\tau`$ 分位数。均值存在而二阶矩无限时，上述有限风险分解不能直接使用。

排序也没有唯一的 estimand：A 群体必消费 100，B 群体以 10% 概率消费 2,000。按超过 50 元的概率，A 更高；按期望收入，B 的 200 更高。

若目的是决定是否补贴，真正相关的可能是增量收益：

```math
E[Y(1)-Y(0)\mid X]-\mathrm{Cost}(X),
```

而非既有策略下的 LTV。高价值用户不一定对补贴有更高响应。

## 3. Hurdle model：零值与正值分开建模

令 $`Z=\mathbf1(Y>0)`$，建立：

```math
p(X)=P(Z=1\mid X),\qquad
\mu_+(X)=E[Y\mid Z=1,X].
```

由全期望公式，精确得到：

```math
\boxed{E[Y\mid X]=p(X)\mu_+(X)}.
```

例如付费概率为 0.1，付费后的平均收入为 500，则总体期望收入为 50。推断时用概率相乘，不能先把概率阈值化成 0 或 1。

若正值密度为 $`f_+(y\mid X)`$，则单样本负对数似然为：

```math
\ell
=
-(1-Z)\log(1-p)
-Z\log p
-Z\log f_+(Y\mid X).
```

### 实务结构

1. 所有标签成熟用户训练付费分类器。
2. 仅在 $`Y>0`$ 用户上训练正值金额模型，如 Gamma 回归。
3. 对所有新用户计算 $`\hat p(X)\hat\mu_+(X)`$。
4. 分别检查概率校准、正值金额校准，并最终检查乘积在全体用户上的校准。

两个部分可以独立训练，也可共享表征、设置不同输出头。第二阶段若使用 log-MSE 或 Huber，其输出通常不是 $`\mu_+`$，必须明确修正或重新定义目标。两个子模型各自粗粒度校准，也不保证乘积在任意分组上都校准。

### Hurdle 与 zero-inflated 的区别

Hurdle 的零值全部来自门槛过程，正值部分只支持 $`y>0`$。如果原分布包含零，需要正截断；Gamma 本来就只支持正值，不必额外截断。

Zero-inflated 模型允许基础过程也产生零。例如：

```math
P(Y=0)=\pi+(1-\pi)e^{-\lambda}
```

是 zero-inflated Poisson 的零概率。两种机制不能混为一谈。

Hurdle 能让变现概率与金额使用不同特征和结构，但会增加估计误差、维护成本，正值样本少时尤其需要收缩；它不是必然优于单模型的定理。

## 4. Gamma deviance：推导、性质与梯度

### 4.1 从 Gamma likelihood 出发

采用均值—离散度参数化：

```math
Y\mid X\sim\mathrm{Gamma}
\left(k=\frac1\phi,\;\theta=\phi\mu(X)\right),
```

其中 $`k`$ 为 shape、$`\theta`$ 为 scale，$`\phi>0`$。于是：

```math
E[Y\mid X]=\mu,\qquad
\mathrm{Var}(Y\mid X)=\phi\mu^2.
```

$`\phi`$ 固定时，变异系数 $`\mathrm{SD}(Y\mid X)/E[Y\mid X]=\sqrt\phi`$。

密度为：

```math
f(y\mid\mu,\phi)
=
\frac{y^{1/\phi-1}e^{-y/(\phi\mu)}}
{\Gamma(1/\phi)(\phi\mu)^{1/\phi}},\qquad y>0.
```

取对数并分离与 $`\mu`$ 无关的项：

```math
\log f(y\mid\mu,\phi)
=
C(y,\phi)-\frac1\phi\left(\frac y\mu+\log\mu\right).
```

在固定 $`\phi`$ 下，优化均值等价于最小化：

```math
L(y,\mu)=\frac y\mu+\log\mu.
```

### 4.2 Unit deviance 与 scaled deviance

饱和模型对每个样本设 $`\mu=y`$。定义 Gamma **unit deviance**：

```math
\boxed{
d(y,\mu)=2\left[\frac y\mu-1-\log\frac y\mu\right]
},\qquad y,\mu>0.
```

固定离散度时，单样本的两倍对数似然差是：

```math
2\{\ell(y;y,\phi)-\ell(y;\mu,\phi)\}
=
\frac{d(y,\mu)}{\phi}.
```

因此 unit deviance 与按离散度缩放后的似然比相差 $`1/\phi`$。使用“deviance”一词时应注明约定，不能在推导中无说明地丢掉 $`\phi`$。

### 4.3 为什么目标仍是算术均值

固定 $`X=x`$，记 $`m=E[Y\mid X=x]\in(0,\infty)`$：

```math
R(\mu)=E[L(Y,\mu)\mid X=x]
=\frac m\mu+\log\mu,
```

```math
R'(\mu)=\frac{\mu-m}{\mu^2}.
```

当 $`\mu<m`$ 时导数为负，当 $`\mu>m`$ 时为正，唯一最优解：

```math
\boxed{\mu^*(x)=E[Y\mid X=x]}.
```

该结论不要求真实条件分布严格服从 Gamma。若直接讨论 $`E[d(Y,\mu)\mid X]`$ 的有限性，还需控制 $`\log Y`$ 的可积性，例如 $`E[|\log Y|\mid X]<\infty`$。

这是不受限函数空间中的总体最优性质。在有限样本、受限模型、正则化或模型失配下，Gamma 与 MSE 可以得到不同预测，也都不自动保证校准。

### 4.4 相对尺度、局部近似与非对称性

对任意 $`c>0`$：

```math
d(cy,c\mu)=d(y,\mu).
```

令 $`\delta=(y-\mu)/\mu`$，则：

```math
d(y,\mu)=2\{\delta-\log(1+\delta)\}
=\delta^2-\frac23\delta^3+O(\delta^4).
```

预测接近真实值时，它近似平方相对误差；这不等于使用真实值作分母的 MAPE 或相对平方损失。

| 真实值 $`y`$ | 预测值 $`\mu`$ | $`y/\mu`$ | Gamma unit deviance |
|---:|---:|---:|---:|
| 100 | 50 | 2 | 0.6137 |
| 100,000 | 50,000 | 2 | 0.6137 |
| 100 | 200 | 0.5 | 0.3863 |

低估为真实值的一半与高估为真实值的两倍，惩罚不同。非对称性来自公式，不意味着业务已经显式定义了不同的高估、低估成本。

### 4.5 梯度、Hessian 与 link function

对 $`L=y/\mu+\log\mu`$，直接以 $`\mu`$ 为输出：

```math
\frac{\partial L}{\partial\mu}=\frac{\mu-y}{\mu^2},
\qquad
\frac{\partial^2L}{\partial\mu^2}=\frac{2y-\mu}{\mu^3}.
```

因此 $`L`$ 对 $`\mu`$ 并非全局凸。

实际常使用 log link：$`\mu=e^\eta,\;\eta=f_\theta(X)`$。注意 log link 是常用选择，Gamma 的 canonical link 是 inverse link（符号依参数约定）。

```math
L(y,\eta)=ye^{-\eta}+\eta,
```

```math
\boxed{g_\eta=1-\frac y\mu},\qquad
\boxed{h_\eta=\frac y\mu>0}.
```

对单个标量 $`\eta`$，损失严格凸；这不意味着神经网络参数 $`\theta`$ 的优化问题凸。Unit deviance 的梯度和 Hessian 是上述值的两倍；完整负 log-likelihood 再乘 $`1/\phi`$。

比较梯度必须使用相同输出坐标。以 $`\frac12(y-\mu)^2`$ 为平方损失：

| 损失 | 对 $`\mu`$ 的梯度 | 对 $`\eta=\log\mu`$ 的梯度 |
|---|---|---|
| Half MSE | $`\mu-y`$ | $`\mu(\mu-y)`$ |
| Gamma $`L`$ | $`(\mu-y)/\mu^2`$ | $`1-y/\mu`$ |

不能直接拿 MSE 对 $`\mu`$ 的梯度与 Gamma 对 $`\eta`$ 的梯度比较数值大小，就宣布后者一定更稳定。

### 4.6 Gamma 不是“自动抗 whale”

当 $`y/\mu\to\infty`$，$`|g_\eta|\to\infty`$，Gamma 的梯度没有被上界截断。

例如 $`y=100000,\mu=100`$ 时，$`g_\eta=-999`$。更关键的是：如果模型只预测一个常数，最小化 Gamma loss 的解仍是样本均值：

```math
\hat\mu=\frac1n\sum_i y_i.
```

所以它没有消除样本均值对极端值的敏感性，也无法凭空解决无限方差、尾部数据稀缺或不可预测 whale。它改变了不同预测尺度上的误差权重，而不是把均值估计变成了稳健位置估计。

### 4.7 零值与数值边界

Gamma deviance 要求 $`y>0,\mu>0`$，这一支持域也见 [scikit-learn Gamma deviance 文档](https://scikit-learn.org/1.9/modules/generated/sklearn.metrics.mean_gamma_deviance.html)。

把零值加一个很小的 $`\epsilon`$ 虽然能计算，但改变了标签与模型解释。对真实零收入，优先比较 Hurdle 和 Tweedie。省略 $`\log y`$ 后的形式在零值上看似有限，也不代表得到了合法的 Gamma 观测似然。

工程上可统一金额单位、监控 $`\eta`$ 与 $`y/\mu`$、设置合理初始化并控制优化步长。硬裁剪预测或梯度需要记录，因为它会改变实际优化行为；不应把标签裁剪当成纯数值处理。

## 5. MSE、log-MSE、Gamma 与 Tweedie

### 5.1 对照表

| 目标函数 | 标签支持 | 不受限总体目标 | 主要优势 | 主要风险 |
|---|---|---|---|---|
| MSE | 实数 | 条件均值，有限平方风险下 | 直接按金额误差优化 | 大误差主导 |
| MAE | 实数 | 条件中位数 | 对异常值较不敏感 | 不估计可加总的均值 |
| Huber | 实数 | Huber 位置泛函 | 平滑折中 | 一般不等于均值 |
| log-MSE | 正值 | $`E[\log Y\mid X]`$ | 压缩尺度 | 反变换存在偏差 |
| log1p-MSE | 非负值 | $`E[\log(1+Y)\mid X]`$ | 可包含零值 | expm1 后通常低估均值 |
| Gamma deviance | 严格正值 | 条件均值 | 尺度不变，均值导向 | 不支持零，仍受极端值影响 |
| Tweedie deviance，$`1<p<2`$ | 非负值 | 条件均值 | 零与正连续值统一建模 | 零概率与均值结构耦合 |

MSE 作为预测损失并不要求正态或同方差；只有把它解释为固定方差 Gaussian likelihood 时才引入相应分布假设。

### 5.2 log-MSE 的反变换

若训练 $`\log Y=f(X)+\varepsilon`$ 且 $`E[\varepsilon\mid X]=0`$，则：

```math
E[Y\mid X]
=
e^{f(X)}E[e^\varepsilon\mid X].
```

直接输出 $`e^{f(X)}`$ 是条件几何均值。在条件 lognormal 假设下：

```math
\log Y\mid X\sim N(a(X),s^2(X)),
```

```math
\mathrm{Median}(Y\mid X)=e^{a(X)},\qquad
E[Y\mid X]=e^{a(X)+s^2(X)/2}.
```

一般情况下，几何均值不必等于中位数。对 log1p 模型，同理：

```math
E[Y\mid X]
=
e^{f(X)}E[e^\varepsilon\mid X]-1.
```

可用留出或交叉拟合残差估计 smearing factor。全局常数修正只有在该因子基本不依赖 $`X`$ 时才合理；异方差时应考虑条件修正，并单独验证尾部和渠道校准。

### 5.3 Tweedie 的方差函数与 compound Poisson-Gamma

Tweedie exponential dispersion family 的方差函数为：

```math
\mathrm{Var}(Y\mid X)=\phi\mu^p.
```

$`p=0`$ 对应 Gaussian，$`p=1`$ 对应 Poisson 类型，$`p=2`$ 对应 Gamma；$`1<p<2`$ 对应 compound Poisson-Gamma。支持域和参数对应关系可参见 [Tweedie deviance 官方文档](https://scikit-learn.org/dev/modules/generated/sklearn.metrics.mean_tweedie_deviance.html)。

在 $`1<p<2`$ 时，可以构造：

```math
N\sim\mathrm{Poisson}(\lambda),\qquad
Y=\sum_{j=1}^{N}A_j,\qquad A_j\sim\mathrm{Gamma}.
```

标准参数化下：

```math
\lambda=\frac{\mu^{2-p}}{\phi(2-p)},\qquad
P(Y=0)=e^{-\lambda}.
```

因此固定 $`\phi,p`$ 时，零概率与均值不是独立自由变化的。Hurdle 则允许 $`p(X)`$ 和 $`\mu_+(X)`$ 分别建模。

### 5.4 Tweedie deviance 的均值性质与梯度

对 $`p\ne1,2`$：

```math
d_p(y,\mu)
=
2\left[
\frac{y^{2-p}}{(1-p)(2-p)}
-\frac{y\mu^{1-p}}{1-p}
+\frac{\mu^{2-p}}{2-p}
\right].
```

在 $`1<p<2`$ 时，$`y=0`$ 使用连续延拓。其导数为：

```math
\frac{\partial d_p}{\partial\mu}
=
2\frac{\mu-y}{\mu^p}.
```

固定 $`X`$ 后取期望，最优解仍是 $`\mu=E[Y\mid X]`$，前提是相关风险存在。

若 $`L_p=d_p/2,\;\mu=e^\eta`$：

```math
g_\eta=\mu^{2-p}-y\mu^{1-p},
```

```math
h_\eta=(2-p)\mu^{2-p}+(p-1)y\mu^{1-p}.
```

在 $`1<p<2`$、$`y\ge0`$ 时 Hessian 为正。令 $`p\to2`$，恢复 Gamma 的 $`g_\eta=1-y/\mu,\;h_\eta=y/\mu`$。

### 5.5 如何选择

建议在同一时间切分与特征集下比较原尺度 MSE、Tweedie、Hurdle + Gamma，以及 log1p 基线。log1p 可作为排序比较，但若用于收入预算必须评估反变换与金额校准。

可以用样本外预测分桶，观察正值用户的均值—方差关系。桶内方差还含有不同条件均值的混合，不能把桶间回归斜率直接视为 $`p`$ 的一致估计。

不同 $`p`$ 的原始 deviance 不能直接相互比较来选 $`p`$：其尺度和单位会变化。应使用固定的业务指标、固定 power 的共同评估损失，或正确归一化的 likelihood。最终选择依据是样本外表现，而非“直方图很右偏”。

## 6. Shrinkage、partial pooling 与 Empirical Bayes

### 6.1 为什么要收缩

小渠道只有 5 个用户，恰好出现一个 whale，其均值可能远高于大渠道。直接用局部均值做出价，会把抽样噪声当成稳定差异。

典型收缩形式：

```math
\tilde\mu_g=w_g\hat\mu_g+(1-w_g)\mu_0.
```

$`\mu_0`$ 应是合理的上层基线，如同国家、同渠道类型的均值；不是不加区分地拉向全站均值。$`w_g`$ 应反映估计可靠性，而非只看用户数。

### 6.2 Normal-normal 示例

作为局部估计的近似模型，假设：

```math
\hat\mu_g\mid\mu_g\sim N(\mu_g,v_g),\qquad
\mu_g\sim N(\mu_0,\tau^2).
```

则：

```math
E[\mu_g\mid\hat\mu_g]
=
\frac{\tau^2}{\tau^2+v_g}\hat\mu_g
+
\frac{v_g}{\tau^2+v_g}\mu_0.
```

若 $`v_g=\sigma^2/n_g`$，则：

```math
w_g=\frac{n_g}{n_g+\sigma^2/\tau^2}.
```

局部噪声大时收缩多；组间真实差异大时收缩少。小样本重尾下正态近似和样本方差都可能不可靠，实际可采用正值层次模型或对 log-mean 设置层次先验。Normal-normal 只是解释公式，不是要求收入本身服从正态。

### 6.3 三种 pooling 与 Empirical Bayes

- **No pooling**：每个组独立估计，保留差异但方差可能大。
- **Complete pooling**：所有组共享参数，稳定但可能抹去差异。
- **Partial pooling**：通过上层分布共享信息，保留有证据支持的差异。

Empirical Bayes 用跨组数据估计 $`\mu_0,\tau^2`$ 等超参数，再做组内后验推断；完整 Bayes 还给超参数先验并传播其不确定性。收缩是更宽泛的思想，正则化与 partial pooling 是常见实现。概念对照见 [Stan partial pooling 案例](https://mc-stan.org/learn-stan/case-studies/pool-binary-trials.html)。

### 6.4 对付费概率做 Beta-binomial 收缩

若渠道 $`g`$ 中有 $`s_g`$ 个付费用户、$`n_g`$ 个总用户：

```math
s_g\mid p_g\sim\mathrm{Binomial}(n_g,p_g),
\qquad p_g\sim\mathrm{Beta}(a,b),
```

```math
E[p_g\mid s_g,n_g]
=
\frac{s_g+a}{n_g+a+b}
=
\frac{n_g}{n_g+a+b}\frac{s_g}{n_g}
+
\frac{a+b}{n_g+a+b}\frac a{a+b}.
```

$`a+b`$ 可理解为先验有效样本量。在 Hurdle 中可分别收缩付费率、正值金额和尾部金额。

对联合 Bayesian Hurdle，严格的后验均值应计算：

```math
E[p_g\mu_{+,g}\mid D]
=
E[p_g\mid D]E[\mu_{+,g}\mid D]
+
\mathrm{Cov}(p_g,\mu_{+,g}\mid D).
```

若后验不独立，不能默认两个后验均值的乘积就是总收入的后验均值。

### 6.5 工程注意事项

按“全局—国家—渠道—campaign”建立有业务依据的层次；避免把真实不同人群过度合并。新渠道可回退到上层先验，并随着数据积累逐步更新。

Target encoding 也可使用收缩，但必须按训练折或时间窗口计算，禁止用验证、测试或未来收入参与编码。超参数与收缩强度同样只能在训练/验证流程中估计。

## 7. Tail decomposition：保留尾部的期望贡献

### 7.1 为什么不能机械 winsorize

把标签改为 $`Y'=\min(Y,c)`$ 后，模型目标变为：

```math
E[\min(Y,c)\mid X].
```

原均值被减去了 $`E[(Y-c)_+\mid X]`$。若尾部贡献可观，截断会系统性删除业务价值；更好训练不代表目标没有改变。

### 7.2 精确的 body + excess 分解

恒等式：

```math
Y=\min(Y,c)+(Y-c)_+
```

给出：

```math
\boxed{
E[Y\mid X]
=
E[\min(Y,c)\mid X]
+
P(Y>c\mid X)E[Y-c\mid Y>c,X]
}.
```

可训练三个部分：有上界的 body 均值、越阈概率、条件超额均值。阈值 $`c`$ 在训练数据中确定，检查多阈值敏感性，并对稀疏尾部做强收缩。

另一种互斥混合分解是：

```math
E[Y\mid X]
=
(1-q_c)\mu_{\le c}+q_c\mu_{>c}.
```

注意不要混用两种定义：$`\min(Y,c)`$ 已经包含尾部样本的 $`c`$ 元基底，若再加完整的尾部收入，会重复计数。

### 7.3 极值理论作为可选扩展

在适当的极值吸引域条件下，高阈值以上的超额可近似为广义 Pareto 分布（GPD）。这是有条件的渐近近似，不意味着任意业务阈值都适用，也不证明 body 与 tail 必然来自两个真实机制。

若 $`E=Y-c\mid Y>c,X`$ 服从 GPD，尺度 $`\beta(X)>0`$、形状 $`\xi(X)`$，则在 $`\xi<1`$ 时：

```math
E[E\mid X]=\frac{\beta(X)}{1-\xi(X)}.
```

$`\xi\ge1`$ 时该模型下均值不存在；$`\xi`$ 接近 1 时外推非常敏感。实际使用需检查阈值稳定性、尾部样本量、跨期稳定性、上界假设与不确定区间。不要仅凭少数 whale 拟合每个 campaign 独立的尾指数。

### 7.4 稀有类别采样与概率修正

设原始正类概率为 $`p`$，正、负类保留概率分别为 $`s_1,s_0`$，采样后概率为 $`p_s`$。若采样只依赖类别：

```math
\frac{p_s}{1-p_s}
=
\frac{s_1}{s_0}\frac p{1-p},
```

```math
\mathrm{logit}(p)
=
\mathrm{logit}(p_s)+\log\frac{s_0}{s_1}.
```

恢复先验后仍应在自然分布验证集上校准。类别权重、focal loss 或更复杂采样也可能影响概率；不能默认一个简单先验修正解决所有失配。

一般而言，用依赖 $`Y`$ 的权重 $`w(X,Y)`$ 优化均值型损失，目标可能变为：

```math
\frac{E[wY\mid X]}{E[w\mid X]}.
```

所以过采样 whale 后若不修正，金额预测可能严重偏高。

## 8. Censoring、survival 与收入过程

### 8.1 标签未成熟不等于零收入

注册 30 天的用户，其 180 天收入尚未知。已观察 $`Y_{30}`$ 不能作为完整 $`Y_{180}`$，否则把未来未观测收入错误填为零。

严格说，事件时间可能发生右删失；固定窗口累计收入则是随访不完整、未来增量未观测。二者相关，但不能直接把累计金额当成 survival 的事件时间。

成熟 cohort 训练易于解释，但样本滞后。近期 cohort 可用于已成熟的短期任务、逐期收入过程或显式的随访模型。

### 8.2 Survival likelihood

对明确的不可逆终止事件，设事件时间 $`T_d`$，删失时间 $`C`$：

```math
U=\min(T_d,C),\qquad
\Delta=\mathbf1(T_d\le C).
```

在适当的条件独立删失假设下，忽略删失机制的事件似然贡献为：

```math
L_i=f(U_i\mid X_i)^{\Delta_i}
S(U_i\mid X_i)^{1-\Delta_i}.
```

未观察到流失不等于永不流失；其贡献是生存概率。

但互联网“今天不活跃”可能明天回流，因此日活概率不是单调 survival curve。可用逐日活跃概率、多状态转移或隐状态模型。非合约场景中 alive 通常是潜变量，概率客户模型是可选路线，见 [Fader、Hardie 等的 BG/BB 研究](https://www.brucehardie.com/papers/020/)。

### 8.3 收入过程分解：何时可以相乘

离散时间下，若不活跃时收入为零：

```math
E[Y_T\mid X]
=
\sum_{t=1}^T d_t
P(A_t=1\mid X)
E[R_t\mid A_t=1,X].
```

这是全期望恒等式，无需把活跃和金额假设为独立。若不活跃时仍有订阅扣费，则必须加入 $`A_t=0`$ 时的收入项。

连续时间、终止后收入为零的模型可写：

```math
E[Y_T\mid X]
=
\int_0^T d(t)S(t\mid X)m_r(t\mid T_d>t,X)\,dt.
```

$`m_r`$ 是存活条件下的期望收入率，不应不加定义地等同于购买 hazard。

若用户购买 $`N`$ 次、单次金额为 $`V_j`$：

```math
E\left[\sum_{j=1}^N V_j\mid X\right]
=
E\left[
\sum_{j=1}^N E[V_j\mid N,X]\mid X
\right].
```

只有在相应条件均值稳定或独立等条件下，才可进一步化为 $`E[N\mid X]E[V\mid X]`$。如果 $`N`$ 已是整个生命周期的购买次数，再乘一个 lifetime 会重复计数。

广告收入若表示为曝光量 $`I_t`$ 乘 eCPM $`C_t`$，须除以 1000：

```math
E[R_t\mid A_t=1,X]
=
\frac{
E[I_t\mid A_t=1,X]E[C_t\mid A_t=1,X]
+\mathrm{Cov}(I_t,C_t\mid A_t=1,X)
}{1000}.
```

忽略协方差是额外近似，不是恒等式。

### 8.4 IPCW 与识别限制

令 $`O_t=\mathbf1(C\ge t)`$，$`G_t(X)=P(C\ge t\mid X)`$。若 $`O_t`$ 与收入增量在给定 $`X`$ 后独立，且 $`G_t(X)>0`$：

```math
E\left[\frac{O_tR_t}{G_t(X)}\mid X\right]
=
E[R_t\mid X].
```

这是逆删失概率加权的基本逻辑；具体 survival 权重实现可参见 [scikit-survival 文档](https://scikit-survival.readthedocs.io/en/v0.23.1/api/generated/sksurv.nonparametric.ipc_weights.html)。时变信息或信息性删失需要更完整建模，不能直接套边际权重。

若某批最近注册用户根本不可能被观察到第 180 天，则该区域 $`G_{180}=0`$。加权不能创造尚未发生的数据，只能依赖跨 cohort 可迁移假设、结构模型或等待成熟。权重过大时也会放大方差。

## 9. 多时间尺度与多任务学习

同时预测 $`Y_1,Y_3,Y_7,Y_{30},Y_{90},Y_{180}`$，并可加入留存、付费、会话数等辅助任务：

```math
L(\theta)
=
\sum_{h\in\mathcal T}
w_h
\frac{\sum_i M_{ih}\ell_h(y_{ih},\hat y_{ih})}
{\sum_iM_{ih}}.
```

$`M_{ih}=1`$ 表示该标签在训练快照时已成熟；分母为零的任务跳过。近期用户可以贡献短期监督，但不能把未成熟长期标签填零。

短期标签反馈快，可能改善共享表征；不过短期收入不必总有更高信噪比，与长期目标冲突时也可能负迁移。用消融实验选择任务及权重，避免大数值任务支配梯度。

### 累计收入的一致性

对于非负收入和相同信息集，应有：

```math
\hat m_7\le\hat m_{30}\le\hat m_{90}\le\hat m_{180}.
```

可预测互不重叠区间的非负增量，再累加：

```math
\hat m_{t_k}=\sum_{j=1}^{k}\hat\delta_j,\qquad \hat\delta_j\ge0.
```

若标签是可下降的净收入，则不应强制该约束。不同预测时点的信息集不同，预测更新值也不必随时间单调。

### 动态更新

D0、D1、D3、D7 可分别使用当时已到达的数据更新剩余价值。训练与离线回放必须重建对应时点的信息状态，包括日志延迟；不能用完整七日行为训练一个声称在注册瞬间出价的模型。

简单的 $`Y_{180}\approx kY_7`$ 会忽略晚付费用户，并假设跨 cohort 增长倍数稳定。可作基线，但要建模 $`Y_7=0`$ 人群并检查倍数漂移。

## 10. 评估：校准、排序、总量与不确定性

评估集必须保留自然零值率与尾部比例，并标明样本量、付费人数、标签成熟度。不同模型使用同一用户集合和预测时点。

### 10.1 校准与总量

按样本外预测分桶 $`B_k`$：

```math
\overline{\hat y}_k=\frac1{|B_k|}\sum_{i\in B_k}\hat y_i,\qquad
\bar y_k=\frac1{|B_k|}\sum_{i\in B_k}y_i.
```

比较两者并给出不确定区间。对渠道、国家、campaign、版本和注册周也做同样检查。

整体或分组偏差：

```math
\mathrm{BiasRatio}
=
\frac{\sum_i\hat y_i}{\sum_i y_i},
\qquad
\mathrm{RelativeBias}
=
\frac{\sum_i\hat y_i-\sum_i y_i}{\sum_i y_i}.
```

分母为零时应单独报告，不以随意加 $`\epsilon`$ 掩盖。整体校准可能来自高估和低估抵消，不能代替分组校准；常数均值模型也可能整体校准但毫无区分能力。

### 10.2 排序与尾部捕获

令 $`S_k`$ 为预测最高的 $`k`$ 比例用户：

```math
\mathrm{RevenueCapture}@k
=
\frac{\sum_{i\in S_k}y_i}{\sum_i y_i},
```

```math
\mathrm{Lift}@k
=
\frac{\mathrm{RevenueCapture}@k}{|S_k|/n}.
```

若 top 10% 用户贡献 60% 收入，则 lift 为 6。

另报实际高价值用户的 recall/precision，并明确真实 whale 阈值、预测选取比例和并列值规则。Spearman 关注次序而非金额贡献；NDCG 必须说明 gain 与折扣定义。

Hurdle 分类部分可评估 log loss、Brier score、PR-AUC 和概率校准；只有 AUC 高不足以支撑金额预测。

### 10.3 原尺度误差仍有用途

RMSE/MSE 能显示大金额误差，应该保留，但不能独占模型选择。MAE、RMSLE 可辅助诊断，不能因为它们更低就推断均值更准确。

Gamma deviance 只在正值标签上评估，对 Hurdle 应主要检查第二阶段；全体用户可使用共同的 Tweedie deviance，但还需检查最终金额总量。

建议主报告同时给出：

| 维度 | 最低报告内容 |
|---|---|
| 金额 | 总量偏差、渠道/cohort 偏差 |
| 校准 | 预测分桶均值与实际均值 |
| 排序 | RevenueCapture@业务触达比例 |
| 尾部 | Whale recall、尾部收入贡献和误差 |
| 稳定性 | 跨月/跨渠道结果与区间 |
| 运营 | 收益、成本、覆盖率与在线实验结果 |

### 10.4 不确定性与时间回测

区分单用户未来收入的预测区间、条件均值估计区间、cohort 总收入区间。三者宽度和用途不同。

有日期或 campaign 共同冲击时，可按相应簇/时间块重采样，而非把所有用户当独立样本。极重尾、尾部样本极少或无限方差时，普通 bootstrap 也可能不可靠，应结合多期回测、极端 cohort 压力测试和尾部假设敏感性分析。

应模拟真实部署日期：在日期 $`D`$ 训练时，180 天完整标签只允许来自在 $`D`$ 前已成熟且已入账的用户。仅按注册月份划 train/test、却使用训练当时尚未出现的未来收入，仍然泄漏。

## 11. 工程落地路线

### 11.1 先做标签审计

检查零值来源、异常币种、重复订单、退款、测试账号、机器人、收入回补和单位。报告 top 0.1%/1% 收入占比、cohort 的成熟度以及极端值对总均值的贡献。

真实 whale 与坏数据要区分；删除错误订单是数据修复，删除真实高收入则改变目标分布。

### 11.2 建立可解释基线，再逐步加复杂度

1. **分组均值 + 收缩**：验证标签、切分和总量评估链路。
2. **原尺度 MSE 与 Tweedie**：在相同特征下建立直接均值基线。
3. **log1p 模型**：比较排序表现，单独检查均值反变换。
4. **Hurdle + Gamma**：诊断概率与金额两部分，评估最终乘积。
5. **多时间尺度/动态更新**：利用近期已成熟反馈，做任务消融。
6. **尾部分解或 survival 收入过程**：只有已有误差诊断支持时加入。

表格特征可从梯度提升树或 GLM 开始；丰富行为序列可比较序列模型。模型复杂度不能替代合适 estimand、成熟标签与尾部样本支持。

### 11.3 训练、校准、测试职责分离

- 训练集：拟合模型、编码、阈值与先验。
- 验证集：选模型、超参数及尾部阈值。
- 独立校准数据或交叉拟合：校准概率与最终金额。
- 最终时间外测试：一次性评估拟上线方案。

简单整体金额校准可用：

```math
c_{\mathrm{cal}}=\frac{\sum_{\mathrm{cal}}y_i}
{\sum_{\mathrm{cal}}\hat y_i},\qquad
\hat y_i^{\,\mathrm{cal}}=c_{\mathrm{cal}}\hat y_i.
```

它能修正全局尺度，但不能修复排序、错误的条件结构或渠道差异；小样本分组校准系数也应收缩。不能在测试集估计系数再报告“校准后测试效果”。

### 11.4 上线监控与反馈

监控预测金额分布、付费概率、正值金额、tail head 贡献、极端梯度、缺失率及新类别占比。长期标签尚未成熟时，先检查短期任务，再用延迟到达的长期收入更新评估。

用于投放后，模型会改变未来获客人群，应记录模型版本、出价与曝光政策。预测模型估计的是历史策略下的价值；预算重新分配或补贴收益的判断还需要实验或因果识别。

优化稳定措施应服务于目标：学习率、正则化、组间共享信息通常比任意裁剪收入更容易解释。任何截断、降采样和金额权重，都要写清对 estimand 的影响。

## 12. 常见误区与参考资料

### 12.1 需要记住的边界

| 常见说法 | 更准确的表述 |
|---|---|
| “长尾就不能用 MSE” | MSE 仍是均值损失；问题在风险矩条件、有限样本方差和业务评价 |
| “Gamma 能稳健忽略 whale” | Gamma 仍估计均值，梯度不封顶，常数解仍是样本均值 |
| “log-MSE 预测的是中位数” | 反变换得到几何均值；仅在特定分布下与中位数一致 |
| “Huber 是更稳的均值预测” | 一般目标是 Huber 位置泛函，不是均值 |
| “Hurdle 一定比直接模型好” | 分解可解释，但额外方差与正值样本不足可能抵消收益 |
| “Lifetime × 频次 × 客单价总成立” | 必须明确频次时间单位和条件依赖，避免重复计算生命周期 |
| “留存就是 survival” | 可回流的日活概率与不可逆终止事件的生存函数不同 |
| “加权能补齐未成熟标签” | 无观测支持区域无法靠 IPCW 识别未来收入 |
| “剪掉 top 0.1% 只是清洗” | 若是真实收入，则改变标签与业务目标 |
| “整体预测总额正确就够了” | 还需检查分组校准、排序和跨期稳定性 |

### 12.2 延伸阅读

- [scikit-learn：Gamma deviance](https://scikit-learn.org/1.9/modules/generated/sklearn.metrics.mean_gamma_deviance.html)：支持域与实现约定。
- [scikit-learn：Tweedie deviance](https://scikit-learn.org/dev/modules/generated/sklearn.metrics.mean_tweedie_deviance.html)：power 与分布类型对应。
- [Stan：Hierarchical Partial Pooling for Repeated Binary Trials](https://mc-stan.org/learn-stan/case-studies/pool-binary-trials.html)：pooling 与层次模型。
- [Fader、Hardie、Shang：Customer-Base Analysis in a Discrete-Time Noncontractual Setting](https://www.brucehardie.com/papers/020/)：购买与潜在存活过程。
- [scikit-survival：Inverse Probability of Censoring Weights](https://scikit-survival.readthedocs.io/en/v0.23.1/api/generated/sksurv.nonparametric.ipc_weights.html)：逆删失概率加权接口。

本文核心恒等式、loss 导数和示例为直接数学推导；模型路线为实践建议，应通过业务数据上的时间外验证选择。
