# Decision-Focused Learning 理论笔记：目标、梯度、统计保证与论文导读

> 本文系统讲解一般的 Decision-Focused Learning（DFL），以线性目标的 predict-then-optimize 为主线，再扩展到非线性优化、组合决策与部分反馈。文中标为“推导”的内容是基于明确假设的教学推演；论文贡献与定理条件另行注明。本文是理论入门与代表性文献导读，不是完整文献综述或最新论文排行榜。

业务应用可配合阅读：[端到端学习与 DFL：Coin Treatment Case](端到端决策学习-以金币档位为例.md)。

## 1. DFL 究竟在学习什么？

传统监督学习先预测未知量，再将预测交给决策器；DFL 用**预测所诱导出的决策质量**训练预测模型。

| 层次 | 输入与输出 | 核心问题 |
| --- | --- | --- |
| 预测 | 特征 x → 未知参数的预测 | 参数预测得准不准？ |
| 优化 | 预测参数 → 满足约束的决策 | 在这个预测下应该如何行动？ |
| 决策评价 | 决策 + 实际结果 → 成本或收益 | 这个行动在真实世界表现如何？ |
| DFL 训练 | 决策损失 → 更新预测器 | 应优先修正哪些预测误差？ |

这里的“端到端”是训练目标贯通预测与决策，不要求推理时只有一个网络，也不要求丢弃已知约束和优化器。

[Wilder、Dilkina 与 Tambe（AAAI 2019）](https://arxiv.org/abs/1809.05504)是组合优化 DFL 的代表性早期工作：把预测模型与下游优化联合训练，以决策质量评价学习结果。它不是“任何端到端系统都更好”的保证。

[Mandi 等的综述](https://arxiv.org/abs/2307.13565)梳理了可微优化、平滑、随机扰动与 surrogate 等路线，并提供多任务基准。其价值是帮助理解方法谱系；具体选择仍取决于问题结构、监督信息和求解成本。

## 2. 一般数学框架

### 2.1 决策时已知什么？

设 X 是决策时可见特征，Y 是决策后才揭示的随机量，z 是决策，损失函数是 C。合理的风险中性决策是：

```math
z_B(x)\in\arg\min_{z\in\mathcal Z(x)}
E[C(z,Y)\mid X=x].
```

下标 B 表示 Bayes 最优：只利用决策时已有的信息，而不是事先知道未来。

预测后优化的方法先构造 $`\hat y=f_\theta(x)`$，再求解：

```math
z^*(\hat y;x)\in\arg\min_{z\in\mathcal Z(x)}C(z,\hat y).
```

DFL 的训练风险可以写成：

```math
\mathcal R_{\mathrm{dec}}(\theta)
=E[C(z^*(f_\theta(X);X),Y)].
```

这个写法默认训练能够评价所选决策在 Y 下的损失。部分反馈场景需要额外识别与估计，见第 12 节。

### 2.2 条件均值何时足够？

**推导：** 若不确定参数只出现在一个线性目标中，记随机成本向量为 c，且可行域在决策时已知，则：

```math
C(z,c)=c^\top z,\qquad
\bar c(x)=E[c\mid X=x].
```

```math
E[c^\top z\mid X=x]=\bar c(x)^\top z.
```

所以预测条件均值后优化，就可实现 Bayes 最优。

但非线性决策问题通常不能只代入均值：

```math
E[C(z,Y)\mid X=x]\ne C(z,E[Y\mid X=x]).
```

例如库存的缺货和积压成本不对称时，最优订货量通常是条件分位数；尾部风险目标需要更多分布信息。DFL 的第一步是确定**决策需要什么信息**，而不是默认所有未知量都应预测均值。

## 3. 两种 Regret 必须分开

### 3.1 事后最优解：Hindsight oracle

在线性问题中，定义固定的最优解选择规则：

```math
z^*(c)\in\arg\min_{z\in\mathcal Z}c^\top z.
```

标准 SPO loss 是：

```math
\ell_{\mathrm{SPO}}(\hat c,c)
=c^\top z^*(\hat c)-c^\top z^*(c).
```

这里的基准 $`z^*(c)`$ 已经看到了实际 c，是事后最优解。训练时能计算它，不代表部署时也能达到它。并列最优时须明确 tie-breaking；有些论文使用最坏并列解的定义，不能混用而忽略差异。

### 3.2 决策时最优解：Bayes oracle

条件均值成本下的决策 regret 是：

```math
r_B(x)=
\bar c(x)^\top z^*(\hat c(x))
-\bar c(x)^\top z^*(\bar c(x)).
```

**推导：** 只要期望存在，就有：

```math
E[\ell_{\mathrm{SPO}}(\hat c(X),c)]
=E[r_B(X)]+G,
```

```math
G=
E[\bar c(X)^\top z^*(\bar c(X))]
-E[c^\top z^*(c)]\ge 0.
```

G 不依赖预测器，是“提前知道实际成本”带来的信息优势。

因此最小化平均 SPO loss 与最小化 Bayes regret 有相同的最优预测策略，但平均 SPO loss 未必能降到零。评估时不能把不可消除的随机性都算成模型失败。

### 3.3 一个两 action 的例子

两个 action 的真实条件均值成本为 (1, 2)，应选第一个：

| 预测成本 | 均方误差 | 所选 action | Bayes regret |
| --- | ---: | --- | ---: |
| (10, 20) | 202.5 | 第一个 | 0 |
| (1.6, 1.5) | 0.305 | 第二个 | 1 |

误差小的模型反而作出差决策。这个例子只说明预测风险与决策风险不等价，不说明忽略数值精度永远有利。

## 4. 决策几何：为什么误差有不同价值？

### 4.1 相同决策对应一片预测区域

对有限可行集或多面体的顶点，固定决策 z 的最优区域为：

```math
\mathcal K_z=
\{u:u^\top(z-v)\le 0,\ \forall v\in\mathcal Z\}.
```

只要预测成本留在同一最优区域，决策不变。边界对应两个或多个决策成本相等，跨过边界才可能改变选择。

因此，普通 MSE 关心每个方向的距离；决策损失还关心误差方向是否越过相关边界、越过去会损失多少。

### 4.2 有些参数方向不可由决策识别

**推导：**

- 对成本向量乘同一个正数，不改变线性目标的 argmin。
- 若某方向 a 满足 $`a^\top(z-v)=0`$ 对所有可行 z、v 成立，那么给成本加 a 不改变决策。
- 所以纯 decision loss 未必能恢复真实参数数值；多个预测向量可能产生相同最优策略。

例如只选择一个 action 时，给所有 action 成本加同一常数不改变选择。因而 DFL 输出可以是有用的决策分数，却不一定是校准的概率或货币预测。

### 4.3 预测误差仍然可以控制 regret

**推导：** 设可行域在某范数下直径有界：

```math
D=\sup_{z,v\in\mathcal Z}\|z-v\|.
```

记对偶范数为 $`\|\cdot\|_*`$。由预测解的最优性：

```math
r_B(x)
\le
(\bar c-\hat c)^\top
\left(z^*(\hat c)-z^*(\bar c)\right)
\le D\|\bar c-\hat c\|_*.
```

所以预测准确确实可以保证决策准确；DFL 的动机不是否认这一点，而是指出该界没有充分利用哪些误差实际上不影响决策。

有限 action 下，若每个 action 的成本误差不超过 ε，且最佳与第二佳的真实成本差大于 2ε，则最优 action 不会翻转。接近边界更容易选错，但边界附近两个 action 的互换损失也可能很小。

## 5. 为什么很难直接训练真实 Regret？

对于离散优化，映射 $`z^*(\hat c)`$ 常常分段常数：

- 区域内，预测稍微变化，决策完全不变。
- 边界处，决策可能突然跳变。
- 普通自动微分通常只能得到零梯度或不可导点。

在可微情形，链式法则希望得到：

```math
\nabla_\theta L
=
J_{f_\theta}(x)^\top
J_{z^*}(\hat c)^\top
\nabla_z C(z,Y).
```

困难主要集中在优化解对预测参数的 Jacobian。

**目标值可导不等于最优解可导。** 包络定理可以给出某些最优值函数的导数，但不能自动给出任意下游损失所需的最优解 Jacobian。

DFL 方法通常修改下列对象之一：优化问题、解映射、损失函数，或梯度估计方式。修改后的梯度究竟对应哪个目标，是阅读论文时最应该追问的问题。

## 6. 路线一：对连续优化层做隐式微分

[Amos 与 Kolter 的 OptNet（ICML 2017）](https://proceedings.mlr.press/v70/amos17a.html)把二次规划嵌入网络，并利用最优性条件求导。[Agrawal 等（NeurIPS 2019）](https://arxiv.org/abs/1910.12430)进一步给出 disciplined parametrized programming 下的通用凸优化层框架。这些是 DFL 的重要技术基础，但“用了可微优化层”不自动等于“在优化真实决策 regret”。

### 6.1 最简单的推导

考虑无约束、光滑优化：

```math
z^*(u)=\arg\min_z F(z,u).
```

若最优解附近 Hessian 可逆，最优性条件为：

```math
\nabla_z F(z^*(u),u)=0.
```

对 u 求导得到：

```math
\frac{\partial z^*}{\partial u}
=
-\left(\nabla^2_{zz}F\right)^{-1}
\nabla^2_{zu}F.
```

实际计算通常解线性系统，而非显式求矩阵逆。

### 6.2 带约束的问题

将原始变量和对偶变量合记为 s，用 KKT 条件表示：

```math
H(s,u)=0,\qquad
\frac{\partial s}{\partial u}
=
-\left(\frac{\partial H}{\partial s}\right)^{-1}
\frac{\partial H}{\partial u}.
```

这个教学公式需要局部可微性、适当约束资格条件和非奇异性。多解、退化约束、活跃集切换都可能带来困难。凸性本身不保证处处有稳定梯度。

### 6.3 技术取舍

- **隐式微分**：对最优性条件求导，依赖求解精度与正则性。
- **展开迭代**：把有限步求解算法作为计算图，所得梯度对应这段有限步算法，可能有内存和截断误差。
- **强凸正则化**：可改善唯一性和稳定性，但改变了原问题。

训练软化后的问题，最终用原问题求解时，要检查二者差距。

## 7. 路线二：正则化与 Softmax 松弛

**推导：** 对 K 个 action 的预测成本 d，允许选择一个概率分布：

```math
\Delta_K=\{\pi:\pi_k\ge0,\ \sum_{k=1}^K\pi_k=1\}.
```

引入负熵正则：

```math
\pi_\tau(d)=
\arg\min_{\pi\in\Delta_K}
\left\{d^\top\pi+\tau\sum_{k=1}^K\pi_k\log\pi_k\right\}.
```

由一阶条件得到：

```math
\pi_{\tau,k}(d)
=\frac{\exp(-d_k/\tau)}
{\sum_j\exp(-d_j/\tau)},\qquad \tau>0.
```

于是可以训练可微的软决策成本 $`c^\top\pi_\tau(\hat c)`$。

### 7.1 温度为何重要？

```math
\frac{\partial\pi_{\tau,i}}{\partial d_j}
=-\frac1\tau\pi_{\tau,i}
\left(\mathbf1\{i=j\}-\pi_{\tau,j}\right).
```

温度降低时决策更尖锐，但非边界区域的概率可能饱和，梯度接近零；边界附近又可能出现较大的局部敏感度。不能只依据公式中的 1/τ 判断所有梯度都会变大。

### 7.2 松弛误差的一个简单界

对同一个成本向量 d，由正则化最优性及熵上界可得：

```math
0\le d^\top\pi_\tau(d)-\min_k d_k\le\tau\log K.
```

这个界比较的是同一个 d 下的软解与硬解，**不是在预测错误时相对于真实成本 c 的完整 regret 保证**。

对于复杂组合约束，简单 softmax 无法自动保证预算、匹配或调度可行性；连续松弛后的解也可能需要舍入，舍入必须纳入评估。

## 8. 路线三：SPO+，不对离散 Argmin 直接求导

[Elmachtoub 与 Grigas，Smart “Predict, then Optimize”](https://arxiv.org/abs/1710.08005)提出利用决策结构构造凸 surrogate。以下记可行域非空紧致，目标线性，并固定最优解选择规则。

### 8.1 损失与次梯度

SPO+ 定义为：

```math
\ell_{\mathrm{SPO+}}(\hat c,c)
=
\max_{z\in\mathcal Z}(c-2\hat c)^\top z
+2\hat c^\top z^*(c)-c^\top z^*(c).
```

它对预测向量 $`\hat c`$ 凸，并上界相应的 SPO loss。一个可用次梯度为：

```math
g_{\mathrm{SPO+}}
=2\left(z^*(c)-z^*(2\hat c-c)\right).
```

训练调用成本为 c 和 $`2\hat c-c`$ 的优化 oracle；真实成本最优解通常可预先计算。这个公式利用两次决策的差异提供方向，不需要把整数求解器本身变成可微函数。

### 8.2 如何理解它？

真实损失在当前决策区域内可能完全平坦；surrogate 使用一个由真实成本与预测成本共同构造的优化问题，比较其解与真实成本下的解，以此惩罚危险预测方向。

三个限制：

- 对预测向量凸，不意味着与神经网络复合后对参数 θ 凸。
- 有优化 oracle，不意味着原组合问题变得容易；训练仍可能昂贵。
- 可计算、上界和统计一致性是不同性质，不能由其中一个推出其余两个。

### 8.3 一致性不能脱离假设

原论文的一个 Fisher consistency 定理使用条件成本分布的中心对称性、连续性、可行域非空内部，以及条件均值最优解唯一等假设，见论文 Assumption 1 与 Theorem 1。任意偏斜分布、退化可行域或受限预测模型不自动适用。具体结论须按论文的损失定义与正则条件理解。

## 9. 路线四：黑盒求解器与随机扰动

### 9.1 黑盒反向传播

[Vlastelica 等，Differentiation of Blackbox Combinatorial Solvers（ICLR 2020）](https://arxiv.org/abs/1912.02175)保留离散求解器，利用下游梯度构造一个扰动后的优化调用，得到有用的反向信号。

在最小化约定下，一种常见表达是：

```math
z=z^*(\hat c),\qquad
g=\nabla_z L,\qquad
z_\lambda=z^*(\hat c+\lambda g),
```

```math
\widetilde{\nabla}_{\hat c}L
=\frac{z_\lambda-z}{\lambda},\qquad \lambda>0.
```

它对应论文构造的插值 / 近似反向机制，不是原始分段常数映射的普通精确导数。λ 太小可能两次求解相同、信号为零；太大则偏离局部任务。要核对实现的最小化 / 最大化符号约定。

### 9.2 对成本加噪声，使平均解平滑

[Berthet 等，Learning with Differentiable Perturbed Optimizers（NeurIPS 2020）](https://arxiv.org/abs/2002.08676)研究随机扰动优化器及其学习损失。基本思路是对参数加噪声后求解，再平均。

以下为标准高斯噪声下的教学表达，假设解有界并满足交换积分和求导的条件：

```math
z_\sigma(u)=E_Z[z^*(u+\sigma Z)],
\qquad Z\sim N(0,I).
```

```math
J_{z_\sigma}(u)
=\frac1\sigma E_Z[z^*(u+\sigma Z)Z^\top].
```

因此可用多次求解估计平滑映射及 Jacobian。噪声越大，平滑偏差可能越大；噪声很小时估计方差和数值问题可能上升。平均离散解未必是可部署的离散解，仍需区分训练对象与最终决策。

这与上一小节不同：一个沿下游梯度构造扰动，一个对随机扰动定义期望平滑，不能把它们当作同一个算法。

## 10. 路线五：排序与学习型 Surrogate

### 10.1 从参数拟合转向候选决策排序

[Mandi 等，Decision-Focused Learning: Through the Lens of Learning to Rank（ICML 2022）](https://proceedings.mlr.press/v162/mandi22a.html)从候选解排序角度构造 DFL 损失。

一个教学示例：若真实成本表明 $`c^\top z_a<c^\top z_b`$，希望预测也把 a 排在前面，可使用：

```math
\ell_{ab}(\hat c)
=\log\left(1+\exp\left(\hat c^\top(z_a-z_b)/t\right)\right),
\qquad t>0.
```

这只是示意性的 pairwise logistic loss，不是该论文全部方法的完整复现。它依赖候选集；若候选解没有包含关键竞争者，训练可能在错误的局部比较上取得很好表现。

### 10.2 把复杂 regret 近似为可学习损失

[Shah 等，Decision-Focused Learning without Differentiable Optimization: Learning Locally Optimized Decision Losses（NeurIPS 2022）](https://arxiv.org/abs/2203.16067)提出 LODL：通过在标签附近采样预测、调用真实优化器测量决策损失，学习局部代理损失，再用于训练预测器。

这减少了“每次反向传播都穿过求解器”的需求，但没有免除获取决策损失样本的成本。损失代理必须覆盖后续预测会到达的区域；局部拟合准确不代表任意分布外预测都可靠。

## 11. 统计理论：一致性、校准与泛化

### 11.1 Fisher consistency 是总体性质

给定 x，若 surrogate 的条件风险最小化解也最小化目标决策风险，称为相应的 Fisher consistency。

它讨论理想总体分布，不自动保证：

- 有限样本上可以准确估计；
- 受限模型类能表示理论最优预测；
- 优化算法找到全局最优解；
- 实际部署分布与训练一致。

而且 DFL 一致并不要求恢复唯一真实成本向量；只要诱导出的决策达到目标风险最优即可。SPO+ 某些更强条件下可识别条件均值，这是额外结论。

### 11.2 Surrogate calibration 不是概率校准

[Liu 与 Grigas，Risk Bounds and Calibration for a Smart Predict-then-Optimize Method（NeurIPS 2021）](https://arxiv.org/abs/2108.08887)研究的是 surrogate 超额风险如何控制 SPO 超额风险。

教学性地记：

```math
\delta\left(\mathcal R_{\mathrm{dec}}(f)
-\mathcal R_{\mathrm{dec}}^*\right)
\le
\mathcal R_{\mathrm{sur}}(f)
-\mathcal R_{\mathrm{sur}}^*.
```

若校准函数满足适当的正性和可逆性，较小 surrogate 超额风险才能转化为较小决策超额风险。这里不是检查“预测 20% 是否真的发生 20%”。

该论文在指定分布条件下分析多面体可行域与强凸函数水平集；多面体情形的某些结果有二次量级校准界。界依赖几何和分布假设，不存在适用于所有 DFL、所有数据的一条无条件速率。

### 11.3 误差来源应逐层拆开

**分析框架：**

```math
\mathcal R_{\mathrm{sur}}(\hat f)-\mathcal R_{\mathrm{sur}}^*
=
\left[\mathcal R_{\mathrm{sur}}(\hat f)
-\inf_{f\in\mathcal F}\mathcal R_{\mathrm{sur}}(f)\right]
+
\left[\inf_{f\in\mathcal F}\mathcal R_{\mathrm{sur}}(f)
-\mathcal R_{\mathrm{sur}}^*\right].
```

第二项是模型类的近似误差；第一项还要分析有限样本估计和训练优化误差。之后才通过适用的校准关系转成决策风险。

此外，还有求解器近似、软硬决策差异、收益测量误差和分布漂移。这些不一定能简单相加成一个普适界，但都应该单独诊断。

### 11.4 为什么两阶段方法仍可能最好？

线性目标下，正确估计条件均值已经足够。平方损失在适当矩条件下识别该均值；分类概率也可用适当评分规则学习。

DFL 的主要机会是：模型受限或样本有限时，预测误差对收益的重要性不均匀。它可能降低近似偏差，也可能增加训练方差、计算成本或 surrogate 偏差。不存在仅凭“端到端”三个字就必然优于两阶段的理论。

## 12. Full Information 与部分反馈：一个关键边界

经典 DFL 经常假设训练时观察完整成本向量，因而可评价多个候选决策；金币 treatment 只有实际 action 的结果，两者监督条件不同。

| 数据条件 | 训练时可以获得什么 | 主要难点 |
| --- | --- | --- |
| 完整成本向量 | 任意可行决策的实际线性成本 | 不可导、求解复杂度、统计泛化 |
| 模拟器 | 给定 action 的模拟结果 | 模拟误差与真实环境差异 |
| 随机试验 / 日志反馈 | 一个 action 的结果与 propensity | 反事实缺失、覆盖、估计方差 |
| 序列环境 | action 改变未来状态 | 长期信用分配、探索和动态建模 |

在部分反馈下，不能把预测成本矩阵当作真值直接套 SPO+，并宣称获得了真实反事实监督。

对固定随机策略 π，记录实际 action A、收益 R 和 propensity e，在一致性、可忽略性和覆盖条件下：

```math
J(\pi)
=E\left[\frac{\pi(A\mid X)}{e(A\mid X)}R\right].
```

该恒等式解释了为什么 policy learning 还需要 IPS、DR 或其他识别方法。它不提供“个体所有 action 的真值”，也不让监督数据变成 full information。

DFL 通常保留“预测参数 → 结构化优化器”；policy learning 可直接输出策略。RL 则进一步处理行动对未来状态和长期收益的影响。它们可以结合，但不是同义词。

## 13. 接回 Coin Treatment：理论如何对应业务？

为避免与成本向量 c 混淆，本节用 a 表示金币金额。设收益单位已统一，且点击后的平均收益 m 不随 action 改变：

```math
V_a(x)=p_a(x)(m(x)-a).
```

将收益取负可写成理论中的最小化形式：

```math
d_a(x)=-V_a(x),\qquad
\mathcal Z=\{e_1,\ldots,e_K\}.
```

这里 e_k 是第 k 个单位向量。选 coin 相当于选一个 action 顶点。

| 一般 DFL 概念 | Coin Case |
| --- | --- |
| 未知参数 | 各档 CTR、收入参数或负价值 |
| 预测器 | CTR 模型与收入估计 |
| 可行域 | 允许的金币档位，或带预算的联合策略 |
| 决策器 | 利润 argmax |
| 决策边界 | 两档预期利润相等 |
| 参数误差的重要方向 | 能改变档位选择且造成收益损失的误差 |
| 信息限制 | 每人只观察一个 treatment 的结果 |
| 实用 surrogate | 加权 BCE、BCE + 合适的 policy value loss |

BCE + 利润 argmax 已是合理的两阶段基线；高价值加权属于 value-aware prediction，但未必利用 action 竞争。真正的 decision loss 需可靠的价值监督或反事实估计。

若 CTR 要用于监控、解释和新策略模拟，可保留预测辅助目标：

```math
L(\theta)=L_{\mathrm{pred}}(\theta)
+\lambda L_{\mathrm{dec}}(\theta).
```

辅助项可以约束参数语义，但不会自动保证完全校准。新 coin 缺少数据时，保留 CTR 中间层也不能自动解决外推问题。

## 14. 如何选方法与验证结果？

### 14.1 按问题结构选路线

| 情况 | 优先考虑 | 核心检查 |
| --- | --- | --- |
| 预测已准确，决策对误差不敏感 | 两阶段基线 | 增加复杂度是否有实测收益 |
| 少量离散 action | softmax surrogate、直接枚举 | soft/hard 差异、监督是否可靠 |
| 光滑且正则的连续优化 | 隐式微分 / 可微凸优化层 | 唯一性、KKT 稳定性 |
| 线性目标、有完整成本和 oracle | SPO+ | 分布假设、求解次数 |
| 现有离散黑盒 solver | 黑盒反向方法 | 扰动尺度、近似信号 |
| 可并行调用 solver | 随机扰动平滑 | Monte Carlo 方差与计算预算 |
| 候选决策缓存较好 | 排序 surrogate | 候选覆盖与更新 |
| 精确损失昂贵但可离线采样 | 学习型局部 surrogate | 局部覆盖、代理泛化 |

这是一份起点清单，不是按优劣排序。

### 14.2 公平的实验至少要报告什么？

1. **同一个部署决策器**：训练方法可以不同，最终预算、约束和评估求解精度应可比。
2. **预测与决策指标并列**：预测损失、实际成本 / 收益、相对基线增益、可行率。
3. **时间与计算预算**：训练时长、solver 调用数、推理延迟；不能只比 epoch。
4. **监督公平**：各方法使用相同信息；完整成本标签与单 action 日志不能直接混比。
5. **独立评估**：调参集与最终测试集分开；策略学习尤其要避免同一日志上训练、选优和报收益。
6. **不确定性**：随机种子、时间切片、置信区间、长尾稳定性。
7. **尺度**：归一化 regret 分母接近零或成本可为负时容易误导，同时报告绝对差。

求解器超时或只有近似解时，事后“最优值”也可能只是近似基准；应报告最优性 gap 或界，不能把近似值当成精确 oracle。

## 15. 阅读论文时的八个问题

1. 决策时可见信息是什么？训练时额外看到了什么？
2. 预测的是实际随机实现、条件均值，还是一个用于决策的分数？
3. 优化器是线性、凸、非凸还是整数问题？可行域是否受未知量影响？
4. regret 的比较基准是 hindsight oracle 还是 Bayes oracle？
5. 梯度对应原问题、正则化问题、插值损失还是学习到的 surrogate？
6. 保证是可微性、上界、一致性还是有限样本风险界？假设是什么？
7. 最终部署是否还使用训练时的软化或随机化策略？
8. 收益来自决策目标、额外标签、更多计算，还是更好的模型与调参？

尤其不要混淆：**可微 ≠ 一致；一致 ≠ 有限样本更好；训练收益上升 ≠ 真实部署收益上升。**

## 16. 论文导读与推荐顺序

下表年份以所列会议 / 正式版本为主；arXiv 首次提交年份可能更早。

| 顺序 | 论文 | 阅读重点 |
| --- | --- | --- |
| 1 | Mandi et al., 2024, *Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities* | 先建立方法地图，区分优化层与损失 surrogate |
| 2 | Wilder, Dilkina & Tambe, AAAI 2019, *Melding the Data-Decisions Pipeline* | 理解预测与组合优化联合训练的动机 |
| 3 | Elmachtoub & Grigas, Management Science 2022, *Smart “Predict, then Optimize”* | SPO、SPO+、oracle 次梯度与一致性条件 |
| 4 | Amos & Kolter, ICML 2017, *OptNet* | 用最优性条件对二次规划层求导 |
| 5 | Agrawal et al., NeurIPS 2019, *Differentiable Convex Optimization Layers* | 通用参数化凸优化层 |
| 6 | Vlastelica et al., ICLR 2020, *Differentiation of Blackbox Combinatorial Solvers* | 黑盒离散求解器的反向机制 |
| 7 | Berthet et al., NeurIPS 2020, *Learning with Differentiable Perturbed Optimizers* | 随机扰动、平滑与梯度估计 |
| 8 | Liu & Grigas, NeurIPS 2021, *Risk Bounds and Calibration for a Smart Predict-then-Optimize Method* | 超额风险转换与几何 / 分布条件 |
| 9 | Mandi et al., ICML 2022, *Decision-Focused Learning: Through the Lens of Learning to Rank* | 候选解的排序视角 |
| 10 | Shah et al., NeurIPS 2022, *Decision-Focused Learning without Differentiable Optimization: Learning Locally Optimized Decision Losses* | 用局部学习损失替代直接可微求解 |

### 原文链接

1. [DFL 综述](https://arxiv.org/abs/2307.13565)
2. [Melding the Data-Decisions Pipeline](https://arxiv.org/abs/1809.05504)
3. [Smart “Predict, then Optimize”](https://arxiv.org/abs/1710.08005)
4. [OptNet](https://proceedings.mlr.press/v70/amos17a.html)
5. [Differentiable Convex Optimization Layers](https://arxiv.org/abs/1910.12430)
6. [Differentiation of Blackbox Combinatorial Solvers](https://arxiv.org/abs/1912.02175)
7. [Learning with Differentiable Perturbed Optimizers](https://arxiv.org/abs/2002.08676)
8. [Risk Bounds and Calibration](https://arxiv.org/abs/2108.08887)
9. [Through the Lens of Learning to Rank](https://proceedings.mlr.press/v162/mandi22a.html)
10. [Learning Locally Optimized Decision Losses](https://arxiv.org/abs/2203.16067)

## 17. 记住这条主线

DFL 把“参数应该预测得多准确”改写为“什么预测足以支持好的决策”。已知目标函数和可行域提供结构，regret 规定误差的经济后果，surrogate 或可微机制提供训练信号，统计假设决定这些信号能否泛化。

是否采用 DFL，最终取决于它能否在相同信息、约束与计算预算下，稳定改善独立数据上的决策质量。参数预测已经足够、校准能解决主要偏差，或反事实监督仍不足时，简单方法完全可能更合适。
