# 端到端学习与 Decision-Focused Learning：以 Coin Treatment 为例

> 基于「讲解端到端学习」对话整理。重点是如何从不同金币档位的 CTR 预测走到净利润决策，以及何时值得引入 DFL。示例数字用于说明原理，不是业务实测结果。本文补充了单位、因果识别和估计误差方面的适用条件。

## 1. 业务问题与符号

给用户展示不同金币档位，金币影响点击概率；点击后产生收入与金币成本。目标是为每个用户选择期望净利润最高的档位。

| 符号 | 含义 |
| --- | --- |
| $`X`$ | 决策前可获得的用户及场景特征 |
| $`\mathcal C=\{c_1,\ldots,c_K\}`$ | 候选金币成本，例如 1、3、5、10 |
| $`A`$ | 日志中实际分配的金币金额，属于候选集合 |
| $`Y(c)`$ | 分配金币 c 时的潜在点击结果，取 0 或 1 |
| $`p_c(x)`$ | 用户在金币 c 下的真实点击概率 |
| $`m(x),\hat m(x)`$ | 单次点击收益的条件均值及其历史数据估计 |
| $`R(c),V_c(x)`$ | 实现净利润、条件期望净利润 |
| $`e(c\mid x)`$ | 历史策略分配金币 c 的概率（propensity） |

本文把用于估计 m 的决策前信息也包含在 X 中。采用共同收益 m 的简化模型时：

```math
p_c(x)=P(Y(c)=1\mid X=x),\qquad
V_c(x)=p_c(x)(m(x)-c).
```

线上先计算各档位的预测价值，再选择最大者：

```math
\hat V_c(x)=\hat p_c(x)(\hat m(x)-c),\qquad
\hat c(x)=\arg\max_{c\in\mathcal C}\hat V_c(x).
```

### 1.1 先统一 CPM、coin 与收益口径

原对话使用业务简写 **profit = CTR × (CPM − coin)**。严格计算时，括号里的收入和金币成本必须是**同一事件、同一货币、同一计量单位**。

- 标准 CPM 表示每千次展示的金额，不能直接当作每次点击收益。若一次点击触发一次可变现展示，其期望收益可以是 CPM/1000；若触发多个展示，还需建模数量和实际结算。
- 本文用 m 表示已经换算到「每次点击发生时」的期望收益，用 c 表示相应金币的货币成本。
- 若金币只在点击后支付，净利润结构是 $`Y(c)(m-c)`$。
- 若金币无论点击与否都支付，则应改为 $`V_c=p_cm-c`$，后续权重、决策边界和方差公式也要随之修改。

如果实际点击收益也是随机变量 B，更一般的正确分解是：

```math
V_c(x)=E[Y(c)(B(c)-c)\mid X=x]
=p_c(x)\left(E[B(c)\mid X=x,Y(c)=1]-c\right).
```

所以共同 m 的假设是：**给定 X，点击后的条件平均收益在各 coin 下相同**。金币可能改变点击人群、后续行为或广告收益；此时需要 action-specific 的 $`m_c(x)`$。不能把一般的 $`E[YB\mid X]`$ 无条件拆成 $`E[Y\mid X]E[B\mid X]`$。

若模型使用历史估计的 m 构造日志奖励，得到的是代理奖励；最终评估应尽可能使用真实结算收入和成本，否则 DFL 也只是在优化这个代理量。

### 1.2 这是多 treatment 决策问题

一个用户只接受一个 coin，日志通常只有：

```math
(X_i,A_i,Y_i,e(A_i\mid X_i)).
```

其余档位的结果是反事实，不能把同一个点击标签复制给全部 coin。

RCT 提供随机分配；观察性数据还需要一致性、给定 X 的无未观测混杂和覆盖性（候选 action 的分配概率为正），才能把 $`P(Y=1\mid X,A=c)`$ 解释为干预后的 $`p_c(X)`$。历史策略依赖 X 本身并不必然造成不可识别，关键是混杂是否被记录、候选档位是否有支持。DFL 不会自动解决这些问题。

## 2. BCE + 利润 Argmax 为什么是强基线

当前流程属于 Predict-then-Optimize：

1. 用点击标签训练 CTR 模型。
2. 对全部候选 coin 预测 CTR。
3. 按已知收益结构计算价值。
4. 选择预测净利润最大的 coin。

单条样本的 binary cross entropy（BCE）是：

```math
\ell_{\mathrm{BCE}}(y,q)=-y\log q-(1-y)\log(1-q).
```

固定用户和 action，真实点击概率为 p，则条件风险是：

```math
L(q)=-p\log q-(1-p)\log(1-q),\qquad
\arg\min_q L(q)=p.
```

因此 BCE 是严格适当评分规则。若因果识别、action 覆盖、模型容量和数据充分，m 也正确，则预测 CTR 收敛到真概率，利润决策可以达到最优。最优 action 并列时，不要求收敛到唯一标签，但可以达到相同价值。

**最终目标是利润，并不意味着 BCE 是错误的目标。** DFL 主要试图改善有限样本、有限模型容量下的资源分配：优先修正真正影响决策收益的预测误差。

另外，CTR 模型可以把 coin 和 m 作为输入；“普通 BCE 不感知业务价值”指的是损失没有显式按照利润后果计价，并不是模型完全看不到这些特征。

## 3. 预测误差、利润误差与决策 Regret

### 3.1 一个贯穿全文的例子

设 m = 10，候选档位为 1、3、5：

| coin | 真实 CTR | 真实期望净利润 |
| --- | ---: | ---: |
| 1 | 0.20 | 1.80 |
| 3 | 0.24 | 1.68 |
| 5 | 0.35 | 1.75 |

最优 coin 是 1。如果模型把 coin=1 的 CTR 预测为 0.19，把 coin=5 的 CTR 预测为 0.36，预测价值分别变成 1.71 和 1.80，策略会选错，损失 0.05。

反过来，预测 0.15 和 0.25 虽然概率误差更大，但对应价值 1.35 和 1.25，仍然选择正确的 coin=1。

因此要区分：**CTR 误差、价值误差、最终决策损失**。

### 3.2 利润误差的放大与 m 的误差

m 已知时，有精确关系：

```math
\hat V_c-V_c=(m-c)(\hat p_c-p_c).
```

定义 $`\delta p_c=\hat p_c-p_c`$ 和 $`\delta m=\hat m-m`$，则：

```math
\hat V_c-V_c
=(m-c)\delta p_c+p_c\delta m+\delta p_c\delta m.
```

这说明只优化 CTR 无法修复全部收益误差。即使 m 对所有 action 相同，它的误差也会通过不同的 CTR 改变档位间的价值差。

### 3.3 Regret 与决策间隔

定义最优档位 $`c^*(x)`$、实际所选档位 $`\hat c(x)`$，单用户条件期望 regret 为：

```math
r(x)=V_{c^*(x)}(x)-V_{\hat c(x)}(x)\ge 0.
```

若最大价值预测误差是：

```math
\epsilon(x)=\max_{c\in\mathcal C}|\hat V_c(x)-V_c(x)|,
```

则由预测 argmax 的定义可得：

```math
r(x)\le
|V_{c^*}-\hat V_{c^*}|+
|\hat V_{\hat c}-V_{\hat c}|
\le 2\epsilon(x).
```

设最佳与第二佳 action 的真实价值差为 $`\Delta(x)`$。当最优 action 唯一，且 $`\Delta(x)>2\epsilon(x)`$，决策不会选错。

需要同时关注**选错概率和选错后的价值损失**。最佳与第二佳极接近时，二者互换损失很小；但若误选第三个差很多的 action，损失仍可很大，不能把小 top-two margin 直接等同于小 regret。

对于 Bernoulli，自然对数 BCE 的超额条件风险是 KL 散度，Pinsker 不等式给出：

```math
|p_c-\hat p_c|
\le \sqrt{\frac{D_{\mathrm{KL}}(\mathrm{Bern}(p_c)\Vert\mathrm{Bern}(\hat p_c))}{2}}.
```

这解释了 BCE 与决策质量并非无关。不过总体平均 BCE 下降，不保证每个 action、每个高价值分群的 regret 都下降；长尾收益下还需要相应的矩条件和覆盖性。

### 3.4 Coin 决策边界

两个 action 的边界是：

```math
p_{c_1}(m-c_1)=p_{c_2}(m-c_2).
```

只有在 $`m>c_2`$ 且 $`p_{c_1}>0`$ 时，选择较高 coin 的条件才可安全写成：

```math
\frac{p_{c_2}}{p_{c_1}}>\frac{m-c_1}{m-c_2}.
```

若分母为零或负数，应直接比较原始价值，避免错误除法或不等号方向。若业务允许“不投放”，应把其真实价值和成本明确作为候选 action；零金币不自动等于零价值的不投放。

## 4. Calibration：不仅看全局 CTR

### 4.1 排序、校准与策略质量是三件事

概率校准的含义是：

```math
P(Y=1\mid \hat p=q)=q.
```

预测 20% 的一批样本，实际点击率应接近 20%。AUC 主要关心排序，不能证明概率数值可靠；即使概率校准良好，也不能单独证明用户级 coin 选择最优。

### 4.2 为什么必须看 coin × m

全局高估和低估可能相互抵消。更有意义的检查是按 coin、决策时可见的 m 估计以及预测概率分桶：

```math
E[Y\mid A=c,\hat p\in B_p,\hat m\in B_m]
\approx
E[\hat p\mid A=c,\hat p\in B_p,\hat m\in B_m].
```

建议至少检查：

- 全局 reliability diagram、BCE、Brier score。
- 各 coin 的概率校准。
- coin × m 分位桶，单列 P90–P99、P99–P99.9、P99.9+ 等尾部。
- 每个桶的样本量、点击数、置信区间及跨时间稳定性。
- m 自身的预测值与真实点击后收入是否对齐。

分桶用线上可获得的历史估计，不能用未来真实收益分桶后宣称线上可实现。观察性日志的 action 人群可能不同，跨 action 价值比较需在共同目标人群上用随机化、标准化或 IPS/DR 等方法评估。

ECE 可写为：

```math
\mathrm{ECE}
=\sum_b\frac{n_b}{n}|\bar y_b-\bar p_b|.
```

ECE 依赖分桶；Brier 和 BCE 同时反映概率预测的其他性质，不是纯校准指标。不能只报一个 ECE。

### 4.3 何时校准误差不影响决策

如果对同一用户，全部 action 的 CTR 都乘以相同正数 a，m 正确，且没有裁剪或其他变换，则：

```math
\hat V_c=aV_c,\qquad
\arg\max_c\hat V_c=\arg\max_c V_c.
```

因此概率不准不一定导致策略错误。但不同 coin 或不同 m 区间的偏差通常更危险。一般单调校准虽可保留 CTR 排序，也不保证保留乘上不同利润系数后的价值排序。

### 4.4 如何做校准

在独立 calibration set 或 out-of-fold 预测上拟合校准器，再在未使用的测试集评估；业务中优先按时间划分，并避免用户泄漏。

| 方法 | 适合情况 | 注意事项 |
| --- | --- | --- |
| Logistic / Platt | 平滑的概率尺度和偏移误差 | 对概率先做安全裁剪再取 logit |
| Temperature scaling | 主要是置信度尺度问题 | 单参数不能修复任意偏移和分群偏差 |
| Isotonic regression | 有足够数据，校准关系单调但非线性 | 稀疏尾部易过拟合，阶梯输出可能造成并列 |
| 条件校准 | 偏差随 coin、m 改变 | 用共享参数、正则化或层级收缩控制方差 |

例如 logistic calibration：

```math
p_{\mathrm{cal}}
=\sigma(a\,\mathrm{logit}(\hat p)+b).
```

条件校准可把 coin、历史 m 和交互项加入校准器，但切得越细，估计越不稳定。BCE 鼓励概率真实性，却不保证有限样本下自动校准；混合 decision loss 后也应重新检查。

### 4.5 Value calibration 比 CTR calibration 更接近业务

按 coin × m 检查预测利润与实测或 DR 估计利润，是很好的诊断；但桶内误差可抵消，不能证明个体最优策略已学好。

进一步检查价值差：

```math
\Delta V_{cd}(x)=V_c(x)-V_d(x),\qquad
\widehat{\Delta V}_{cd}(x)=\hat V_c(x)-\hat V_d(x).
```

用独立 RCT 或合适的反事实估计检查预测价值差分桶的均值、符号、策略分歧区域的收益。真实个体反事实差通常不可直接观测，不能宣称通过逐用户标签验证了全部 action。

## 5. 高 m 长尾为什么特别难

高 m 用户可能贡献大量收入，同时会放大 CTR 误差和利润标签波动。收入贡献大也不意味着策略改善空间大：如果最优 coin 已很明显，继续提高该用户 CTR 精度可能不改变收益。

在 m 给定、点击后收益视作确定的简化模型中：

```math
R=Y(m-c),\qquad
E[R\mid X,A=c]=p_c(m-c),
```

```math
\mathrm{Var}(R\mid X,A=c)=p_c(1-p_c)(m-c)^2.
```

例如 p = 0.05、m−c = 100，平均利润为 5，但标签有 95% 为 0、5% 为 100，方差为 475。利润回归的标签既稀疏又可能极端。

若点击后的收益 B 也随机，令其条件均值为 m、条件方差为 s²，则：

```math
\mathrm{Var}(Y(B-c)\mid X,A=c)
=p_c s^2+p_c(1-p_c)(m-c)^2.
```

需要同时处理点击随机性和收入随机性。全局方差可能在重尾下不存在；涉及方差的结论默认相关二阶矩有限。

## 6. 直接回归利润与 Rao–Blackwellization

### 6.1 直接回归利润在目标上没有错

用平方损失回归 R，其总体最优预测是条件均值：

```math
f^*(X,A)=E[R\mid X,A].
```

困难是有限样本下的长尾标签、容量分配和优化稳定性，而不是“利润不能作为监督目标”。这个总体最优性质也不等于任意具体训练算法自动具有一致性。

已知收益结构时，可让模型只学未知 CTR：

```math
\hat V_c(x)=\hat p_c(x)(m(x)-c).
```

如果对这种结构化价值预测使用利润 MSE，有：

```math
\left(Y(m-c)-\hat p(m-c)\right)^2
=(m-c)^2(Y-\hat p)^2.
```

它等价于利润系数平方加权的 Brier loss，可能非常强调尾部；并不是简单换一个标签就天然更稳。

### 6.2 条件期望降低哪部分方差

设 Z 二阶可积，T 是条件信息。全期望和全方差公式给出：

```math
E[E[Z\mid T]]=E[Z],
```

```math
\mathrm{Var}(Z)
=\mathrm{Var}(E[Z\mid T])
+E[\mathrm{Var}(Z\mid T)].
```

因此用真实条件期望替代随机实现值，保留均值并去掉条件内随机性。在本例中：

```math
R=p_c(m-c)+(Y-p_c)(m-c).
```

第二项条件均值为零。若真实 p 已知，条件期望可积分掉点击噪声。

**关键限定：p 实际需要估计。** 用 $`\hat p`$ 替代 Y 会引入模型误差和可能的偏差；CTR 模型本身仍用随机点击标签训练。结构分解不保证任何数据集上都比利润回归方差更低或 MSE 更小，也不能保证学习过程无噪声。

经典 Rao–Blackwell 定理通常对估计量按充分统计量条件化；这里更准确的说法是借鉴 **conditional-expectation variance reduction** 思想，而非直接获得经典定理对学习算法的保证。

### 6.3 是“先验”还是“缩小模型空间”？

把价值模型限制为：

```math
\mathcal F=
\{f(x,c)=p(x,c)(m(x)-c):p\in\mathcal P\}
```

是在引入**归纳偏置或结构约束**：已知的乘法和成本关系不用模型重新学习。它不必是贝叶斯概率先验，也与“对真实随机量取条件期望”不是同一件事。

结构正确时，可能改善样本效率；结构错误时，会带来系统偏差。例如 coin 改变点击后收入、存在长期留存收益或支付机制不同，就需要扩展利润结构。

## 7. 样本加权：让有限能力关注重要区域

### 7.1 三种不同的加权目的

| 目的 | 典型权重 | 解决的问题 |
| --- | --- | --- |
| 高价值用户优先 | g(m) | 哪些用户更值得提高预测精度 |
| 利润敏感度 | g(绝对值 m−c) | 哪个 action 的 CTR 误差对价值影响更大 |
| 分布或 action 修正 | 目标概率 / 日志概率 | 把训练或评估目标迁移到指定分布 |

利润敏感度来自：

```math
\left|\frac{\partial V_c}{\partial p_c}\right|=|m-c|.
```

但它只是有动机的权重，**不是唯一最优权重，也不等于 regret loss**。BCE 的局部曲率还取决于 p，真正的决策损失还依赖各 action 的竞争关系。

加权 BCE：

```math
L_{\mathrm{WBCE}}
=\frac{\sum_i w_i\ell_{\mathrm{BCE}}(Y_i,\hat p(X_i,A_i))}
{\sum_i w_i}.
```

若用实际 margin 做权重，须非负；m−c 为负也不能把 BCE 乘负权重。用绝对值、平滑函数和正的权重下限，可避免该区域完全失去监督。

### 7.2 c 是 RCT 分配的，究竟取哪个值？

**取当前样本实际接受的 coin，即 A_i。**

例如用户 m = 20，实际分到 coin=5，则敏感度是 15。没有观察到其余档位的标签，就不能凭空给其余档位补监督。

若 logging policy 为 e，期望训练目标的 action 分布为 q，则可使用：

```math
w_i=\frac{q(A_i\mid X_i)}{e(A_i\mid X_i)}g(|m_i-A_i|).
```

等概率 RCT 且目标也等概率时，比值为 1。非均匀分配也不意味着普通条件 CTR 拟合必然有偏；这个比值是为了匹配指定的整体 action 风险。若只是强调高 m 用户，直接用 g(m) 更清楚，不必把 action 敏感度混入同一个目标。

### 7.3 何时加权不改变真实 CTR 目标？

若权重是模型条件输入的严格正函数，且不依赖给定输入后的标签，则固定输入时它只是常数：

```math
\arg\min_q
w(x,c)\left[-p\log q-(1-p)\log(1-q)\right]=p.
```

这只是充分灵活模型下的总体性质，不保证有限样本和有限容量下的校准。若权重依赖 m，而模型条件输入未包含 m 或足以解释它的信息，上述结论不一定成立。

若给正负标签不同权重，则最优输出变成：

```math
q^*(x)=
\frac{w_+p(x)}{w_+p(x)+w_-(1-p(x))}.
```

因此 class weighting、正负样本不等比例采样通常改变输出的概率含义，需要正确修正或在目标分布上校准。

### 7.4 长尾权重的代价与缓解

权重会放大梯度：对 logit z，单条未归一化加权 BCE 有：

```math
g=w(\hat p-y),\qquad h=w\hat p(1-\hat p).
```

可以从温和压缩开始，例如 m 非负时：

```math
w_i=
\min\left\{w_{\max},
\left(1+\frac{m_i}{m_0}\right)^\alpha\right\},
\qquad 0<\alpha<1,\quad m_0>0.
```

再用训练集的固定均值归一化。逐 batch 除以随机权重和会形成另一种随机比率估计，不能总视作完全相同的目标。

常用集中度诊断：

```math
\mathrm{ESS}=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}.
```

ESS 是权重集中的粗略指标，不等于所有模型任务的精确信息量。还应检查最大权重、尾部梯度贡献、验证波动和样本重复次数。

截断、收缩会改变原加权目标；对 IPS 权重截断通常引入偏差。多种权重相乘可能加剧长尾，不能在 m 权重、逆 propensity 和 class weight 上重复叠加却不核算目标分布。

## 8. 样本加权与采样的对比

令归一化目标权重为 a：

```math
a_i=\frac{w_i}{\sum_jw_j},\qquad
L_w=\sum_i a_i\ell_i.
```

若按 $`q_i=a_i`$ 有放回采样，再用未加权 loss，则：

```math
E_{i\sim q}[\nabla\ell_i]=\sum_i a_i\nabla\ell_i.
```

在固定数据集、固定参数点，两者期望梯度一致；有限步训练路径、梯度方差、正则化和树分裂过程不一定一致。

| 维度 | 样本加权 | 过采样 / 分层采样 |
| --- | --- | --- |
| 机制 | 放大单次贡献 | 提高出现频率 |
| 稀疏尾部 | 可能造成单次大梯度 | 更频繁见到尾部，但会重复同一噪声 |
| 信息量 | 不增加独立样本 | 同样不增加独立样本 |
| 计算与覆盖 | 通常保留原数据覆盖 | 预算固定时可能减少其他人群覆盖 |
| 可解释性 | 权重目标较直观 | 必须明确采样概率与修正方式 |
| 校准 | 看权重依赖哪些变量 | 看采样是否改变条件标签分布 |

若为稳定 batch 而改用任意采样概率 q，但仍想优化原来的加权目标，应使用：

```math
\hat g_i=\frac{a_i}{q_i}\nabla\ell_i,\qquad i\sim q.
```

若先按 w 过采样，又未经修正再乘 w，目标会近似变成按 w² 强调样本。温和采样与加权可以组合，但必须计算最终有效权重。

## 9. DFL：训练由预测诱导出的决策

DFL 不是特定模型，而是让训练目标直接感知下游决策质量。标准表述保留预测层和优化器；direct policy learning 则直接输出 action 或 action 概率。二者相关，但不是完全相同的术语。[Wilder 等的原始论文](https://arxiv.org/abs/1809.05504)给出了预测与优化联合训练的代表性框架。

一般形式：预测未知参数 u，求解下游优化问题：

```math
\hat u=f_\theta(x),\qquad
z^*(\hat u)=\arg\min_{z\in\mathcal Z}C(z,\hat u).
```

决策 regret 为：

```math
L_{\mathrm{DFL}}(\hat u,u)
=C(z^*(\hat u),u)-C(z^*(u),u).
```

训练评价的是**预测所诱导的决策在真实目标下的质量**，不是只看预测误差。库存、调度等场景还可能有预算、容量和组合约束。

本 case 中，预测是各 coin 的 CTR，优化器是利润 argmax。加权 BCE 只强调样本重要性；DFL 进一步利用所有 action 的相对价值和错选后果。

硬 argmax 通常分段常数，不能直接获得有用的普通梯度。常见方法包括平滑松弛、可微优化层、决策 surrogate、排序损失和策略梯度。不是所有 DFL 都必须反向穿过一个精确优化器。

## 10. Coin Case 中可训练的 Decision Loss

### 10.1 Softmax 松弛

令温度为 $`\tau>0`$：

```math
\pi_{\theta,c}(x)=
\frac{\exp(\hat V_{\theta,c}(x)/\tau)}
{\sum_{d\in\mathcal C}\exp(\hat V_{\theta,d}(x)/\tau)}.
```

若知道真实各 action 价值，就可以最小化：

```math
L_{\mathrm{dec}}(x)=-\sum_{c\in\mathcal C}\pi_{\theta,c}(x)V_c(x).
```

温度越低越接近硬选择，但梯度可能饱和或集中在很窄区域；温度须与价值的单位、尺度一起设置。训练 soft policy、上线 hard argmax 时，应分别评估二者，不能假设表现完全相同。

**不能把同一个模型自己的预测价值当作无约束的真实奖励再最大化。** 否则模型可能只是抬高预测，而没有学到真实增益。

### 10.2 只有一个 action 的日志：IPS

令观察到的真实净利润为 R_i，策略价值的 IPS 估计为：

```math
\hat J_{\mathrm{IPS}}(\pi)
=\frac1n\sum_{i=1}^n
\frac{\pi(A_i\mid X_i)}{e(A_i\mid X_i)}R_i.
```

可用负估计价值作为 decision loss。需要正确 propensity、因果识别和覆盖；高 m 与小 propensity 叠加时，方差尤其大。

### 10.3 DR：预测基线加残差修正

用独立或交叉拟合的 outcome 模型 $`\hat\mu_c(x)`$ 估计各 action 的条件平均净利润：

```math
\hat J_{\mathrm{DR}}(\pi)
=\frac1n\sum_{i=1}^n
\left[
\sum_{c\in\mathcal C}\pi(c\mid X_i)\hat\mu_c(X_i)
+\frac{\pi(A_i\mid X_i)}{\hat e(A_i\mid X_i)}
\left(R_i-\hat\mu_{A_i}(X_i)\right)
\right].
```

第一项承担可预测部分，第二项用日志残差修正。若结构成立，可由独立 CTR 模型与收入模型构造 $`\hat\mu_c`$。

对固定策略，在识别条件和常规估计条件下，propensity 或 outcome 模型至少一方正确可获得一致性；并不保证每个有限样本估计都无偏，也不保证方差一定低于 IPS。

训练时应冻结或停止梯度进入用于评价的 nuisance 模型，优先用交叉拟合结果；策略的梯度通过 $`\pi_\theta`$ 传播。不能让同一网络同时任意调整“策略”和“评判策略的奖励标签”。

即使用了交叉拟合，在同一份数据上挑选表现最好的策略后再报告其价值，也会有选择乐观偏差。最终仍需要独立测试集或在线实验。

### 10.4 Pairwise surrogate 的适用边界

也可学习各 coin 的价值差排序，而不直接求导 argmax。[Learning to Rank 视角的 DFL 论文](https://proceedings.mlr.press/v162/mandi22a.html)讨论了这种联系。

但本业务缺少每个用户的完整反事实标签，pairwise 标签只能来自额外数据、模拟器或估计；对高噪声 DR 差值直接取符号可能很不稳定。不能把估计出来的最佳 coin 当作无噪声分类真值。

## 11. BCE + Decision Loss：实用的折中方案

一个自然的混合目标是：

```math
L(\theta)=L_{\mathrm{BCE}}(\theta)+\lambda L_{\mathrm{dec}}(\theta),
\qquad \lambda\ge 0.
```

用 DR 价值时，可以明确尺度：

```math
L(\theta)
=L_{\mathrm{WBCE}}(\theta)
-\lambda\frac{\hat J_{\mathrm{DR}}(\pi_\theta)}{s},
\qquad s>0.
```

s 是固定的业务价值尺度。

- BCE 提供点击监督，约束输出接近真实概率。
- Decision loss 引导模型改善最终策略收益。
- 二者混合不保证概率完全校准，也不保证一定优于纯 BCE。

建议先用 BCE 预训练，再从较小 decision 权重开始微调；把 $`\lambda=0`$ 保留为对照。系数不是“两个目标占比”：loss 单位和梯度大小不同，0.5 不代表各一半。

联合选择权重、温度、收益尺度和正则化，评价重点是独立数据上的真实收益、稳定性、校准与成本约束。若目标已经通过高 m 加权强调了价值，再加利润损失时要检查是否过度放大尾部。

## 12. 树模型能做吗？

能。对表格数据，树模型值得作为第一批基线。

| 路线 | 可行性 | 实务定位 |
| --- | --- | --- |
| GBDT + BCE | 直接可行 | 预测各 coin 的 CTR，再计算利润 |
| GBDT + weighted BCE | 直接可行 | 低成本引入用户价值或利润敏感度 |
| 树模型 + 独立校准器 | 直接可行 | 修复概率尺度或分群偏差 |
| 自定义平滑 objective | 取决于接口与损失结构 | 需要正确梯度、Hessian 和数值稳定性 |
| 多 action 耦合的 decision loss | 不能当普通逐行 BCE 直接替换 | 需要用户分组、联合预测及相应近似 |
| 树教师 + 小型策略模型 | 可行 | 用独立 / OOF 估计价值辅助策略学习 |

可以训练共享模型 f(X,coin)，也可以用各 action 模型；前者共享信息，后者更灵活但稀疏档位更容易方差大。树对未观察 coin 的插值和外推能力有限，更换档位仍需要覆盖或可验证的结构假设。

GBDT 自定义 objective 往往要求平滑、逐行可加的损失以及适合的二阶信息；跨 action loss 会产生耦合或非对角 Hessian，不能只写出梯度就认为可以无缝接入。相关限制见 [XGBoost 官方自定义目标文档](https://xgboost.readthedocs.io/en/stable/tutorials/advanced_custom_obj.html)。

因此树模型完全可以做 value-aware 或部分 decision-aware 学习；需要复杂联合反向传播时，神经网络实现通常更直接。

## 13. 什么时候无需 DFL？

以下情况下应优先保留简单方案：

1. BCE 模型在重要 coin × m 区域的 CTR 和价值已足够准确。
2. 策略分歧区域的验证收益已很小，或主要用户的决策间隔明显大于误差。
3. 温和加权、校准或简单业务 surrogate 已稳定改善独立测试和在线收益。
4. 主要瓶颈来自收益口径、m 估计、缺少 action 覆盖或日志偏差；更复杂 loss 无法补足信息。
5. DFL 的估计方差、维护成本超过可验证的增益。

更值得尝试 DFL 的信号是：预测指标已经不错，但独立实验显示仍有明显策略损失；这些错误集中在具有经济意义的 action 竞争区域，而且有足够可靠的数据学习和评估改进。

**无需为了“端到端”而端到端。** 好的业务 surrogate 可能已经编码了决策信息；名称不是判断标准，稳定的净收益才是。

## 14. 推荐实验路线与评估清单

### 14.1 由简到繁做增量实验

| 阶段 | 方案 | 要回答的问题 |
| --- | --- | --- |
| A | 校验单位、支付条件、日志 propensity 和时间口径 | 优化的是否为真实业务利润 |
| B | BCE + 利润 argmax | 当前基线究竟多强 |
| C | 独立全局 / 条件校准 | 问题是否主要是系统偏差 |
| D | 压缩、截断、归一化的 m 权重 | 尾部重分配是否值得 |
| E | 利润敏感度权重，或分层采样及正确修正 | 哪种方式更稳定 |
| F | BCE + IPS/DR decision loss | 决策结构能否带来额外收益 |
| G | 独立离线策略评估与受控在线实验 | 增益是否真实且可持续 |

### 14.2 同时看三类指标

- **预测**：BCE、Brier、AUC；全局与 coin × m 分群校准。
- **决策**：独立 policy value、相对基线的收益差、金币成本、action 分布、策略分歧区域价值、收益置信区间。
- **稳定性**：多时间窗、尾部样本量、ESS、propensity 覆盖、随机种子和极端用户敏感性。

用户重复出现时，区间估计应考虑用户内相关性；长尾下不能只报平均收益点估计。分桶漂亮、训练 DR 价值上升、与老师的 action 一致率高，都不能单独证明真实策略改善。

若存在跨用户预算约束，就不能各用户独立 argmax；预测后应进入带预算的联合决策器，并用同一约束评价策略。

## 15. 核心结论

- **BCE + 利润 argmax 是统计上合理、工程上很强的基线。**
- 高 m 会放大收益误差和标签方差；优先统一收益口径，诊断 m、CTR 和价值的条件校准。
- 加权是调整关注区域，采样是调整出现频率；二者不增加独立信息，组合时需核算最终目标。
- 利用已知利润结构可减少需要学习的自由度；这与真实条件期望降方差相关，但不等于对估计误差的自动保证。
- DFL 的核心是学习哪些预测误差会造成真实决策损失，coin case 还必须处理反事实缺失。
- BCE + decision loss 是可尝试的折中，必须用独立收益评估决定是否保留。
- 简单 loss、校准和树模型已经足够时，无需额外引入 DFL。

## 16. 参考资料与公式格式

本文主要整理原对话，公式推导和业务限定在正文中明确列出。补充参考：

- [Wilder et al.：Melding the Data-Decisions Pipeline](https://arxiv.org/abs/1809.05504)
- [Mandi et al.：Decision-Focused Learning Through the Lens of Learning to Rank](https://proceedings.mlr.press/v162/mandi22a.html)
- [XGBoost：Advanced Usage of Custom Objectives](https://xgboost.readthedocs.io/en/stable/tutorials/advanced_custom_obj.html)
- [GitHub：Writing mathematical expressions](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)

格式遵循仓库笔记规范：块公式采用 math 围栏，行内公式采用 GitHub 的美元符号加反引号语法；避免自定义宏、Unicode 数学控制符和不必要的复杂 LaTeX。GitHub 的数学渲染基于 MathJax，并非 KaTeX；正文使用两者共同支持的基础命令，实际交付仍以 GitHub 页面显示为准。
