# RCT 场景下的两类 CATE 难题：高价值稀有用户 vs 弱效应用户

> 整理自“讲解因果机器学习模型”对话，补充识别条件、验证方法与实务边界。本文是方法笔记，数值仅用于说明，不代表业务实验结果。
>
> **2026-10 增补**：新增第十一节——文献调研中的六个可落地方法（X-learner、重尾稳健推断、CUPED/CUPAC、FDR 控制的 HTE 发现、anytime-valid 顺序检验、DR-learner），延伸阅读同步扩充。
>
> **核心结论：高价值稀有人群往往缺少局部有效样本；弱效应场景往往缺少可辨识的因果信号。两者都要求模型复杂度与有效信息量匹配。**

## 一、统一理论：把基线、因果效应与噪声分开

### 1.1 先固定估计目标与实验条件

设二元处理为 $`T\in\{0,1\}`$，$`X`$ 仅包含处理前可获得的信息，$`Y(t)`$ 是潜在结果。本文首先考虑固定分流概率的 RCT：

```math
P(T=1\mid X)=p,\qquad 0<p<1,
\qquad (Y(0),Y(1))\perp T\mid X.
```

还需一致性、与实验设计相容的无干扰假设，以及明确的观察窗口。若随机的是“是否获邀”，通常估计分配的 ITT 效应，不能直接解释为实际接受处理的效应。聚类随机化应按随机化单位处理依赖和推断。

定义：

```math
\mu_t(x)=E[Y(t)\mid X=x]=E[Y\mid T=t,X=x],
\qquad
\tau(x)=\mu_1(x)-\mu_0(x).
```

$`\tau(x)`$ 是条件平均处理效应（CATE），并不是可以直接观测的个人因果效应。

### 1.2 m(X) 是两组条件均值的加权平均

由全期望公式：

```math
\boxed{
m(X)=E[Y\mid X]=p\mu_1(X)+(1-p)\mu_0(X)
}
```

只有在 50/50 分流时才是算术平均：

```math
p=\frac12
\quad\Longrightarrow\quad
m(X)=\frac{\mu_1(X)+\mu_0(X)}2.
```

因此：

```math
\mu_1(X)=m(X)+(1-p)\tau(X),
\qquad
\mu_0(X)=m(X)-p\tau(X).
```

令 $`\varepsilon=Y-E[Y\mid X,T]`$，得到统一分解：

```math
\boxed{Y=m(X)+(T-p)\tau(X)+\varepsilon},
\qquad E[\varepsilon\mid X,T]=0.
```

| 组成 | 含义 | 建模任务 |
|---|---|---|
| 基线函数 m | 在当前实验混合分配下，可由处理前信息预测的结果水平 | 尽量消除可预测的结果波动 |
| 效应函数 τ | 处理组相对对照组的条件均值差 | 找到可复现的效应异质性 |
| 噪声 ε | 给定特征与处理后仍不可预测的波动 | 承认并量化不确定性 |

这里的“基线”是方便的称呼：$`m(X)`$ 并不等于对照组均值 $`\mu_0(X)`$。

> **框线结论｜结果容易预测，不等于处理效应容易预测。高价值也不等于高 uplift。**

若分层 RCT 各层分流比例不同，应使用已知设计概率 $`e(X)=P(T=1\mid X)`$，把以上所有 $`p`$ 换为 $`e(X)`$；不能仍减去全局平均分流比例。

## 二、两类困难：同样不稳定，不同的信息瓶颈

| 维度 | 高价值稀有用户 | 弱效应用户 |
|---|---|---|
| 典型指标 | 收入、LTV、利润 | 未来 7 天活跃天数 |
| 主要困难 | rare local n：局部样本少，常伴长尾和异方差 | small signal：效应或效应差异小于估计噪声 |
| 效应大小 | 可能很大，也可能没有增量 | 常见平均增量小，需另行判断异质性 |
| 主要误判 | 把少数极端结果当成大 uplift | 反复搜索后把随机差异当成 HTE |
| 数据策略 | 增加真实高价值人群实验样本 | 扩大有效样本、提高结果预测质量 |
| 结构约束 | partial pooling、收缩、较大叶子 | 低维效应模型、少量修饰变量、强正则 |
| 验证重点 | 尾部影响、局部两臂样本、跨批次复现 | 搜索校正、异质性复现、相对常数模型的增益 |
| 共同终点 | 独立样本上的净 policy value 与不确定性 | 独立样本上的净 policy value 与不确定性 |

对两个预先定义、互不重叠的人群 A、B，可用下式理解异质性可辨识度：

```math
\mathrm{SNR}_{\mathrm{HTE}}
\approx
\frac{\lvert\tau_A-\tau_B\rvert}
{\mathrm{SE}(\widehat\tau_A-\widehat\tau_B)}.
```

高价值问题常使分母变大；弱异质性问题使分子变小。二者可以同时存在，并不是互斥分类。

尤其要区分：

```math
\boxed{
\lvert E[\tau(X)]\rvert\approx0
\;\not\Rightarrow\;
\mathrm{Var}(\tau(X))\approx0
}
```

正负效应可能互相抵消。小 ATE 不能证明“没有值得学习的异质性”；同样，大 ATE 也不能证明“需要个性化”。

## 三、共同底座：OOF / cross-fitting residualization

### 3.1 为什么先预测结果，再学习效应

把处理组与对照组混合，用处理前 $`X`$ 预测 $`Y`$，不把 $`T`$ 输入这个 pooled 模型，目标就是 $`m(X)`$。也可以分别估计两臂均值后按设计概率组合；哪种更好由独立验证决定。

以高价值用户为例，若结果为 1000，预测基线为 950，则残差为 50。但 **50 不是该用户的处理效应**，它同时含有中心化的处理效应与噪声：

```math
Y-m(X)=(T-p)\tau(X)+\varepsilon.
```

50/50 RCT 中，两组条件残差均值分别为正负半个效应：

```math
E[Y-m(X)\mid X,T=1]=\frac{\tau(X)}2,
\qquad
E[Y-m(X)\mid X,T=0]=-\frac{\tau(X)}2.
```

它们的差才是 CATE。

### 3.2 OOF 的操作步骤

1. 先留出最终测试集，之后不用于特征筛选、调参或策略阈值选择。
2. 将开发样本按用户或随机化单位分成 K 折；同一用户的多行记录不跨折。
3. 对每一折，用其余折训练结果模型，预测当前折，得到 OOF 预测。
4. 汇总残差，再在开发数据内训练和选择效应模型。
5. 完成选择后冻结模型及策略，在最终测试集评价一次。

```math
R_{Y,i}=Y_i-\widehat m^{(-k(i))}(X_i),
\qquad
R_{T,i}=T_i-p_i.
```

RCT 中 $`p_i`$ 取已知设计概率，通常无需用模型重新估计。

注意估计残差关系中仍存在第一阶段误差：

```math
R_{Y,i}
=
R_{T,i}\tau(X_i)+\varepsilon_i
+
\left[m(X_i)-\widehat m^{(-k(i))}(X_i)\right].
```

OOF 避免把训练内拟合噪声误当成可消除的波动，并不使最后一项凭空消失。

### 3.3 R-Learner 的第二阶段

```math
\boxed{
\widehat\tau
=
\arg\min_{f\in\mathcal F}
\left\{
\frac1n\sum_{i=1}^n
\left[R_{Y,i}-R_{T,i}f(X_i)\right]^2
+\lambda J(f)
\right\}
}
```

这里 $`J(f)`$ 约束复杂度。直接把 $`R_Y`$ 对 $`X`$ 做普通回归不是这个目标。

等价的伪结果形式为：

```math
Z_i=\frac{R_{Y,i}}{R_{T,i}},
\qquad w_i=R_{T,i}^2,
\qquad
\sum_i w_i[Z_i-f(X_i)]^2.
```

不等分流时不能丢掉权重；直接优化原始 R-loss 更便于避免除法带来的数值问题。

在已知分流概率、折外预测等条件下，结果模型误差与中心化处理满足相应的条件正交关系：

```math
E[(T-p)\{m(X)-\widehat m(X)\}\mid X]=0.
```

这解释了 residualization 的稳健性，但不保证任意有限样本、任意正则或任意学习器都无偏。R-Learner 的框架和相应理论见 [Nie 与 Wager 原论文](https://arxiv.org/abs/1712.04912)。

> **框线结论｜cross-fitting 防止第一阶段的训练内泄漏；honesty 分离树结构选择与叶内估计；独立测试集检验整个选择过程。三者各有职责，不能互相替代。**

## 四、高价值稀有用户：效应可能大，但局部信息很少

### 4.1 人群必须在处理前定义

例如按过去 28 天收入、历史付费等级或处理前预测价值划分高价值人群。不能按实验后的收入挑出“高价值用户”再比较两组，否则处理本身会改变入选资格，破坏原来的随机可比性。

高价值群体 H 的简单两臂均值差，在有限方差及相应独立抽样近似下有：

```math
\mathrm{Var}(\widehat\tau_H)
\approx
\frac{\sigma_{1,H}^2}{n_{1,H}}
+
\frac{\sigma_{0,H}^2}{n_{0,H}}.
```

总实验人数很多，不意味着 $`n_{1,H}`$ 与 $`n_{0,H}`$ 足够。少量大额交易又可能使两臂方差很高。

收入右偏并不自动等于数学意义上的无限方差重尾。若二阶矩不存在，常规标准误、中心极限定理近似和普通 bootstrap 都可能失效。

### 4.2 residual tail diagnostics：看去基线之后还剩什么

先做好 OOF 结果模型，再按处理臂及预定义价值层检查：

- 残差分位数、平方残差贡献与极端点占比。
- 最大若干观测对组效应和策略价值的影响。
- 各臂真实独立用户数，聚类数及加权有效样本量。
- 删除单个高影响单位后的敏感性，以及跨时间批次的复现。
- 预测基线是否在高价值层系统性失准。

若少量点贡献绝大部分残差平方和，问题仍然是尾部噪声，增加效应模型复杂度通常无法解决。

> **框线结论｜基线建模可以移除可预测的高价值；不能移除不可预测的巨额波动。诊断对象应从原始标签延伸到折外残差。**

不应为了让结果显著而删除大额真实交易。截尾、winsorization、对数变换会改变目标或引入偏差；若作为敏感性分析，必须同时报告原始均值目标与变换后的目标。对数尺度效应不能直接指数变换成原始收入均值效应。

### 4.3 partial pooling：信息不足时，向共同结构收缩

设预定义人群 g 的原始效应估计和方差为：

```math
\widehat\tau_g\mid\tau_g\approx N(\tau_g,s_g^2),
\qquad
\tau_g\sim N(\tau_0,a^2).
```

在固定超参数的正态近似下，后验均值为：

```math
\boxed{
\widetilde\tau_g
=
w_g\widehat\tau_g+(1-w_g)\tau_0,
\qquad
w_g=\frac{a^2}{a^2+s_g^2}
}
```

局部方差越大，越应借用总体或相似人群的信息。实际可用层级模型、正则化交互项或低维共享结构，而不是为每个稀有子群独立训练复杂模型。

收缩中心也可以随少量可信变量变化。若超参数由数据估计，区间需考虑其不确定性；过强 pooling 可能掩盖真实特殊群体，需做收缩强度敏感性分析。

### 4.4 honesty 与 large leaf

树模型会搜索“哪里两组差最多”。如果同一份数据又决定切分又估计差异，极端点容易同时制造分组和高效应。

Honesty 使用不同子样本选择分裂和估计叶内效应。较大叶子、浅树及每臂支持约束，可减少局部自由度。但 honesty 也消耗局部样本预算，因此稀有群体更需要避免过细切分。

实现时检查最终估计样本中的每臂人数，不能只看配置里的叶大小参数。森林树数增加主要降低算法随机性，不会创造新的独立高价值用户。实现细节见 [GRF 官方说明](https://grf-labs.github.io/grf/REFERENCE.html)。

### 4.5 oversampling / stratified RCT：从设计端增加信息

真正有用的 oversampling 是**额外招募高价值人群进入实验**，并在层内随机分配。复制已有样本、反复抽同一个 whale，不能增加独立信息。

分清两种概率：

| 概率 | 作用 | 需要处理的事情 |
|---|---|---|
| 入样概率 | 决定哪些用户进入实验 | 从富集样本推广到总体时进行抽样加权或分层汇总 |
| 处理分配概率 | 决定入样后分到哪一臂 | 使用对应的已知倾向概率做因果估计 |

若目标总体中各层占比为 $`q_g`$，则：

```math
\tau_{\mathrm{target}}=\sum_g q_g\tau_g.
```

不能用富集实验的样本层占比替代目标总体占比。推广还要求在给定分层变量后，入样与目标人群的效应具有可迁移性。

在固定层样本量、同等成本及方差已知的理想近似中，最小化均值差方差的分配满足：

```math
\frac{n_{1,g}}{n_{0,g}}=\frac{\sigma_{1,g}}{\sigma_{0,g}}.
```

这不是“高价值用户都进实验组”的理由；必须保留两臂支持，并综合成本、未知方差与运营约束设计实验。

## 五、弱效应用户：避免从大量噪声里搜索出异质性

### 5.1 small signal 与 max-over-noise

假设未来 7 天活跃天数属于 $`\{0,\ldots,7\}`$，平均提升仅为 0.03 天。指标有界，只说明尾部受约束，不说明相对效应的噪声小。

若真实效应接近常数，搜索 M 个分组后挑最大的估计：

```math
\widehat\tau_j=\tau_0+\eta_j,
\qquad
\max_{1\le j\le M}\widehat\tau_j
=
\tau_0+\max_{1\le j\le M}\eta_j.
```

即使每个噪声均值为零，最大值的期望通常也为正。独立同方差高斯噪声下，大 M 时有启发性近似：

```math
E[\max_j\eta_j]\approx s\sqrt{2\log M}.
```

真实候选通常相关，该式不是通用校正公式，但揭示了搜索代价。搜索特征、切分点、模型、随机种子与策略阈值都在扩大选择空间。

> **框线结论｜“找到一个很高的 uplift”与“找到可复现的 uplift”是两件事。搜索自由度也要计入统计预算。**

### 5.2 strong m(X)：先增加可预测信息

可使用处理前的：

- 7/14/28 天活跃天数与活跃时长；
- session 数、登录次数、连续活跃天数；
- recency、生命周期、活跃趋势；
- 处理前已确定的渠道、设备或历史消费信息。

两人过去都活跃 7 天，但每天 5 分钟与每天 3 小时，可能代表不同的未来基线。辅助指标有价值的条件是它们确实改善折外预测，而非仅改善训练集拟合。

oracle 分解下：

```math
E[(Y-m(X))^2\mid X]
=
p(1-p)\tau(X)^2+E[\varepsilon^2\mid X].
```

残差保留因果信号与不可解释噪声。提高第一阶段质量可减少估计误差，但“m 更准”不保证任意有限样本 CATE 方差单调下降，最终仍看独立验证。

### 5.3 丰富的结果特征，少量的 effect modifiers

预测 $`Y`$ 的重要变量未必能解释 $`\tau(X)`$。建议区分两套特征预算：

```math
\boxed{
\mathrm{Complexity}(\tau)
\ll
\mathrm{Complexity}(m)
}
```

这是弱异质性场景的建模原则，不是数学定理。

结果模型可以较丰富；效应模型先只保留少量有机制依据的变量，如历史活跃、recency、生命周期。特征筛选与参数选择必须在开发样本内完成，不能先用全数据筛选再声称测试集独立。

一个结构化模型是：

```math
\tau(X)=\beta_0+f_1(X_{\mathrm{activity}})
+f_2(X_{\mathrm{recency}})
+f_3(X_{\mathrm{tenure}}).
```

采用低自由度 spline / GAM 或少量正则交互项；必要时把偏离总体效应的部分收缩到零。

### 5.4 post-treatment mediator：不能当普通辅助协变量扣掉

若实验后活跃时长 M 位于如下路径：

```math
T\longrightarrow M\longrightarrow Y,
```

把它加入 $`m(X,M)`$ 再残差化，可能扣掉真实间接效应，或产生碰撞点偏差。简单调整 mediator **并不能自动识别直接效应**；直接效应需要额外定义与识别假设。

> **框线结论｜估计总效应时，主模型使用处理前协变量。处理后辅助指标可以作为辅助结果，不能随意作为基线控制变量。**

可把活跃天数与活跃时长作为两个结果，共享表示但保留独立效应输出：

```math
X\longrightarrow h(X)\longrightarrow
\begin{cases}
\tau_{\mathrm{days}}(X),\\
\tau_{\mathrm{duration}}(X).
\end{cases}
```

只有两个任务的效应异质性确有共享结构时，多任务学习才可能帮助；辅助任务也可能产生负迁移。以主结果的独立评估为准，部署输入仍只用决策时可获得的信息。

### 5.5 复杂度阶梯：每升一级都要有证据

```math
\boxed{
\text{Constant effect}
\;\longrightarrow\;
\text{Structured CATE}
\;\longrightarrow\;
\text{Flexible R-Learner}
\;\longrightarrow\;
\text{Causal Forest}
}
```

| 阶段 | 具体做法 | 升级依据 |
|---|---|---|
| 常数效应 | 一个总体效应参数 | 少量预定义人群有可复现差异 |
| 结构化 CATE | 线性修饰项、低自由度 GAM、少量组效应 | 结构约束遗漏稳定非线性 |
| 灵活 R-Learner | 浅树、小叶数、大叶样本、强 L1/L2、早停 | 验证集显示额外价值 |
| Causal Forest | honesty、较大估计叶、支持约束 | 足够局部信息且独立策略评价改善 |

这是实践上的探索顺序，并不是算法的严格包含或复杂度排序：常数模型和 GAM 都可以使用 R-loss；正交化 CF 与 R-Learner 也有联系。受约束的 CF 可能比自由度很高的 boosting 更保守。

多 seed 用来检查算法稳定性，不能替代独立批次复现。若较复杂模型没有稳定增益，停在常数效应或结构化模型是合理结果。

## 六、验证：从“预测得像”到“因果差异可复现”

### 6.1 个体 CATE 真值不可直接观测

单个用户只能观测一种潜在结果，因此不能直接在真实 RCT 数据上计算个人 CATE 的普通 RMSE。需要多层证据：

| 验证层 | 高价值场景重点 | 弱效应场景重点 |
|---|---|---|
| 实验有效性 | 分层随机化、两臂样本、标签成熟、异常交易 | 分流、缺失、观察窗口、功效 |
| 第一阶段 | 价值层内 OOF 校准与残差尾部 | OOF 误差相对简单基线的下降 |
| 因果预测 | 预定义组效应及区间、尾部敏感性 | 常数效应基准、整体异质性检验 |
| 排序与校准 | 独立样本高分组是否仍有增量 | 分组效应趋势是否复现 |
| 决策 | 净增量是否由少数 whale 主导 | 个性化是否优于全处理或不处理 |

对冻结的效应预测，在独立样本上按开发阶段确定的分位点分组，再估计组内实验效应及置信区间。若检查很多组或很多阈值，应预注册少量主要对比，或采用适当的多重检验控制。探索性发现要标明并另行验证。

### 6.2 验证集 R-loss

```math
\widehat L_R(f)
=
\frac1{n_{\mathrm{test}}}
\sum_{i\in\mathrm{test}}
\left[Y_i-\widehat m_{\mathrm{train}}(X_i)
-(T_i-p_i)f(X_i)\right]^2.
```

在 oracle nuisance 条件下，相对真效应的总体超额风险为：

```math
L_R(f)-L_R(\tau)
=
E[e(X)(1-e(X))\{f(X)-\tau(X)\}^2].
```

固定分流时，它与 CATE 均方误差成比例；有限样本中的验证损失仍可能很嘈杂，微小领先不是充分证据。验证集用于选择后，最终测试集继续保持独立。

### 6.3 不确定性必须匹配设计

按随机化单位或聚类单位估计标准误、重抽样，并保留分层结构。不能把同一用户的多天记录当成独立样本。

对高价值尾部，报告方差贡献和影响诊断；对弱效应，预先做最小可检测效应与异质性对比的功效分析。固定分流、局部近似同方差和有限方差条件下：

```math
\mathrm{SE}(\widehat\tau_g)
\approx
\frac{\sigma_{\mathrm{res},g}}
{\sqrt{n_g p(1-p)}}.
```

若其他因素相同，希望识别的差异缩小一半，所需样本量约变为四倍。这里是功效直觉，真实设计还需处理多重比较、异方差与聚类。

## 七、policy value：建模最终服务什么决策

### 7.1 从 CATE 到净收益

设 $`\pi(X)\in\{0,1\}`$ 为策略，$`c(X)`$ 为处理的增量成本，且成本已转换为与结果相同的效用单位：

```math
V(\pi)
=
E[Y(0)+\pi(X)\{\tau(X)-c(X)\}].
```

相对“全部不处理”的净增量为：

```math
\boxed{
\Delta V(\pi)=E[\pi(X)\{\tau(X)-c(X)\}]
}
```

没有容量限制时，oracle 策略为：

```math
\pi^*(X)=\mathbf 1\{\tau(X)>c(X)\}.
```

活跃天数不能直接减去货币成本，需先定义业务效用。若成本本身也是随机、受处理影响的结果，可直接对净效用结果建模。容量约束下应明确预算并比较同等处理比例的策略。

### 7.2 独立 RCT 上用正交得分评估

用测试集之外拟合的两臂结果预测，构造 AIPW / DR 效应得分：

```math
\widehat\Gamma_i
=
\widehat\mu_1(X_i)-\widehat\mu_0(X_i)
+
\frac{T_i}{p_i}\{Y_i-\widehat\mu_1(X_i)\}
-
\frac{1-T_i}{1-p_i}\{Y_i-\widehat\mu_0(X_i)\}.
```

对冻结策略：

```math
\widehat{\Delta V}(\pi)
=
\frac1{n_{\mathrm{test}}}
\sum_{i\in\mathrm{test}}
\pi(X_i)\{\widehat\Gamma_i-c(X_i)\}.
```

已知随机化概率与独立拟合使该得分有清晰的随机化基础。若评估人群经过富集抽样，还需按目标总体分布汇总或加权。正交得分与策略学习的理论背景见 [Athey 与 Wager 原论文](https://arxiv.org/abs/1702.02896)。

至少比较：全部不处理、全部处理、现有业务策略、常数效应规则与个性化策略。在相同测试用户上比较策略差值：

```math
\widehat D_i
=
\{\pi_a(X_i)-\pi_b(X_i)\}
\{\widehat\Gamma_i-c(X_i)\}.
```

再依据实验设计为平均差构造区间。策略阈值如果是在同一测试集上挑出的，区间不再具有原来的解释，需要新的独立评价。

AUUC、Qini 或其他排序指标可作辅助诊断，但无法单独回答净收益、成本约束与是否优于现有策略。

> **框线结论｜CATE 数值很漂亮，不等于策略有价值；策略有价值，也不要求把每一个人的 CATE 都估得很精确。**

## 八、从理论到实务：两条可执行 pipeline

### 8.1 高价值稀有人群

1. 固定原始收入或利润目标、成熟窗口、处理前价值分层和目标总体。
2. 汇总每层两臂独立样本与尾部贡献，判断是否需要额外实验招募。
3. 训练 pooled OOF 结果模型，检查高价值层残差尾部。
4. 以总体效应和预定义分层效应为基线，引入 partial pooling。
5. 信息足够时尝试 honest CF 或受约束 R-Learner，检查实际局部支持。
6. 独立评价组效应、跨批次复现、尾部影响与净 policy value。
7. 若结论仍由少数点决定，优先补充真实样本或简化目标结构。

### 8.2 活跃天数弱效应

1. 先估计总体效应及功效，区分小 ATE 与小异质性。
2. 充分使用处理前活跃时长、历史活跃与 recency 改善 OOF 结果预测。
3. 从常数效应开始，加入少量预先选择的 effect modifiers。
4. 逐级尝试低维结构、强正则 R-Learner 和受约束 CF。
5. 把特征、模型、阈值和辅助任务选择留在开发阶段。
6. 独立验证异质性与策略净增益，避免只看最好的子群。
7. 没有稳定个性化证据时采用简单规则，并把下一轮实验用于增加信息。

## 九、统一统计哲学：统计预算必须支付模型自由度

### 9.1 有效信息量不等于数据表行数

加权样本常用的诊断量是：

```math
n_{\mathrm{eff}}
=
\frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
```

它只描述权重集中度，不能完整替代因果信息量；还需分别看两臂支持、局部邻域、残差方差、聚类依赖和搜索规模。

在局部近似同方差、独立且有限方差条件下，可用下面的启发性量理解“能识别多大的异质性差异”：

```math
\mathcal I_{\mathrm{HTE}}
\propto
\frac{n_{\mathrm{local}}p(1-p)\delta^2}
{\sigma_{\mathrm{res}}^2}.
```

其中 $`\delta`$ 是关心的效应差异。这不是适用于所有模型的精确 Fisher 信息公式。

### 9.2 每种手段都在调整信息与自由度

| 方法 | 对统计预算的作用 | 不能解决什么 |
|---|---|---|
| 强处理前结果预测 | 降低可解释的结果波动 | 不可预测尾部与缺失实验支持 |
| 真实富集招募 | 增加稀有人群局部信息 | 自动保证总体代表性 |
| partial pooling | 用结构假设借用邻近人群信息 | 无代价保留任意局部异质性 |
| 低维效应模型与强正则 | 减少需要估计的自由度 | 修复错误因果识别 |
| cross-fitting | 降低 nuisance 训练内过拟合影响 | 消除所有模型选择偏差 |
| honesty | 分离切分选择与叶内估计 | 创造更多稀有用户 |
| 独立策略验证 | 检验整条研究流程是否泛化 | 让极少样本产生高功效 |

可把最终原则写为：

```math
\boxed{
\text{Model complexity}
\quad\text{must match}\quad
\text{effective information}
}
```

高价值用户的信息预算常被局部稀缺与尾部方差耗尽；弱效应用户的信息预算常被低信噪比与反复搜索耗尽。更复杂的模型只有在独立证据支持时才值得增加。

> **最终结论｜增加有效信息、减少无效自由度、分离选择与评价，并围绕目标总体的决策价值验证。模型的任务是提取数据支持的异质性，而不是把所有随机差异解释成异质性。**

## 十一、文献调研补充：六个可落地的方法

> 本节补足正文六处“指出了问题、但未给出正式解法或工具”的空缺。每小节先注明对应正文位置；收录标准是具备严格理论或工业级实现。

### 11.1 X-learner：两臂样本悬殊时的效应学习器（对应 §4.1 局部样本）

正文核心困难是“局部两臂样本都少”，现实更常见的形态是两臂悬殊。Künzel、Sekhon、Bickel 与 Yu（PNAS 2019）把 CATE 估计器整理为 meta-learner 框架：S-learner（单模型含处理指示）、T-learner（两臂各建模型）、X-learner。X-learner 三步：

1. 分臂估计 $`\widehat\mu_0,\widehat\mu_1`$；
2. 用互补伪结果回归两个效应模型：对照组 $`\widehat\Delta_0=Y_0-\widehat\mu_1(X_0)`$，处理组 $`\widehat\Delta_1=\widehat\mu_1(X_1)-Y_1`$，得 $`\widehat\tau_0,\widehat\tau_1`$；
3. 按倾向得分合成 $`\widehat\tau(x)=g(x)\widehat\tau_0(x)+\{1-g(x)\}\widehat\tau_1(x)`$。

理论卖点：**一臂样本远多于另一臂时可证有效**——多臂信息经填补路径传给少臂；若 CATE 结构简单（如线性而响应函数 Lipschitz），正则条件下仍达参数速率。直觉：CATE 通常比两臂响应函数更平滑，先让充足样本扛响应函数、再利用简单性估差异。边界：合成权重依赖倾向得分估计；文献模拟中无 meta-learner 一致最优；应作为复杂度阶梯中“结构化 CATE”层级的候选，由独立验证裁决。

### 11.2 重尾的正式处理：从截尾规范到无穷方差推断（对应 §4.2 尾部诊断）

正文指出了失效条件（“二阶矩不存在时常规标准误与 bootstrap 失效”），但未给解法。分两层补齐。

实务层——winsorization 规范（大规模实验平台的标准做法）：分位数用**合并样本**计算（分臂各算会引入不公平截断）；阈值**预先注册**进指标定义（事后挑阈值是研究者自由度，直接虚增假阳性）；诚实报告 estimand 的变化——截尾效应是“截尾均值之差”，合法但不同，须同时报告原始目标与多档阈值（1/99、2/98、5/95）敏感性表。

理论层——稳健估计：**中位数均值**（分桶取均值、桶间取中位数）与 **Catoni 型估计**在只知方差上界时给出有效置信区间，单个极端值只能毁掉一个桶；**Catoni 型置信序列**（Wang & Ramdas 2023）把该性质推广到任意停时有效，且对无穷方差（仅 $`p`$ 阶矩存在，$`1<p\le2`$）亦有构造（Bhatt et al. 2022），收敛速率由 $`1/\sqrt t`$ 退为 $`O((\log\log t/t)^{(p-1)/p})`$。

> **框线结论｜截尾管“可容忍的尾部”，稳健估计管“无法截断的尾部”，复杂效应模型管不了任何一个。**

### 11.3 CUPED / CUPAC：强 m(X) 的工程化形态（对应 §5.2 与 §4.2）

CUPED（Deng et al., WSDM 2013，Bing）构造调整结果：

```math
Y^{\mathrm{cuped}}=Y-\theta\,(X-\bar X),
\qquad
\theta=\frac{\mathrm{Cov}(Y,X)}{\mathrm{Var}(X)},
\qquad
\mathrm{Var}(Y^{\mathrm{cuped}})=\mathrm{Var}(Y)(1-\rho^2)
```

$`X`$ 为处理前同指标或强相关指标，$`\rho`$ 为其与结果的相关。Bing 实测约 50% 缩减——等价于用户数或时长减半；DiD 是 $`\theta\equiv1`$ 特例，最优 $`\theta`$ 永不劣于 DiD。CUPAC 用处理前特征训练的预测替代单一协变量，即正文“强 m(X)”的直接工程对应。

对弱效应：0.03 天量级的效应配上减半方差，功效等价样本翻倍，是不动实验设计能拿到的最大杠杆之一。对高价值：CUPED 只移除**可预测**的价值波动，whale 的不可预测部分按 $`(1-\rho^2)`$ 原样留在残差中——定量印证正文“基线建模不能移除不可预测的巨额波动”。适用条件：协变量必须在随机化之前测量；$`\theta`$ 用合并样本估计；协变量以总体均值中心化；新用户需单独处理；收益完全由 $`\rho^2`$ 决定，弱相关时（$`\rho=0.3`$ 仅 9%）不值得加系统复杂度。可与截尾、trigger 分析叠加，各除一种噪声。

### 11.4 FDR 控制的 HTE 发现：把“搜索校正”变成机器（对应 §5.1 与 §5.5）

正文给出了搜索假阳性的形态与原则，这里补可执行工具。利用变换结果 $`Z_i=Y_i(T_i-p)/\{p(1-p)\}`$（条件期望恰为 $`\tau(X_i)`$），把子群发现转为多重检验：

- **HTE-BH**：对预定义子群跑 Benjamini–Hochberg，控制 FDR 于 $`q`$（在线实验案例 $`q=0.2`$）；
- **HTE-Knockoff**：model-X knockoff 过滤，**有限样本**控制 FDR，信号微弱时比 BH 更保守（主动少报以省调查成本）。

在线实验实证（Pang et al.）：朴素方法 FDR 失控、Bonferroni 功效大减；受控搜索在真实数据中发现了事后可解释的异质性（新排版利好非英语内容）。更进一步，子群选择性 knockoff 过滤（Gablenz et al., JASA）允许**数据驱动地**提出局部条件假设（“$`X_j`$ 仅在子群 $`g`$ 内有效应”），同一份数据既学习假设又检验，无需样本切分且 FDR 受控——把正文“搜索自由度计入预算”落实为“搜索的代价可显式计价并控制”。边界：BH 在统计量相关、设计非正交时可失守；knockoff 依赖协变量分布知识（model-X 假设）；$`q=0.2`$ 意味着清单平均两成是噪声。

### 11.5 anytime-valid 顺序检验：偷看的明码标价（对应 §5.1 搜索预算）

弱效应实验跑得久，天然诱发偷看。固定样本检验在中途反复查看并在 $`p<0.05`$ 时停止，两周日查的假阳率可升至约 **26%**（Johari et al. 模拟口径）。mSPRT 对效应取先验混合构造似然比 $`\Lambda_t`$，报告 $`p_t=\min(1,1/\Lambda_t)`$；零假设下 $`\Lambda_t`$ 是鞅，故 $`P(\exists t: p_t\le\alpha\mid H_0)\le\alpha`$——**任意停时有效**。Wish 的平台已落地（always-valid p-value），Optimizely、Statsig 同类实现。代价：同功效约需 20%–40% 更大最大样本；固定次数中期查看可用 alpha spending（O'Brien–Fleming 约 +2%–4%，Pocock 约 +15%–18%）更省但日程死板。高价值场景中期估计被尾部大单剧烈摆动时，应与 11.2 的稳健估计配合，避免单个 whale 制造中途假显著。

### 11.6 DR-learner：让学习与评估同源（对应 §3.3 与 §7.2）

正文 §7.2 用 DR 得分评估策略、R-Learner 学习效应；Kennedy（2023）指出同一个 DR 伪结果可直接回归出 CATE：

```math
\widehat\phi_i=
\widehat\mu_1(X_i)-\widehat\mu_0(X_i)
+\frac{T_i}{\widehat e(X_i)}\{Y_i-\widehat\mu_1(X_i)\}
-\frac{1-T_i}{1-\widehat e(X_i)}\{Y_i-\widehat\mu_0(X_i)\},
\qquad
\widehat\tau_{\mathrm{DR}}(x)=\mathbb E_n[\widehat\phi_i\mid X_i=x]
```

$`E[\widehat\phi_i\mid X=x]=\tau(x)`$。理论优势在二阶余项：第一阶段误差以乘积形式进入，对 nuisance 收敛速率的要求**低于** R-Learner 类方法（后者常需 $`n^{-1/4}`$）；CATE 较平滑时 oracle 效率条件进一步放宽。实践意义：学习与评估用同一正交化物件，避免“学习一套分数、评估另一套分数”的隐蔽错配。边界：倾向得分接近 0/1 时伪结果方差爆炸，需裁剪或稳定化权重；两臂悬殊场景应与 X-learner 在独立验证中赛马。

### 11.7 一张根基注脚：partial pooling 的经典出身

§4.3 的收缩权重 $`w_g=a^2/(a^2+s_g^2)`$ 是 James–Stein 收缩与经验贝叶斯的直接后裔（Efron & Morris 1975：15 名棒球手半季数据向均值收缩后，对下半季预测全面优于原始估计）。Stein 悖论：三个及以上参数联合收缩可在均方误差上严格优于各自单独估计。partial pooling 因此不是可选技巧，而是高维估计的默认最优形态——稀有子群越少越嘈杂，收缩收益越大。

> **本节小结｜X-learner 与 CUPED 补分子（把信息传给少臂、把可预测波动移走）；截尾与稳健估计压分母（把尾部从不可推断变成可推断）；FDR 过滤与 mSPRT 管过程（把特征搜索与时间偷看都明码计价）；DR-learner 保证学习与评估同源。**

---

## 十二、延伸阅读

- [Quasi-Oracle Estimation of Heterogeneous Treatment Effects — Nie & Wager](https://arxiv.org/abs/1712.04912)：R-Learner 与 nuisance residualization。
- [GRF 官方算法说明](https://grf-labs.github.io/grf/REFERENCE.html)：honesty、局部估计、正交化与实现边界。
- [Policy Learning with Observational Data — Athey & Wager](https://arxiv.org/abs/1702.02896)：正交得分与策略价值。
- **第十一节增补文献**：
  - [Metalearners for estimating heterogeneous treatment effects — Künzel, Sekhon, Bickel & Yu](https://arxiv.org/abs/1706.03461)：X-learner，两臂不均衡时可证有效（PNAS 2019）。
  - [CUPED — Deng, Xu, Kohavi & Walker](https://ai.stanford.edu/~ronnyk/2013-02CUPEDImprovingSensitivityOfControlledExperiments.pdf)：处理前数据的方差缩减（WSDM 2013）。
  - [Catoni-style confidence sequences — Wang & Ramdas](https://doi.org/10.1016/j.spa.2023.05.007)：重尾与无穷方差下的置信序列。
  - [FDR Controlled HTE Detection for Online Controlled Experiments — Pang et al.](https://arxiv.org/abs/1808.04904)：HTE-BH 与 HTE-Knockoff。
  - [Searching for local associations while controlling the FDR — Gablenz et al.](https://arxiv.org/abs/2412.02182)：子群选择性 knockoff 过滤（JASA）。
  - [Always valid inference: Continuous monitoring of A/B tests — Johari, Pekelis & Walsh](https://arxiv.org/abs/1512.04922)：mSPRT 与 always-valid p-value。
  - [Towards optimal doubly robust estimation of heterogeneous causal effects — Kennedy](https://arxiv.org/abs/2004.14497)：DR-learner。
- [本仓库：选择与估计分离](选择与估计分离.md)。
- [本仓库：互联网新用户 LTV 长尾建模](新用户LTV长尾建模-理论与实务.md)。
- [本仓库：现代统计思想](现代统计思想-复杂理论背后的统计哲学.md)。

---

排版约定：块公式使用 GitHub 支持的 math 围栏；行内公式使用美元符号包裹反引号；表格内尽量用文字，避免条件概率竖线干扰 Markdown 列分隔。
