# AFML 第 6—9 章：泛化控制与研究哲学

> 阅读对象：Marcos López de Prado，《Advances in Financial Machine Learning》（2018）第 6—9 章。基于“交易时间与日历时间”对话整理，结合统计推导与实务评议。本文区分书中方法、一般统计原理和扩展建议，不把经验方法写成普遍定理。整理日期：2026-10-07。
>
> 前篇：[第 2—5 章：统计哲学、统一逻辑与量化实务](AFML-第2-5章-统计哲学与量化实务.md)。

## 1. 主线：防止把历史偶然性当作稳定规律

第 2—5 章定义样本、标签、权重与特征；第 6—9 章追问：在这些对象上训练出来的模型，为什么值得相信？

| 章节 | 主要风险 | 方法 | 无法单独解决的问题 |
| --- | --- | --- | --- |
| 6：集成 | 单模型对训练扰动过于敏感 | Bagging、随机森林、boosting | 共同偏差、泄漏、制度变化 |
| 7：交叉验证 | 训练间接见过测试信息 | Purging、embargo、时间切分 | 反复选择带来的研究过拟合 |
| 8：特征重要性 | 追逐高分而不理解信息来源 | MDI、MDA、SFI、跨资产比较 | 因果识别、自动发现稳定机制 |
| 9：调参 | 从噪声评分中挑出幸运赢家 | 有限预算搜索、nested CV、合适评分 | 外层结果被反复用于修改研究 |

四章形成递进关系：**让预测更稳定 → 让评估更可信 → 理解模型依赖什么 → 让选择过程也接受独立评估。**

集成降低预测波动，purging 修复一种信息污染，独立外层评估处理选择偏差，研究记录约束人为反复试验。它们作用在不同层次，不能互相替代。

## 2. Chapter 6：集成的关键是误差结构

### 2.1 方差推导：模型数量为什么会遇到收益递减？

固定预测点，第  $b$  个学习器的预测为  $f_b$ ，随机性来自训练样本与算法随机化。平均预测为：

$$
\bar f=\frac{1}{B}\sum_{b=1}^{B}f_b.
$$

若各学习器方差相同为  $\sigma^2$ ，两两相关系数相同为  $\rho$ ，则：

$$
\mathrm{Var}(\bar f)=\frac{1}{B^2}\left[B\sigma^2+B(B-1)\rho\sigma^2\right]
=\sigma^2\left[\rho+\frac{1-\rho}{B}\right].
$$

当共同相关性保持为正，增加模型数只能压低第二项。以  $\rho=0.2$  为例，10 个模型的方差为  $0.28\sigma^2$ ，100 个为  $0.208\sigma^2$ ，极限为  $0.2\sigma^2$ 。降低共同误差往往比继续加树更有价值。

这是具有明确假设的方差恒等式，不是分类准确率或策略 Sharpe 的公式。一般加权集成满足：

$$
\mathrm{Var}\left(\sum_b a_b f_b\right)=a^\top\Sigma a,\qquad \sum_b a_b=1.
$$

降低相关性可能同时削弱单个学习器，因此“弱而不同一定优于强而相似”并不成立。对平方损失，在通常的回归噪声假设下：

$$
\mathrm{MSE}(x)=\mathrm{Bias}(x)^2+\mathrm{Var}(\hat f(x))+\mathrm{Var}(\varepsilon\mid X=x).
$$

应共同考察偏差、方差与误差相关性。如果所有模型使用同一泄漏特征，平均只会让错误结论显得更稳定。

### 2.2 Bagging、Random Forest、boosting

| 方法 | 构造方式 | 统计直觉 | 金融研究注意点 |
| --- | --- | --- | --- |
| Bagging | 重抽样、分别拟合，再平均或投票 | 平滑不稳定学习器，主要降低方差 | 行重抽不等于信息重抽；OOB 可能污染 |
| Random Forest | 通常结合 bootstrap 与每个节点的随机候选特征 | 减少强特征让所有树长得相似 | 去相关有偏差代价；输出概率仍需检验 |
| Boosting | 按序增加修正项，解释前序模型的不足 | 逐步优化损失、提高表达能力 | 可能追逐噪声；需正则化与正确的停止规则 |

AdaBoost 提高错分样本权重；gradient boosting 拟合损失的负梯度。不能把所有 boosting 都解释为同一种“错分加权”。通用加性形式为：

$$
F_M(x)=F_0(x)+\eta\sum_{m=1}^{M}\gamma_m h_m(x).
$$

学习率  $\eta$ 、树深、迭代数、子采样率和 early stopping 共同控制复杂度。AFML 对 bagging 的偏好可理解为面对低信噪比、异常值与重叠标签时的稳健倾向；它不证明 boosting 在金融中必然更差。比较应使用相同信息边界和计算预算。

集成也不是自动校准器。投票比例或平均概率是否对应真实频率，仍需独立预测检验；校准器自身也必须在训练侧拟合。

### 2.3 金融样本冗余：不同的行可能是同一份证据

事件  $i$  的标签依赖区间  $I_i=[t_{i,0},t_{i,1}]$  内的价格路径。不同事件会共享收益增量。在离散行情时点上定义：

$$
a_{t,i}=\mathbf{1}\lbrace t\in I_i\rbrace ,\qquad c_t=\sum_i a_{t,i}.
$$

事件平均 uniqueness 为：

$$
\bar u_i=\frac{1}{\lvert I_i\rvert}\sum_{t\in I_i}\frac{1}{c_t}.
$$

这里  $\lvert I_i\rvert$  表示离散活跃时点数，活跃时  $c_t$  必为正。若两个标签全程完全重叠，各自 uniqueness 为  $1/2$ ；若事件独占其路径，则为 1。

它衡量价格路径占用的冗余程度，不是预测能力，也不是严格的有效样本量。

Sequential bootstrap 每抽入一个事件，就根据已选事件重新计算候选事件的边际 uniqueness。设已选多重集合的并发数为  $c_t^{(S)}$ ，候选评分及抽样概率可写为：

$$
\tilde u_i(S)=\frac{1}{\lvert I_i\rvert}\sum_{t\in I_i}\frac{1}{c_t^{(S)}+1},
\qquad
\Pr(i\text{ is next}\mid S)=\frac{\tilde u_i(S)}{\sum_j\tilde u_j(S)}.
$$

已抽事件仍可再次被抽中，但重复占用路径降低其后续吸引力。

**准确定位：它倾向于提高单个重抽样本内的信息独特性，并不直接最小化树与树之间的误差相关性。** 是否改善最终集成需要实证检验。它不保证 IID，不消除共同市场冲击，也不处理制度变化。

某行未进入某棵树，不代表该行标签使用的行情未进入训练。因此普通 OOB 分数不能自动替代金融 OOS；样本重抽样解决不了验证设计问题。

## 3. Chapter 7：OOS 是信息边界，不只是行号边界

### 3.1 Filtration：决策只能使用当时可得信息

令  $\mathcal F_t$  表示时刻  $t$  已可获得的信息，满足对  $s\le t$  有  $\mathcal F_s\subseteq\mathcal F_t$ 。时刻  $t$  的特征、已训练模型与仓位，都应对  $\mathcal F_t$  可测：

$$
q_t=g(\hat f_t(X_t)).
$$

这个要求覆盖模型参数、预处理和选择规则，而不仅是特征列没有显式未来收益。

标签结束时间与标签可用时间还应区分。若  $a_i^Y$  为标签实际可取得的时点，历史训练数据必须满足：

$$
D_t=\lbrace (X_i,Y_i):a_i^Y\le t\rbrace .
$$

财报期末日期不等于公告日期；宏观数据初值不等于后来修订值；供应商时间戳不等于本地到达时间。还应保存退市证券、历史指数成分、暂停交易、复权和数据版本。

**Point-in-time 是数据工程契约，不能靠 CV 补救。** 原始记录属于过去，不代表今天下载到的版本当时就存在。

### 3.2 Purging：删掉与测试标签共享结果路径的训练事件

设测试事件集合为  $J$ ，覆盖的结果区间为：

$$
\mathcal I_J=\bigcup_{j\in J}[t_{j,0},t_{j,1}].
$$

若训练候选事件  $i$  满足：

$$
[t_{i,0},t_{i,1}]\cap\mathcal I_J\ne\varnothing,
$$

则删除它。实现时要明确端点是否共享同一收益增量；这里采用保守的闭区间记法。

例：测试事件从第 10 日到第 15 日；训练事件从第 8 日到第 12 日。后者虽然开始得早，标签却包含测试期价格，应被 purge。只按事件开始时间切分仍会泄漏。

Purging 只修复已建模的区间重叠，无法修复全样本标准化、未来修订值或错误交易时间。非连续测试区间应按并集处理；用总包络区间会保守地删除更多数据。

### 3.3 Embargo：为边界附近的信息依赖留缓冲

AFML 的 embargo 通常排除测试区段之后的一段训练事件。一个明确的保守设计是令  $T_J=\max_{j\in J}t_{j,1}$ ，排除：

$$
T_J\lt t_{i,0}\le T_J+h.
$$

这是本文的描述约定；实际代码可能按测试索引、bar 数或比例实现，必须核对锚点和单位。

例如测试结果覆盖到第 15 日，第 16 日训练特征回看第 12—16 日行情，即使标签不重叠，仍可能间接利用测试期信息。缓冲长度应结合特征回看、数据延迟、标签跨度与部署方式。

固定百分比是 heuristic，不是“设 1% 就独立”的定理。长记忆过程也不会因为任意有限缓冲就变成独立样本。

**共享当时已经公开的过去信息，不一定是前视泄漏；但它仍可能导致评估误差相关。** 泄漏控制与不确定性估计应分别讨论。

### 3.4 Purged K-fold 与 walk-forward 的估计对象不同

| 方案 | 训练能否位于测试期之后？ | 主要问题 | 局限 |
| --- | --- | --- | --- |
| Purged K-fold | 可以 | 在跨期稳定性假设下，映射能否跨区块复用？ | 不等于历史可执行训练流程 |
| Expanding walk-forward | 不可以 | 历史逐步积累时如何部署？ | 旧制度数据可能妨碍适应 |
| Rolling walk-forward | 不可以 | 只用最近窗口如何部署？ | 数据较少，窗口也是选择参数 |

Purging 不会让“使用未来时期训练”变成真实可交易。若模拟历史上线，外层应遵守时间顺序，在每个截止点排除尚未成熟的标签。

纯 walk-forward 没有测试之后的训练集，AFML 式右侧 embargo 可能没有作用对象；训练末端的标签成熟检查和必要间隔仍不可省略。

不同折还可能对应不同市场制度。折间标准差不是天然独立的标准误，不能机械除以折数平方根后宣称显著。

### 3.5 先切分，再学习所有数据驱动步骤

标准化、缺失值填补、PCA、特征筛选、权重规则、概率校准、阈值、early stopping、标签参数与训练窗口，都可能参与模型选择。

即使 PCA 不使用标签，全样本 PCA 也会让历史模型知道未来协方差结构。若部署允许某种实时横截面信息，应明确哪些证券数据在同一时刻可得；既不一概禁止，也不一概放行。

Pipeline 能组织这些步骤，但不会自动理解事件区间、公告延迟或研究者的试验历史。

## 4. Chapter 8：特征重要性是带条件的问题

### 4.1 MDI、MDA、SFI 分别在问什么？

设  $S$  为越大越好的评分；使用损失时先统一正负方向。

| 方法 | 问题 | 计算 | 主要陷阱 |
| --- | --- | --- | --- |
| MDI | 树训练时多大程度使用了变量？ | 累计加权不纯度下降 | 训练内偏差、候选切分数、替代效应 |
| MDA | 固定模型对变量扰动多敏感？ | 验证集置换前后评分差 | 破坏相关结构、模型依赖、置换设计 |
| SFI | 单独使用变量能预测多少？ | 单变量训练并交叉验证 | 看不到纯交互，不是条件增量价值 |

MDI 的一个未归一化表达为：

$$
I_j^{\mathrm{MDI}}=\frac{1}{B}\sum_{b=1}^{B}
\sum_{v:\quad \mathrm{split}(v)=j}
\frac{W_v}{W_{\mathrm{root},b}}
\left(H_v-\frac{W_L}{W_v}H_L-\frac{W_R}{W_v}H_R\right).
$$

 $W$  是节点训练权重总量， $H$  是不纯度。常见实现会再归一化。可选切分很多时，噪声变量也更容易改善训练不纯度，高 MDI 因而不等于高 OOS 贡献。

对第  $k$  个验证折，固定模型  $\hat f^{(-k)}$ ，置换第  $j$  列：

$$
I_{j,k}^{\mathrm{MDA}}=S(\hat f^{(-k)},D_k)-S(\hat f^{(-k)},D_k^{\pi_j}).
$$

名字中虽有 accuracy，MDA 也可使用负 log-loss 等评分。AFML 还讨论归一化差值；实务要明确分母，避免基准接近零时产生不稳定比值。

**MDA 不严格等于删除该特征后重新训练的增量价值。** 前者固定模型并干预输入，后者改变学习问题。相关变量下，两者可能给出不同结论。参见官方 [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance)。

SFI 可以采用相对基准的表达：

$$
I_j^{\mathrm{SFI}}=\frac{1}{K}\sum_{k=1}^{K}
\left[S(\hat f_j^{(-k)},D_{k,j})-S(\hat f_0^{(-k)},D_k)\right].
$$

 $\hat f_j$  只用第  $j$  列， $\hat f_0$  为训练侧拟合的基准。书中也可直接报告单特征 CV 分数，不应把本文的基准差值当作唯一官方定义。

### 4.2 Substitution effect 与 interaction effect

若两个变量复制同一信号，树可能只选择其中一个，或把使用次数分摊。置换一列时，另一列可能提供替代信息，使单列 MDA 很小。

但“相关变量 MDA 必然偏低”也过强。无条件置换可能制造训练分布中不存在的组合，迫使模型外推，从而夸大重要性。结果取决于模型使用相关性的方式。

补充诊断包括：按经济含义联合置换特征组、训练侧相关聚类、删除后重训、在合理条件分布内置换。分组与条件置换回答不同问题，各有额外假设。

SFI 避免同一次多特征拟合中的替代竞争，却看不到纯交互。若独立均衡的二元变量满足：

$$
Y=X_1\oplus X_2,
$$

单看任一变量都无法改善随机猜测，联合观察却可完全确定标签。因此不应把所有低 SFI 变量直接删除。

时间序列逐行置换会破坏自相关；区块置换、同时间横截面置换保留的结构不同。重复置换的波动主要反映置换随机性，不是自动有效的经济显著性检验。

### 4.3 PCA / orthogonal features：换坐标不等于发现信号

在训练折内标准化并中心化，设矩阵为  $X$ ：

$$
\hat\Sigma=\frac{X^\top X}{n-1}=V\Lambda V^\top,\qquad Z=XV.
$$

于是训练样本上：

$$
\frac{Z^\top Z}{n-1}=\Lambda.
$$

主成分线性不相关，但通常不独立，也不保证未来仍正交。它减少线性共线性带来的替代竞争，代价是解释对象变成了原始变量的线性组合。

PCA 最大化解释输入方差，不最大化预测标签的能力。高方差  $X_1$  可能与标签无关，低方差  $X_2$  却决定标签；只保留第一主成分可能丢掉全部信号。

AFML 比较主成分方差排名与监督重要性排名，可作为结构诊断。一致不证明真实 alpha，不一致也不自动说明模型无效。保留成分数与变换均应在训练侧选择。

### 4.4 Parallelized、stacked 与 partial pooling

这里讨论跨工具、跨市场的数据组织。AFML 的 stacked feature importance 不应与“多个模型预测再训练二层模型”的 stacking 混淆。

| 方式 | 做法 | 取舍 |
| --- | --- | --- |
| Parallelized | 每个工具分别拟合、计算重要性，再比较汇总 | 保留异质性，单工具估计可能不稳定 |
| Stacked / pooled | 对齐特征定义，纵向合并各工具样本训练 | 利用共同结构，可能掩盖市场差异 |
| Partial pooling | 共享总体部分，同时允许工具偏离 | 在共同性和异质性之间收缩 |

Partial pooling 是本文的统计扩展，不是 AFML 在该节完整提出的层级模型。例如：

$$
Y_{a,t}=X_{a,t}^\top(\beta+u_a)+\varepsilon_{a,t}.
$$

可用惩罚估计资产偏离：

$$
\min_{\beta,\lbrace u_a\rbrace }
\left\lbrace \sum_{a,t}\ell\left(Y_{a,t},X_{a,t}^\top(\beta+u_a)\right)
+\lambda\sum_a\lVert u_a\rVert_2^2\right\rbrace .
$$

较大  $\lambda$  倾向共享映射，较小  $\lambda$  允许更大差异。端点的可识别性取决于约束和损失，不能把这个例式当作所有 pooling 方法的定义。

合并前应统一标签语义、量纲、预测跨度和权重，否则大样本或高波动工具可能主导结论。多个资产共享市场冲击，不代表增加了同等数量的独立证据。

预测已知工具的未来，应按时间联合切分相关资产；预测未见工具，还需留工具或市场验证。随机拆散面板行容易把同一冲击放入训练和测试。不同模型的重要性若尺度不同，也不能直接平均。

### 4.5 Feature importance 也会产生 research overfitting

反复看重要性、换特征、再看 CV 分数，仍在搜索同一历史。图表更容易解释，不代表研究过程天然诚实。

合理用途是提出机制假说与诊断：信号是否集中在一个制度？是否依赖可疑时间戳？是否仅由单一市场驱动？随后以独立时期、工具或事先规定的外层流程检验。

预测重要性不是因果效应。变量可能只是代理或伴随现象；从“模型依赖它”到“改变它会改变收益”，还缺少识别假设。

## 5. Chapter 9：调参是有限预算下的实验设计

### 5.1 Grid Search 与 Random Search

若有  $d$  个参数，第  $j$  个取  $m_j$  个值，网格候选数为：

$$
M_{\mathrm{grid}}=\prod_{j=1}^{d}m_j.
$$

Random search 固定试验数  $M$ ，从预设分布抽参数。若只有少数维度真正重要，它能减少无关维度上的重复网格探索。这是效率优势，不是防过拟合保证。[Bergstra 与 Bengio，2012](https://www.jmlr.org/papers/v13/bergstra12a.html)

若可接受区域在抽样分布下的概率为  $p$ ，独立抽  $M$  次至少命中一次的概率为：

$$
1-(1-p)^M.
$$

这说明搜索范围与分布的重要性，不保证验证评分能认出真正的好参数。总成本还要乘折数、拟合和预处理成本，不能把计算量仅写成与训练成本无关的  $O(M)$ 。

Grid 适合少数离散且有清楚含义的选择；random search 适合较大连续空间。自适应搜索也无法修复泄漏，甚至可能更有效地利用错误评分器。

### 5.2 Log-uniform：对数量级均匀分配机会

设  $0\lt a\lt b$ ：

$$
U\sim\mathrm{Uniform}(\log a,\log b),\qquad X=e^U.
$$

变量变换得到：

$$
f_X(x)=\frac{1}{x\log(b/a)},\qquad a\le x\le b.
$$

区间外密度为零。在支持区间内：

$$
F_X(x)=\frac{\log(x/a)}{\log(b/a)}.
$$

只要  $[c,kc]\subseteq[a,b]$  且  $k\gt 1$ ：

$$
\Pr(c\le X\le kc)=\frac{\log k}{\log(b/a)}.
$$

相同倍数区间有相同概率。在  $[10^{-3},10^3]$  上六个十倍区间各有  $1/6$  概率；线性均匀抽样落入小于 1 的区域只有约 0.1%。

适用于严格正且比例变化有意义的参数，如正则化强度、SVM 的  $C$ 、核尺度和学习率。0 无法取对数，应作为特殊候选单独处理；树深和类别选项不能机械照搬。

它是常用尺度型搜索分布，不是 AFML 独有的普遍最优先验。现代 SciPy 已提供 [scipy.stats.loguniform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loguniform.html)，无需复制旧自定义分布类。

### 5.3 Winner’s curse：最高分包含被选中的噪声

候选  $m$  的评分写为：

$$
\hat S_m=S_m+\varepsilon_m,\qquad \hat m=\arg\max_m\hat S_m.
$$

各候选评分误差均值为零，不意味着被选中候选的误差均值也为零。真实表现相同、误差独立且为方差  $\sigma^2$  的高斯变量时，大  $M$  的典型量级为：

$$
\mathbb E\left[\max_{1\le m\le M}\varepsilon_m\right]\approx\sigma\sqrt{2\log M}.
$$

这只是说明乐观偏差的近似，不是可直接套在相关 CV 候选上的校正公式。真实表现不等、候选相关、搜索自适应时，需要更细分析。

试验数包括手工删特征、换标签、换窗口、改成本和重试种子，不只是搜索器日志里的参数数目。

### 5.4 Nested CV：评估整个选择算法

外层第  $k$  折先锁定训练和测试，仅在外层训练内部选择：

$$
\hat\theta_k=\arg\max_{\theta\in\Theta}\widehat S_{\mathrm{inner},k}(\theta).
$$

再重训并评估：

$$
\hat f_k=\mathrm{Fit}(D_{\mathrm{train},k};\hat\theta_k),
\qquad
\hat S_{\mathrm{outer}}=\frac{1}{K}\sum_{k=1}^{K}S(\hat f_k,D_{\mathrm{test},k}).
$$

平均式假设等权汇总；折大小、时期或经济目标不同时，应提前定义权重。外层评价“选择并拟合”的算法，不是为一个固定参数提供无条件的未来保证。[Nested versus non-nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)

例如，用 2018—2022 年作为外层训练、2023 年测试；只在前一段内部用历史到未来的切分选择树深，然后在全部合格训练标签上重训，最后评估 2023。下一外层折可重新调参，参数不同是正常现象。

内层应包括筛选、PCA、阈值、校准和停止规则；内外层都必须尊重信息边界。AFML 的调参函数只是内层搜索例程，搜索后重训不自动形成 nested CV。若最终再套 bagging，外层也必须评估这个最终过程。

看完外层结果又修改研究流程，外层便参与了下一轮选择。Nested CV 能隔离预先定义的程序，不能隔离研究者无限次事后修改。

### 5.5 Accuracy、F1、log-loss 与经济目标

| 评分 | 衡量什么 | 适合情形 | 遗漏内容 |
| --- | --- | --- | --- |
| Accuracy | 离散分类是否正确 | 类别成本接近、分类本身是目标 | 概率、收益幅度、不对称成本 |
| F1 | 正类 precision 与 recall 的协调 | 筛选正类、关注漏选与误选 | 真负类、校准、收益和成本 |
| Log-loss | 对真实标签分配的概率 | 概率参与决策 | 仓位规则与完整经济效用 |
| 净收益、效用、风险调整收益 | 决策经济结果 | 政策及成本明确 | 估计噪声大、路径依赖、选择偏差 |

F1 为：

$$
F_1=\frac{2TP}{2TP+FP+FN}.
$$

它不是“类别不平衡时总是最优”的定理。二元标签也不自动意味着 meta-labeling。AFML 在 meta-labeling 中选择 F1 是应用选择；若仓位依赖概率，仍需检验概率质量。

二分类 log-loss：

$$
L=-\frac{1}{N}\sum_{i=1}^{N}
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].
$$

两个真实标签分别为 1 和 0，预测都为正类：两个概率均为 0.51 时，平均损失约 0.69335；均为 0.99 时，约 2.30761。Accuracy 都是 50%，后者却对错误方向极度自信。搜索器若统一最大化评分，便使用负 log-loss。

设真实条件概率为  $q$ ，报告概率为  $p$ ：

$$
\mathbb E[\ell(Y,p)\mid X]
=-q\log p-(1-q)\log(1-p)
=H(q)+D_{\mathrm{KL}}(\mathrm{Bern}(q)\Vert\mathrm{Bern}(p)).
$$

KL 散度非负，期望损失在  $p=q$  时最小。这是 log-loss 作为严格 proper scoring rule 的理由；它关心概率质量，不只是校准误差。

加权版本为：

$$
L_w=-\frac{\sum_iw_i\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]}{\sum_iw_i}.
$$

权重可表达经济重要性，但会改变目标。当权重依赖标签或实现收益时，最优概率一般为：

$$
p_w^*(x)=\frac{\mathbb E[wY\mid X=x]}{\mathbb E[w\mid X=x]}.
$$

推导很直接：令  $A=\mathbb E[wY\mid X=x]$ 、 $B=\mathbb E[w(1-Y)\mid X=x]$ ，条件风险为  $-A\log p-B\log(1-p)$ ；令导数为零，得到  $p=A/(A+B)$ 。

它不一定等于原始胜率。若权重只是正的  $X$  的函数，条件最优概率才仍等于  $q(x)$ 。收益加权模型的输出不能未经检验就被解释为原始条件概率。

### 5.6 Weighted log-loss 为什么不等于 PnL？

一个单期简化组合模型为：

$$
R_{t+1}^{\mathrm{net}}=q_t r_{t+1}-c_t\lvert q_t-q_{t-1}\rvert.
$$

相同标签和概率可对应不同收益幅度、重叠持仓、冲击成本与风险预算。单笔事件加权损失没有自动构建资金可行的组合路径。

若做对赚  $G$ 、做错亏  $L$ 、每次成本为  $c$ ：

$$
pG-(1-p)L-c\gt 0
\quad\Longleftrightarrow\quad
p\gt \frac{L+c}{G+L}.
$$

0.5 不是通用交易阈值；概率也不是仓位大小的完整充分统计量。赔率、相关风险、杠杆与成本都重要。

可先用稳定的概率评分做有限调参，再用锁定的政策评价独立净收益、换手和尾部风险。也可直接优化效用，但需要足够样本和更严格的选择控制，不存在永远最佳的单一评分。

## 6. 统一统计哲学：四个层次的“重复使用证据”

1. **样本层：** 标签重复占用行情。Uniqueness 与 sequential bootstrap 调整其贡献。
2. **模型层：** 学习器重复同一种误差。集成需要控制共同误差，而非只增加数量。
3. **验证层：** 训练间接见过测试信息。Purging、point-in-time 与训练侧预处理维护边界。
4. **研究层：** 多次试验反复消费同一历史。Nested CV、最终保留集与研究记录维护选择和估计的分离。

不同层次的“重复”并不等价。去除标签重叠不等于去除误差相关；去相关不等于消除选择偏差；留下测试集不等于研究者没从它学习。

与前四章连接，整个系统可看作：

$$
\mathcal A:D_{\mathrm{past}}\longmapsto
(\text{sampling},\text{labels},\text{weights},\text{features},\hat f,\text{policy}).
$$

最终要检验的是这个完整系统在新时期的表现。固定分类器却不断修改 bars、屏障、差分阶数、特征名单和搜索范围，仍是在选择  $\mathcal A$ 。

**好研究不是让历史分数最大，而是让关于未来的判断与证据强度匹配。** 应同时报告表现、不确定性、适用制度与失败方式。

## 7. 量化实务：可以落实的最小流程

### 7.1 一个连贯的时间例子

假设每日收盘后预测未来最多 5 个交易日的事件结果，用最近 20 日构造特征，每月重训。

1. 固定信号时间、成交延迟、退出规则与股票池历史口径。
2. 月末只使用标签已结束且已可取得的训练事件，不能因为开始得早就纳入尚未完成的结果。
3. 在外层训练期间做时间有效的内层验证，比较有限模型与参数。
4. 每个内层训练折重新拟合标准化、PCA、筛选及校准，按实际事件区间处理重叠。
5. 重训最终完整流程，预测下一月，按预定成本和成交规则生成组合路径。
6. 时间推进后，旧测试期标签成熟可以进入后续训练，这符合真实部署；但不能反复修改已经宣称独立评估的研究规则。

20 日回看不意味着必须盲目设置 20 日 embargo。应检查训练在测试之前还是之后、信息何时可得，以及依赖如何影响误差估计。

### 7.2 实施优先级

| 优先级 | 工作 | 验收标准 |
| --- | --- | --- |
| 首先 | Point-in-time、标签成熟、交易时间对齐 | 能重建每个决策时点的信息 |
| 其次 | 简单基准与外层时间验证 | 完整流程有可复现 OOS 预测 |
| 再次 | 稳定性、特征组诊断、有限调参 | 表现不由单折、单工具或少数交易独占 |
| 随后 | 精细重抽样、pooling、复杂搜索 | 同一验证预算下存在重复增益 |

计算预算有限时可减少候选数、模型种类和无根据的搜索范围，不应把最终测试集改作日常调参。并行化也要避免内外层、搜索与集成同时占满所有线程。

### 7.3 仍然无法保证什么？

- **少量独立机会：** 很多交易不等于很多独立证据；更诚实的验证可能只让置信区间变宽。
- **制度变化：** 历史多折稳定不证明未来稳定；rolling 与 expanding 的取舍本身也需验证。
- **成本与容量：** 预测优势可被换手、冲击、融资和可成交规模抵消。
- **尾部风险：** 平均损失优秀不保证极端情景安全，有限历史无法穷尽压力状态。
- **有效推断：** 折、资产和事件有依赖；区块重抽样或分组推断也有适用假设。
- **研究治理：** Holdout 被用于修改流程后就失去原先角色；需保存失败试验、数据版本、搜索空间和评分规则。

## 8. Enduring principles、heuristics 与过时实现

| 类型 | 值得保留的内容 | 应避免的过强解释 |
| --- | --- | --- |
| 长期原则 | 决策只能使用当时可得信息 | 时间戳正确就代表版本正确 |
| 长期原则 | 选择与最终估计分离 | Nested CV 自动消除研究者事后修改 |
| 长期原则 | 集成依赖协方差结构 | 模型更多就有更多独立证据 |
| 长期原则 | 评分服务最终决策 | 所有金融任务都应用同一个评分 |
| AFML heuristic | Uniqueness、sequential bootstrap | 重抽后恢复 IID |
| AFML heuristic | 固定比例 embargo | 任意比例都可确保独立 |
| 应用偏好 | Bagging、meta-label 使用 F1 | 不允许任务及实证推翻默认偏好 |
| 结构诊断 | PCA 排名与重要性比较 | 一致就证明真实 alpha 或因果机制 |

截至整理日期，现代实现需要注意：

- [BaggingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) 当前使用 estimator 参数；书中的 base_estimator 是旧接口。
- [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) 当前不再接受书中的 iid=False。更新 API 不等于研究设计已经正确。
- SciPy 已直接提供 loguniform，无需重写旧自定义分布。
- 传入训练 sample_weight，不代表评分自动收到相同权重；必须核对版本、scorer、metadata routing 及每折样本对齐。
- 标签集合恰为 0 和 1，不足以判定是 meta-labeling，任务语义应显式配置。
- Pipeline 不会自动执行 purging，更不会记录研究者在历史结果上做了多少次手工试验。

## 9. 复习时应能回答的六个问题

1. 增加树数降低哪一项方差，什么共同误差不会消失？
2. 某行是 OOB，为什么仍可能含有训练见过的结果信息？
3. Purged K-fold 高分为什么不等于可执行的 walk-forward 收益？
4. MDA、删除后重训、SFI 分别改变什么，为什么不能互相替代？
5. 收益加权 log-loss 学到的概率是否仍是原始胜率？
6. 看完 outer CV 又改特征，哪一层独立性已被消耗？

能解释这些问题，比记住旧版 API 更接近四章的长期价值。

## 10. 来源与阅读边界

- 基础内容：Marcos López de Prado，Advances in Financial Machine Learning，Wiley，2018，第 6—9 章；第 4 章用于连接标签重叠与重抽样。本文是对话基础上的重组分析，不是逐页翻译。
- [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/v13/bergstra12a.html)：随机搜索的原始研究。
- [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance)：现代实现的解释边界。
- [Nested versus non-nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)：区分模型选择与评估。
- [SciPy loguniform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loguniform.html)：分布及现有接口。
- [GitHub 数学表达式](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)：采用行内美元符号和独立块级双美元符号，不使用自定义宏、公式编号或复杂环境。

Filtration 的统一叙述、partial pooling 示例、加权概率目标推导、实施优先级与研究治理建议属于本文扩展，不代表书中逐字主张或已验证的交易收益。
