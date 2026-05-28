# 🧬 Bonnie-Lab

> *A digital space for evolving thoughts and cross-disciplinary insights.*

欢迎来到我的数字实验室。这里是我系统化沉淀知识的基地，涵盖了从底层数学逻辑到前沿 AI 应用的全栈学习笔记，重点关注 **交通 (ITS)** 与 **金融 (Finance)** 的交叉演进。

---

## 🏗️ 1. 基础科学 (Fundamental Science)
支撑复杂系统建模的底层工具与数学框架。

### 📐 数学基础 (Mathematics)
*   [**算子 (Operator)**](./Math/operator.md)
    *   系统化梳理算子作为处理“过程”与“映射关系”的超级规则，解析微分、积分及线性算子在空间变换中的作用，探讨算子在深度学习框架中作为“原子积木”的解构思想，并详述邻近算子在非光滑优化中作为平滑预处理工具的核心机制。
*   [**梯度算子 (Gradient Operator)**](./Math/gradient_operator.md)
    *   系统化梳理作为向量微分算子的梯度本质，解析其在标量场中指向函数增长最快方向的几何法则，探讨梯度算子与等值面（等高线）的正交特性，并深度阐述其在 AI 损失函数优化中的导航作用及在物理场（如热流、电势）中的负梯度反馈意义。
*   [**邻近算子 (Proximal Operator)**](./Math/proximal_operator.md)
    *   详述非光滑复合优化的核心桥梁，涵盖近端映射数学定义、常见闭式解（软阈值/投影）及在大规模分布式控制中的应用。
*   [**不动点算子 (Fixed-Point Operator)**](./Math/fixed_point.md)
    *   探讨如何将复杂的耦合系统（如 MFG）转化为算子求平衡点问题，并分析其收敛性。
*   [**维纳过程 (Wiener Process)**](./Math/wiener_process.md)
    *   解析连续时间随机过程的数学定义，厘清物理现象（布朗运动）与理想化模型之间的映射关系。
*   [**随机扰动 (Stochastic Perturbation)**](./Math/stochastic_perturbation.md)
    *   探讨确定性系统在不确定因素干预下的演化规律，以及随机微分方程的建模基础。
*   [**空间离散化 (Spatial Discretization)**](./Math/spatial_discretization.md)
    *   探讨如何将连续物理空间转化为计算机可处理的有限单元，及其在交通流仿真与路径规划中的实现。
*   [**Wasserstein 距离 (Optimal Transport)**](./Math/wasserstein_distance.md)
    *   深度解析“推土机距离”的数学原理，探讨其在 WGAN 优化及概率分布对齐中的几何优势。
*   [**Lipschitz 连续 (Lipschitz Continuity)**](./Math/lipschitz_continuity.md)
    *   探讨函数变化率的有界性定义及其在 WGAN 稳定性、微分方程求解中的核心支撑作用。
*   [**Hamilton-Jacobi-Bellman (HJB) 方程**](./Math/hjb_equation.md)
    *   详细解析连续时间最优控制的充分必要条件，涵盖价值函数推导、随机扩散项修正及动力学耦合机制。
*   [**Fokker-Planck (FPK) 方程**](./Math/fokker_planck_equation.md)
    *   详细解析概率密度函数的演化规律，涵盖漂移-扩散机制、质量守恒定律及其在 MFG 中的正向演化作用。
*   [**邻近点算法 (Proximal Point Algorithm)**](./Math/proximal_point_algorithm.md)
    *   探讨隐式凸优化的鼻祖算法，解析近端正则化平滑机制、隐式迭代的鲁棒收敛性以及向现代变分不等式、ADMM 算法的演化逻辑。
*   [**Nesterov 加速梯度方法（NAG）**](./Math/nesterov_accelerated_methods.md)
    *   深度系统化梳理 Nesterov 第一、第二、第三加速方法的数学架构、适用场景以及在对偶与复合优化中的演化关系。
*   [**次梯度方法 (Subgradient Method)**](./Math/subgradient_method.md)
    *   解析非光滑凸优化的核心工具，涵盖次微分数学定义、非下降方向特性及动态步长收敛策略。
*   [**加速邻近梯度法 (Accelerated Proximal Gradient Method)**](./Math/accelerated_proximal_gradient.md)
    *   解析 Nesterov 动量外推机制，剖析 FISTA 算法架构，探讨复合凸优化中 $\mathcal{O}(1/k^2)$ 极限收敛速率的数学证明。
*   [**邻近梯度法 (Proximal Gradient Method)**](./Math/proximal_gradient_method.md)
    *   详述前向-后向分裂架构，解析复合凸优化中光滑梯度与非光滑近端映射的耦合机制，探讨 FISTA 加速原理。
*   [**坐标下降法 (Coordinate Descent Method)**](./Math/coordinate_descent_method.md)
    *   系统化梳理坐标下降法的高维分而治之机制、算法递推流程，深度剖析非凸不收敛经典反例（Powell 震荡），以及坐标极小点在连续可微与非光滑凸函数场景下的收敛性定理。
*   [**牛顿法 (Newton's Method)**](./Math/newtons_method.md)
    *   系统化梳理基于二阶泰勒展开的经典牛顿法推导与迭代流程，深度剖析其在最优解邻域内的局部二次收敛性定理，并针对经典牛顿法的局限性，详述阻尼牛顿法、混合算法以及拟牛顿法（BFGS/L-BFGS）三大现代改进变体。
*   [**拟牛顿法 (Quasi-Newton Method)**](./Math/quasi_newtons_method.md)
    *   系统化梳理旨在免除海瑟矩阵显式计算的拟牛顿思想与通用迭代流程，深度剖析割线方程这一核心数学本质，并详述 DFP、BFGS 以及适用于大规模高维特征存储受限场景的 L-BFGS 等经典低秩修正算法与超线性收敛性质。
*   [**海瑟矩阵 (Hessian Matrix)**](./Math/hessian_matrix.md)
    *   深入阐述多元函数二阶偏导数方阵的数学定义与几何物理意义，系统解析广义不等式中半正定算子 $\succeq 0$ 的空间多维曲率本质，并详述其在判定凸函数全局最优性以及保障对偶函数天然凹性与全局收敛性中的核心应用场景。
*   [**凸优化符号解析：半正定算子(Succeeds or equals)的本质**](./Math/convex_optimization_symbols.md)
    *   深度辨析标量一维比较运算符 $\ge$ 与矩阵凸锥偏序运算符 $\succeq$ 的本质区别，解析二次型不等式下多维空间全方位曲率判定的几何物理意义，并阐明引入广义不等式以规避矩阵元素级比较歧义的数学严谨性。

<!--
*   **[优化理论与邻近算子 (Proximal Operators)](./Mathematics/proximal_algorithms.md)**
    *   探讨非光滑优化、复合函数优化及在大规模数据处理中的应用。
*   **[现代组合投资理论](./Mathematics/markowitz_model.md)**
    *   **Markowitz 均值-方差模型**：资产配置的核心逻辑，及其在路网容量分配中的类比应用。
*   **[矩阵分析](./Mathematics/matrix_analysis.md)**：线性代数的高级进阶与算子理论。
-->

### 🎲 博弈论 (Game Theory)
*   [**纳什均衡 (Nash Equilibrium)**](./Game/Nash_Equilibrium.md)
    *   探讨在非合作博弈中，各参与者在对手决策不变时均无动机单方面改变策略的稳定状态。
*   [**斯塔克尔伯格博弈 (Stackelberg Game)**](./Game/stackelberg_game.md)
    *   解析具有先后决策顺序的层级化博弈，探讨领导者通过先行决策引导跟随者行为的机制。
<!--
*   **[非合作博弈与纳什均衡](./Game_Theory/nash_equilibrium.md)**
*   **[平均场博弈 (MFG)](./Game_Theory/mean_field_games.md)**：研究大规模异构智能体群体的交互行为。
-->

---

## 🤖 2. 人工智能核心 (AI Core)
从感知、推理到技能习得的智能体技术栈。

### 🧠 AI 基础理论
*   [**蒙特卡洛树搜索 (Monte Carlo Tree Search)**](./AI_core/mcts_decision_making.md) 
    *   解析 AI 决策规划的核心框架：详述选择、扩展、模拟、反向传播四大环节，分析其在动态环境下如何通过随机模拟逼近最优策略，以及在 AlphaGo 等复杂博弈系统中的关键地位。 
*   [**置信上限树算法 (UCT, Upper Confidence Bound for Trees)**](./AI_core/uct_decision_balance.md) 
    *   剖析 MCTS 的搜索决策策略：探讨 UCT 公式如何在“利用（Exploitation）”已发现优势与“探索（Exploration）”未知空间之间建立数学平衡，解决复杂决策过程中的路径选择难题。
<!--
*   **[深度学习 (Deep Learning)](./AI_Core/deep_learning.md)**：Transformer 架构、特征表达与神经网络稳定性。
*   **[强化学习 (Reinforcement Learning)](./AI_Core/reinforcement_learning.md)**：从 MDP 到 PPO，探讨智能体在复杂环境中的最优决策。
*   **[技能习得与知识迁移 (Skills & Transfer)](./AI_Core/skill_learning.md)**：研究 Agent 如何获取原子技能 (Atomic Skills) 并进行长程任务规划。
-->

---

## 🚦 3. AI + 交通 (AI & Traffic)
*   [**异构交通环境 (Heterogeneous Traffic Environment)**](./AI_Traffic/heterogeneous_traffic_environment.md)
    *   解析混合自动化背景下，不同智能等级与驾驶风格的冲突与共存。
*   [**影响场理论 (Influence Field Theory)**](./AI_Traffic/influence_field.md)
    *   探讨车辆在物理空间中的虚拟排斥场及其对微观决策的影响。
*   [**Frenet 坐标系 (Frenet Frame)**](./AI_Traffic/Frenet_coordinates.md)
    *   通过将弯曲道路“拉直”为纵向与侧向维度，简化自动驾驶中的路径规划与运动决策。
*   [**地板油与动力饱和 (Full Throttle & Saturation Control)**](./AI_Traffic/full_throttle.md)
    *   解析极端加速行为在自动驾驶纵向规划中的数学约束及其对多智能体博弈的影响。

<!--*   [**跨尺度仿真 (L2S2D)**](./AI_Traffic/l2s2d_framework.md)：连接微观博弈与宏观分布的钥匙。-->

---

## 💰 4. AI + 金融 (AI & Finance)
*   [**金融风险深度解析 (Financial Risk)**](./AI_Finance/financial_risk.md)
    *   系统化定义市场、信用、流动性及操作风险，并提供应对策略框架。
*   [**金融策略概论 (Financial Strategy)**](./AI_Finance/financial_strategy.md)
    *   从权益投资到 AI 驱动的智能量化，解析策略构建的核心逻辑。
*   [**量化金融基础 (Quant Finance)**](./AI_Finance/what_is_quantitative_finance.md)
    *   数据驱动的投资范式，以及从统计模型向强化学习的演进。


---
**Bonnie-Lab** | *Keep experimenting, keep learning.*  
*Last Updated: 2026-05-14*
