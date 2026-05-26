# 坐标下降法 (Coordinate Descent Method) 深度解析

> **核心思想（分而治之）**：当自变量 $x$ 的维度较高或目标函数 $f(x)$ 较为复杂时，直接计算全局的（次）梯度往往代价高昂甚至不切实际。坐标下降法通过将高维优化问题拆解，在每次迭代中仅选择一个坐标轴方向进行精确或近似的极小化，从而完全不借助或少借助全局的（次）梯度信息来求解问题 $\min_{x} f(x)$。

---

## 1. 算法流程与核心机制

坐标下降法在整个算法流程中通过**循环遍历**所有的坐标方向来逐步逼近极值点。

### 核心分量更新特征
在每次迭代中，算法选择第 $i$ 个坐标轴方向，并在当前点处沿该轴向进行极小化。其**核心分量更新递推公式**为：

$$
x_i^{k+1} \leftarrow \arg\min_{x_i} f(x_1^{k+1}, \dots, x_{i-1}^{k+1}, x_i, x_{i+1}^k, \dots, x_n^k)
$$

> **特征解析**：每次更新第 $i$ 个分量 $x_i^{k+1}$ 时，所有比它先更新的变量已经使用了最新步长值 $x_1^{k+1}, \dots, x_{i-1}^{k+1}$，而未更新的变量则保持旧值 $x_{i+1}^k, \dots, x_n^k$。整个算法流程中循环遍历所有的坐标方向。

### 算法伪代码 (Algorithm 2)

* **输入**：初始点 $x^0$
* **输出**：最优解估计值 $x^k$

> **初始化**: $k \leftarrow 0$  
> **repeat** > &emsp;&emsp; $x_1^{k+1} \leftarrow \arg\min_{x_1} f(x_1, x_2^k, x_3^k, \dots, x_n^k);$  
> &emsp;&emsp; $x_2^{k+1} \leftarrow \arg\min_{x_2} f(x_1^{k+1}, x_2, x_3^k, \dots, x_n^k);$  
> &emsp;&emsp; $\dots$  
> &emsp;&emsp; $x_n^{k+1} \leftarrow \arg\min_{x_n} f(x_1^{k+1}, x_2^{k+1}, x_3^{k+1}, \dots, x_n);$  
> &emsp;&emsp; $k \leftarrow k + 1$  
> **until** 满足收敛准则;  

---

## 2. 坐标下降法的不收敛反例

坐标下降法**并非在所有场景下都一定收敛**。数学家 Powell 曾在 1973 年提出了一个经典的非收敛反例。

### Powell 反例场景
考虑以下 3 维非凸优化问题：

$$
\min_{x_1, x_2, x_3} -x_1x_2 - x_2x_3 - x_3x_1 + \sum_{i=1}^3 \left( [ (x_i - 1)_+ ]^2 + [ (-x_i - 1)_+ ]^2 \right)
$$

其中 $(\cdot)_+$ 表示 $\max(0, \cdot)$。

当固定 $(x_2, x_3)$ 对 $x_1$ 做极小化时，可以推导得到其更新公式为：

$$
x_1 \in \begin{cases} (1 + 0.5|x_2 + x_3|)\,\text{sign}(x_2 + x_3), & \text{if } x_2 + x_3 \neq 0 \\ [-1, 1], & \text{otherwise} \end{cases}
$$

同理可得关于 $x_2$ 和 $x_3$ 的更新公式。

### 循环震荡行为
若给定一个微小的正数 $\varepsilon > 0$，并将初始点设为 $\left( -1 - \varepsilon, \, 1 + \frac{1}{2}\varepsilon, \, -1 - \frac{1}{4}\varepsilon \right)$，按照 $x_1, x_2, x_3, x_1, x_2, x_3, \dots$ 的顺序轮流进行极小化，状态序列将按如下规律演转：

$$
\xrightarrow{x_1} \left( 1 + \frac{1}{8}\varepsilon, \, 1 + \frac{1}{2}\varepsilon, \, -1 - \frac{1}{4}\varepsilon \right) \xrightarrow{x_2} \left( 1 + \frac{1}{8}\varepsilon, \, -1 - \frac{1}{16}\varepsilon, \, -1 - \frac{1}{4}\varepsilon \right)
$$

$$
\xrightarrow{x_3} \left( 1 + \frac{1}{8}\varepsilon, \, -1 - \frac{1}{16}\varepsilon, \, 1 + \frac{1}{32}\varepsilon \right) \xrightarrow{x_1} \left( -1 - \frac{1}{64}\varepsilon, \, -1 - \frac{1}{16}\varepsilon, \, 1 + \frac{1}{32}\varepsilon \right)
$$

$$
\xrightarrow{x_2} \left( -1 - \frac{1}{64}\varepsilon, \, 1 + \frac{1}{128}\varepsilon, \, 1 + \frac{1}{32}\varepsilon \right) \xrightarrow{x_3} \left( -1 - \frac{1}{64}\varepsilon, \, 1 + \frac{1}{128}\varepsilon, \, -1 - \frac{1}{256}\varepsilon \right) \dots
$$

**结论**：随着迭代的进行，算法不仅无法收敛，其迭代点反而会在以下 **6 个非最优极限点** 附近发生无限循环震荡：

$$
(1, 1, -1) \rightarrow (1, -1, -1) \rightarrow (1, -1, 1) \rightarrow (-1, -1, 1) \rightarrow (-1, 1, 1) \rightarrow (-1, 1, -1)
$$

---

## 3. 核心子概念：坐标极小点 (Coordinate-wise Minimizer)

由于坐标下降法每次只沿一个轴搜索，算法停滞不前时达到的点被称为“坐标极小点”，它与全局最优解有着微妙的关系。

### 定义 1
对于任意函数 $f : \mathbb{R}^n \rightarrow (-\infty, +\infty]$，若存在一个点 $\bar{x} \in \text{dom}\,f$ 满足：

$$
f(\bar{x} + \alpha e_i) \ge f(\bar{x}), \quad \forall i \in [n], \;\forall \alpha \in (-\infty, +\infty)
$$

则称 $\bar{x}$ 为 $f$ 的一个**坐标极小点**。其中 $e_i = [0, \dots, 1, \dots, 0]^T \in \mathbb{R}^n$ 表示第 $i$ 个坐标轴的标准基向量。

> **物理含义**：在点 $\bar{x}$ 处，**沿着任何一个单一的坐标轴方向**进行局部极小化，都无法使目标函数值 $f$ 进一步减小。

---

## 4. 坐标极小点 $\Rightarrow$ 全局最优解？（性质与定理）

即使函数 $f$ 是凸函数，坐标极小点也**不一定**是全局最优解。这取决于函数的**连续可微性**与**可分性**。

### 经典案例对比

#### 示例 1：凸函数 + 不连续可微 $\Rightarrow$ 陷入局部坐标极小点
考虑函数：

$$
f(x_1, x_2) = x_1^2 + x_2^2 + 20|x_1 - x_2|
$$

可以验证点 $(\bar{x}_1, \bar{x}_2) = (-3, -3)$ 是一个坐标极小点，因为：

$$
\bar{x}_1 \in \arg\min_{x_1} f(x_1, \bar{x}_2) = x_1^2 + 20|x_1 + 3|
$$

$$
\bar{x}_2 \in \arg\min_{x_2} f(\bar{x}_1, x_2) = x_2^2 + 20|x_2 + 3|
$$

在 $(-3, -3)$ 处沿 $x_1$ 或 $x_2$ 轴向外探测函数值都会上升。**但 $(-3, -3)$ 并不是全局最优解**，该函数的全局最优解显然在 $(0,0)$。

#### 示例 2：凸函数 + 连续可微 $\Rightarrow$ 成功等价
考虑函数：

$$
f(x_1, x_2) = x_1^2 + x_2^2 + 20(x_1 - x_2)^2
$$

可以验证 $(\bar{x}_1, \bar{x}_2) = (0, 0)$ 是该函数的坐标极小点，并且它同时也是全局最优解。

---

## 5. 核心收敛性定理结论

令 $\bar{x}$ 是目标函数 $f$ 的一个坐标极小点，则有以下基本结论：

1. **连续可微凸函数**：
   若 $f$ 是**连续可微的凸函数**，则坐标极小点 $\bar{x}$ **一定**是 $f$ 的一个全局最优点！
   *(其等高线呈现平滑的椭圆状，算法通过交叉步进最终必能滑向圆心最优处)*
2. **非连续可微凸函数**：
   若 $f$ 是凸函数，但**不连续可微**，则 $\bar{x}$ **不一定是** $f$ 的全局最优点！
   *(如等高线带有尖锐的“棱角”且棱角未与坐标轴对齐时，坐标下降法极易在非最优的尖点处卡死，无法继续下降)*

---
*Last Updated: 2026-05-26*
