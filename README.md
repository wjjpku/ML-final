# ML-final

 ## 贡献者

 - Jiaju Wu
 - Kairui Li
 - Kehan Huang

---

### **1. 数据集**
#### **数据生成**
- 数据集由二元运算表组成，形式为 $a \circ b = c$，其中 $a, b, c$ 是离散符号，$\circ$ 是二元运算。
- 使用的二元运算包括：
  1. $x \circ y = (x + y) \mod p$，其中 $0 \leq x, y < p$。
  2. $x \circ y = (x - y) \mod p$，其中 $0 \leq x, y < p$。
  3. $x \circ y = x / y \mod p$，其中 $0 \leq x < p, 0 < y < p$。
  4. $x \circ y = [\,x / y \mod p\ \text{ if } y \text{ is odd },\ \text{ otherwise } x - y \mod p\,]$。
  5. $x \circ y = x^2 + y^2 \mod p$，其中 $0 \leq x, y < p$。
  6. $x \circ y = x^2 + xy + y^2 \mod p$，其中 $0 \leq x, y < p$。
  7. $x \circ y = x^2 + xy + y^2 + x \mod p$，其中 $0 \leq x, y < p$。
  8. $x \circ y = x^3 + xy \mod p$，其中 $0 \leq x, y < p$。
  9. $x \circ y = x^3 + xy^2 + y \mod p$，其中 $0 \leq x, y < p$。
  10. $x \circ y = x \cdot y$，其中 $x, y \in S_5$（对称群）。
  11. $x \circ y = x \cdot y \cdot x^{-1}$，其中 $x, y \in S_5$。
  12. $x \circ y = x \cdot y \cdot x$，其中 $x, y \in S_5$。

- **模数 $p$**：在实验中使用 $p = 97$。

#### **数据划分**
- 从所有可能的方程中随机选择一部分作为训练集，其余作为验证集。
- 训练集的比例可以是 50%、30%、25% 等，具体比例会影响模型的泛化能力。

#### **符号表示**
- 每个符号（包括 $a, b, c, \circ, =$）都被表示为独立的 token。
- 数据集中的每个方程形式为 $⟨x⟩⟨op⟩⟨y⟩⟨=⟩⟨x \circ y⟩$，其中 $⟨a⟩$ 表示对应元素 $a$ 的 token。

---

### **2. 模型架构**
- **模型类型**：使用标准的解码器-only Transformer，具有因果注意力掩码。
- **模型参数**：
  - 层数：2 层
  - 隐藏层宽度：128
  - 注意力头数：4
  - 非嵌入参数总数：约 4×10⁵

---

### **3. 优化器与超参数**
- **优化器**：
  - 主要使用 AdamW 优化器，部分实验使用 Adam 优化器。
  - AdamW 的默认设置：
    - 学习率：10⁻³
    - 权重衰减：1
- 超参数 $\beta_1 = 0.9$，$\beta_2 = 0.98$
    - 学习率预热：线性预热，前 10 次更新
    - 小批量大小：512 或训练集大小的一半（取较小值）
    - 优化步数：10⁵ 步
  - 其他实验设置：
    - Adam 优化器（无权重衰减）：用于强调过拟合后泛化能力的延迟提升。
    - Adam 优化器（全批量）：计算整个训练集的精确梯度。
    - Adam 优化器（梯度噪声）：在更新方向中添加高斯噪声。
    - Adam 优化器（残差 dropout = 0.1）：在模型中添加 dropout。
    - AdamW 优化器（权重衰减到初始化点）：权重衰减目标为初始化权重。
    - Adam 优化器（学习率 = 3×10⁻⁴ 或 3×10⁻³）：测试不同学习率。
    - Adam 优化器（权重噪声）：在模型中添加高斯权重噪声（标准差 = 0.01）。

---

### **4. 训练细节**
- **损失计算**：仅在方程的输出部分计算损失和准确率。
- **随机种子**：每个实验重复 3 次，部分实验重复 7 次以统计结果。
- **噪声和正则化**：
  - 残差 dropout（0.1）或高斯权重噪声（标准差 0.01）。
  - 在梯度更新中添加高斯噪声（标准差 = 1）。

---

### **5. 实验设置**
- **数据集大小**：通过改变训练数据的比例（如 50%、30%、25% 等）来研究数据效率。
- **泛化测量**：通过验证集准确率来评估模型的泛化能力。
- **优化步数**：
  - 一般实验：10⁵ 步
  - 部分实验（如研究泛化时间曲线）：5×10⁵ 步
  - 部分实验（如研究过拟合后泛化能力）：10⁶ 步

---

### **6. 分析与可视化**
- **可视化嵌入**：
  - 使用 t-SNE 投影输出层权重，观察模型是否学习到数学对象的结构（如模加法的环状结构）。
  - 例如，模加法的嵌入可能形成一个“数轴”，通过添加 8 来连接每个元素。
- **损失曲线**：
  - 记录训练和验证损失曲线，观察是否存在双下降现象（即验证损失在训练损失下降后再次下降）。

---

### **7. 其他注意事项**
- **硬件需求**：实验可以在单个 GPU 上快速复现。
- **代码实现**：可以使用 PyTorch 或其他深度学习框架实现 Transformer 模型和训练流程。
- **结果对比**：
  - 关注模型在不同数据集大小和优化设置下的泛化能力，尤其是“grokking”现象（即在严重过拟合后验证准确率突然提升）。
  - 关注训练时间曲线，验证准确率达到 99% 所需的优化步数随数据集大小的变化。

---

### **项目架构**
- 入口脚本：`train.py`（解析命令行参数并启动训练）
- 模块目录：`mlfinal/`
  - `config.py`：配置数据类 `Config`（.\mlfinal\config.py:4）
  - `data.py`：数据与词表，支持模运算与 `S5` 群（.\mlfinal\data.py:93）
  - `model.py`：解码器-only Transformer（.\mlfinal\model.py:1）
  - `utils.py`：优化器构建、训练曲线、t-SNE 可视化（.\mlfinal\utils.py:1）
  - `trainer.py`：训练与评估主循环（.\mlfinal\trainer.py:10）

### **使用方法**
- 快速运行：`python train.py --op mod_add --steps 3000 --target-val-acc 0.90`
- 切换操作：
  - 模运算：`--op mod_add|mod_sub|mod_div|div_or_sub_by_y_parity|x2_y2|x2_xy_y2|x2_xy_y2_plus_x|x3_xy|x3_xy2_plus_y`
  - 群运算：`--op s5_mul|s5_conj|s5_x_y_x`
- 典型实验：
  - Grokking：`python train.py --op mod_add --train-ratio 0.25 --steps 100000 --target-val-acc 0.9995 --optimizer adam --dropout 0.1 --grad-noise-std 1.0`
  - 权重噪声：`python train.py --op mod_add --steps 100000 --weight-noise-std 0.01`
  - 学习率变体：`python train.py --op mod_add --lr 3e-4` 或 `--lr 3e-3`
- 依赖安装：`pip install torch matplotlib`，可选 t-SNE：`pip install scikit-learn`

### **Jupyter Notebook**
- 可视化与汇总：打开 `experiments.ipynb`，按顺序运行各单元以：
  - 运行多种运算的短训练并绘制 Loss/Accuracy 曲线
  - 进行输出层权重的 t-SNE 可视化（如安装了 `scikit-learn`）
  - 在项目根目录启动可直接引用模块 `mlfinal/*`

### **支持的运算**
- 模运算（`p=97`）：加、减、除、奇偶分支、若干多项式组合（.\mlfinal\data.py:28）
- 群运算：`S5` 的乘法、共轭、`x·y·x`（.\mlfinal\data.py:65）
- 统一样本格式：`⟨x⟩⟨op⟩⟨y⟩⟨=⟩⟨x∘y⟩`，仅监督最后一位（.\mlfinal\trainer.py:72）

### **关键超参数**
- 数据：`--p`、`--train-ratio`、`--op`
- 模型：`--d-model`、`--n-layers`、`--n-heads`、`--dropout`
- 优化器：`--optimizer adam|adamw`、`--lr`、`--weight-decay`、`--warmup-steps`
- 训练：`--steps`、`--batch-size`、`--full-batch`、`--eval-interval`、`--target-val-acc`
- 正则与噪声：`--grad-noise-std`、`--weight-noise-std`、`--decay-to-init --decay-to-init-lambda`
- 可视化：`--tsne` 或 `--no-tsne`

### **原理要点**
- 序列建模思路：将二元运算表转为长度 5 的短序列，最后一位为监督目标，避免不必要的语言损失污染（.\mlfinal\trainer.py:78）
- 因果注意力：下三角掩码，确保第 5 位的预测仅依赖前 4 位输入（.\mlfinal\model.py:17）
- 训练/评估一致：训练时与评估时都以等号 token 作为占位输入第 5 位（.\mlfinal\trainer.py:72, .\mlfinal\trainer.py:49）
- 优化与正则：支持 Adam/AdamW、线性预热、梯度裁剪、梯度与权重噪声、向初始化点衰减，覆盖 README 中主要实验设置（.\mlfinal\trainer.py:81, .\mlfinal\utils.py:23）
- 嵌入分析：输出层权重 t-SNE，直观观察模型学习到的结构（.\mlfinal\utils.py:27）

### **常见问题**
- OpenMP 冲突：如出现 `libiomp5md.dll` 重复初始化，代码已设置环境变量自动绕过（`mlfinal/utils.py`）；仍建议统一依赖版本。
- t-SNE 依赖：未安装 `scikit-learn` 时自动跳过并正常训练。
- `S5` 数据集规模：词表大小 120，样本 14400，训练更慢；建议提高步数与合适超参。