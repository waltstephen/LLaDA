# 技术方案文档：使用Prompt Tuning和MeanFlow蒸馏LLaDA实现1-Step生成（我们的模型叫LLaDA-MeanFlow）

## 1. 目标

本项目旨在利用参数高效微调（PEFT）技术，将一个预训练好的、多步生成的LLaDA模型，通过蒸馏的方式，转化为一个能够进行高质量**单步（1-Step）**生成的模型。我们将采用**Prompt Tuning**作为PEFT方法，并借鉴**MeanFlow**理论来构建蒸馏过程中的学习目标。

### 1.1 实现细节

已知meanflow在dit上的实现在/home/wangkz/yijia/LLaDA/MeanFlow/meanflow.py下，这是一个1step生成图片的diffusion模型，现在我要把它应用到diffusion LLM（LLaDA）见/home/wangkz/yijia/LLaDA/README.md上微调出来一个1stept生成的diffusion LLM,具体实现如下

请注意这里有LLaDA是离散空间，Meanflow是连续空间，但是这种平均速度流再数学上是可以变换等价的，你需要考虑这个差异

请一定要使用pytorch-lighting架构！！！

## 2. 核心思想

1.  **教师-学生框架 (Teacher-Student Framework)**:
    *   **教师 (Teacher)**: 预训练好的、冻结参数的LLaDA模型。它代表了当前最优的“瞬时速度”预测能力。
    *   **学生 (Student)**: 同样是LLaDA模型，但其基础权重被冻结。我们将为其添加一个可训练的**软提示（Soft Prompt）模块**。这个学生模型的目标是学习“平均速度”。

2.  **MeanFlow 恒等式作为学习目标**:
    *   我们将利用MeanFlow理论的核心恒等式 `u = v - (t-r) * du/dt` 来构建损失函数。
    *   `u` (平均速度) 由**学生模型**预测。
    *   `v` (瞬时速度) 由**教师模型**提供。
    *   `du/dt` (平均速度的变化率) 通过对学生模型计算**雅可比-向量积 (JVP)** 得到。

3.  **Prompt Tuning 作为条件注入机制**:
    *   MeanFlow模型需要接收三个时间输入 `(xₜ, r, t)`。标准LLaDA只接收 `(xₜ, t)`。
    *   我们将使用一个小的**提示网络 (Prompt Network)**，根据时间区间 `(r, t)` 生成一组可训练的“软提示”向量。
    *   这些软提示向量将被注入到学生模型的每一层，从而让学生模型的行为能够根据时间区间 `(r, t)` 进行条件化调整，使其能够学习平均速度 `u`。

## 3. 模型架构与组件

### 3.1 教师模型 (`LLaDA_teacher`)

*   **来源**: 预训练好的LLaDA模型。
*   **状态**: **完全冻结 (Frozen)**，在训练中不更新任何参数。
*   **作用**: 在每个训练步骤中，接收带噪输入 `(xₜ, t)`，并提供“瞬时速度 `v`”的预测。
    ```python
    # 伪代码
    v_prediction = LLaDA_teacher(input_ids=x_t, time=t)
    ```

### 3.2 学生模型 (`LLaDA_student`)

这是一个复合模型，由两部分组成：

#### 3.2.1 冻结的LLaDA基础 (`LLaDA_base`)

*   **来源**: 与教师模型结构和权重完全相同。
*   **状态**: **完全冻结 (Frozen)**。

#### 3.2.2 可训练的软提示模块 (`SoftPrompt_Module`)

这是我们**唯一需要训练**的部分。

*   **输入**: 起始时间 `r` 和结束时间 `t`。
*   **结构**:
    1.  **时间编码器 (Time Encoder)**: 将标量 `r` 和 `t` 转换为高维向量 `emb(r)` 和 `emb(t)`。
    2.  **提示网络 (Prompt Network)**: 一个小型的MLP（例如，2层），接收 `emb(r)` 和 `emb(t)` 的拼接，输出软提示向量。
        *   `prompt_vectors = MLP(concat(emb(r), emb(t)))`
    3.  输出的 `prompt_vectors` 维度应为 `(num_layers, 2, num_heads, prompt_length, head_dim)`，以便注入到每一层的Key和Value缓存中。
*   **作用**: 生成条件化的软提示，用于指导`LLaDA_base`的行为。

### 3.3 学生模型的完整前向传播

学生模型的前向传播 `u = LLaDA_student(xₜ, r, t)` 过程如下：

1.  用软提示模块生成提示向量：`prompts = SoftPrompt_Module(r, t)`。
2.  将 `xₜ` 和时间 `t` 输入到 `LLaDA_base`。
3.  在`LLaDA_base`的每个Transformer层的自注意力计算中，将 `prompts` 注入到Key和Value缓存中（Prefix-Tuning的方式）。
4.  返回最终的预测结果，这个结果就是我们定义的“平均速度 `u`”。

## 4. 训练流程与损失函数

这是一个蒸馏循环，对于每个训练批次：

1.  **数据准备**:
    *   从数据集中采样一个干净的文本序列 `x₀`。
    *   随机采样时间 `t` 和 `r`，满足 `0 <= r < t <= 1`。
    *   通过随机掩码 `x₀` 生成带噪输入 `xₜ`。

2.  **计算瞬时速度 `v` (教师)**:
    *   `v = LLaDA_teacher(x_t, t)`。
    *   这个 `v` 是一个logits向量，代表了对原始序列 `x₀` 的预测分布。

3.  **计算平均速度 `u` (学生)**:
    *   `u = LLaDA_student(x_t, r, t)`。
    *   `u` 同样是一个logits向量。

4.  **计算JVP `du/dt` (学生)**:
    *   我们需要计算 `u` 对时间 `t` 的全导数。根据链式法则，`du/dt = (∂u/∂xₜ) * (dxₜ/dt) + (∂u/∂t)`。
    *   在MeanFlow中，`dxₜ/dt` 就是瞬时速度 `v`。
    *   因此，我们需要计算 `JVP = (∂u/∂xₜ) @ v + (∂u/∂t)`。
    *   **实现注意**: `∂u/∂xₜ` 是一个雅可比矩阵，我们不需要显式构造它。现代框架（JAX, PyTorch）可以直接计算雅可比-向量积 `(∂u/∂xₜ) @ v`。
    *   `du_dt = jax.jvp(lambda xt, t: LLaDA_student(xt, r, t), (x_t, t), (v, 1.0))`
        *   这里的 `(v, 1.0)` 是切向量（tangent vector），分别对应 `xₜ` 和 `t` 的变化方向。

5.  **构建MeanFlow目标 `u_tgt`**:
    *   `u_tgt = v - (t - r) * du_dt`

6.  **计算损失**:
    *   我们希望学生预测的 `u` 尽可能接近 `u_tgt`。可以使用交叉熵损失或MSE损失。
    *   **MSE Loss (在logits空间)**:
        `loss = mean_squared_error(u, stop_gradient(u_tgt))`
    *   **Cross-Entropy Loss**:
        `loss = cross_entropy(logits=u, labels=softmax(stop_gradient(u_tgt)))`
    *   `stop_gradient` 至关重要，确保教师和JVP部分不参与梯度计算，只作为目标。

7.  **反向传播与优化**:
    *   计算 `loss.backward()`。
    *   梯度只会流向**软提示模块 (`SoftPrompt_Module`)** 的参数。
    *   使用优化器（如AdamW）更新软提示模块的参数。

## 5. 推理 (1-Step 生成)

训练完成后，推理过程如下：

1.  加载冻结的 `LLaDA_base` 和训练好的 `SoftPrompt_Module`。
2.  给定一个完全被 `[MASK]` 的初始序列 `x₁`。
3.  设置 `r=0` 和 `t=1`。
4.  调用学生模型进行一次前向传播：`u = LLaDA_student(x₁, r=0, t=1)`。
5.  根据MeanFlow的采样公式 `x₀ = x₁ - (t-r)u`，在 `t=1, r=0` 时简化为 `x₀ = x₁ - u`。
    *   在离散文本空间，这个操作可以解释为：**直接使用 `u` 的预测logits来贪心解码或采样，生成最终的文本序列 `x₀`**。
    ```python
    # 伪代码
    initial_mask = create_fully_masked_sequence(length)
    # u_logits 是学生模型在 r=0, t=1 条件下的输出
    u_logits = LLaDA_student(initial_mask, r=0.0, t=1.0)
    # 一步生成最终文本
    generated_ids = torch.argmax(u_logits, dim=-1)
    ```

## 6. 实施要点与伪代码

```python
# main_training_loop.py

# 1. 初始化模型
LLaDA_teacher = load_pretrained_llada().eval().requires_grad_(False)
LLaDA_base = load_pretrained_llada().eval().requires_grad_(False)
soft_prompt_module = SoftPrompt_Module(...) # 可训练
optimizer = AdamW(soft_prompt_module.parameters(), lr=1e-4)

# 2. 训练循环
for batch in dataloader:
    x0 = batch['input_ids']
    
    # 采样时间和噪声
    t, r = sample_times() # e.g., t in [0,1], r in [0,t)
    xt = mask_data(x0, t)
    
    # 定义学生模型函数，用于JVP计算
    def student_fn(current_xt, current_t):
        prompts = soft_prompt_module(r, current_t)
        # LLaDA_base 内部使用 prompts
        return LLaDA_base(current_xt, current_t, prompts=prompts)

    # 教师提供瞬时速度 v
    with torch.no_grad():
        v = LLaDA_teacher(xt, t)

    # JVP 计算 du/dt
    # PyTorch 2.0+ 的 functorch 或 JAX 的 jvp
    u, du_dt = torch.func.jvp(
        lambda _xt, _t: student_fn(_xt, _t),
        (xt, t),
        (v, torch.ones_like(t)) # 切向量
    )
    
    # 构建目标 u_tgt
    with torch.no_grad():
        u_tgt = v - (t - r) * du_dt
        
    # 计算损失
    # u 是上面 jvp 计算返回的 primal output
    loss = F.mse_loss(u, u_tgt)
    
    # 优化
    optimizer.zero_grad()
    loss.backward() # 梯度只会流向 soft_prompt_module
    optimizer.step()

# 技术方案文档：使用Prompt Tuning和MeanFlow蒸馏LLaDA实现1-Step生成 (离散空间优化版)

## 1. 目标

本项目旨在利用参数高效微调（PEFT）技术，将一个预训练好的、多步生成的离散扩散语言模型（LLaDA），通过蒸馏的方式，转化为一个能够进行高质量**单步（1-Step）**生成的模型。我们将采用**Prompt Tuning**作为PEFT方法，并借鉴**MeanFlow**理论来构建蒸馏过程中的学习目标，**所有公式均针对离散数据的概率/logits空间进行适配**。

## 2. 核心思想 (离散空间视角)

1.  **教师-学生框架**:
    *   **教师 (Teacher)**: 冻结的LLaDA模型。它提供的是**瞬时恢复概率分布**，即在`t`时刻，对每个被掩码位置的正确词元的预测概率。
    *   **学生 (Student)**: 带有可训练软提示的LLaDA模型。它的目标是学习**平均恢复概率分布**，即从`r`到`t`整个时间区间的平均恢复效果。

2.  **MeanFlow 恒等式 (离散概率空间版)**:
    *   在离散空间，速度场被概率流所取代。恒等式 `u = v - (t-r) * du/dt` 的核心思想是：**当前的平均流 `u`，等于瞬时流 `v` 减去一个由平均流自身变化所引起的修正项**。
    *   我们将这个思想应用在**logits空间**，因为logits是无界的，更适合进行类似加减的线性操作，并且最终可以通过`softmax`转换回概率。

3.  **Prompt Tuning**: 机制不变，依然是根据时间区间 `(r, t)` 生成软提示，注入到学生模型中，使其具备条件化预测能力。

## 3. 模型架构与组件

模型架构部分与前一版文档相同：包含一个冻结的`LLaDA_teacher`，以及一个由冻结的`LLaDA_base`和可训练的`SoftPrompt_Module`组成的学生模型。关键变化在于我们如何解释和处理它们的输出。

*   **模型输出**: 教师和学生模型的输出都是**logits向量**，维度为 `(batch_size, seq_length, vocab_size)`。

## 4. 训练流程与损失函数 (离散空间适配版)

这是一个蒸馏循环，对于每个训练批次：

1.  **数据准备**:
    *   采样干净文本 `x₀`。
    *   采样时间 `t` 和 `r` (`0 <= r < t <= 1`)。
    *   生成带噪（掩码）输入 `xₜ`。

2.  **计算瞬时流 `v` (教师)**:
    *   `v_logits = LLaDA_teacher(x_t, t)`。
    *   `v_logits` 代表了在`t`时刻，对`x₀`的预测logits。这是我们最可靠的“瞬时”信息。

3.  **计算平均流 `u` (学生)**:
    *   `u_logits = LLaDA_student(x_t, r, t)`。
    *   `u_logits` 是学生模型对`x₀`的预测logits，它应该蕴含了从`r`到`t`的平均信息。

4.  **计算JVP `du/dt` (学生，在logits空间)**:
    *   这是最关键的适配步骤。`du/dt`衡量的是当时间`t`和带噪数据`xₜ`（其变化由瞬时流`v`驱动）发生微小变化时，`u_logits`会如何变化。
    *   **切向量 (Tangent Vector)**:
        *   对于输入`xₜ`，它的变化方向不再是简单的向量`v`。在离散空间，`xₜ`的变化由一个概率流驱动。我们可以用**教师预测的logits `v_logits`** 作为这个流的代表。
        *   对于时间`t`，它的变化方向依然是`1.0`。
    *   **JVP计算**: 我们计算学生模型函数（输出`u_logits`）在点`(xt, t)`处，沿着切向量`(v_logits, 1.0)`的JVP。
    *   `u_logits_primal, du_dt_logits = jax.jvp(...)` 或 `torch.func.jvp(...)`
    *   `du_dt_logits` 的维度也是 `(batch_size, seq_length, vocab_size)`，它代表了平均流logits的变化率。

5.  **构建MeanFlow目标 `u_tgt` (在logits空间)**:
    *   `u_tgt_logits = v_logits - (t - r) * du_dt_logits`
    *   这个线性操作在logits空间是稳定且有意义的。

6.  **计算损失**:
    *   我们希望学生预测的 `u_logits` 尽可能地匹配目标 `u_tgt_logits`。
    *   **首选损失：KL散度 (Kullback-Leibler Divergence)**。它专门用于衡量两个概率分布的差异，比MSE更适合分类任务。
        *   `p_tgt = softmax(stop_gradient(u_tgt_logits), dim=-1)`
        *   `log_q_student = log_softmax(u_logits, dim=-1)`
        *   `loss = kl_div(log_q_student, p_tgt, reduction='batchmean')`
    *   **备选损失：交叉熵 (Cross-Entropy)**。如果我们将教师的最终预测（或真实`x₀`）作为硬标签，也可以用交叉熵，但KL散度能更好地利用教师分布的“软”信息。
    *   `stop_gradient` 依然至关重要。

7.  **反向传播与优化**:
    *   `loss.backward()`。梯度只会更新**软提示模块**的参数。
    *   `optimizer.step()`。

## 5. 推理 (1-Step 生成，离散空间版)

推理过程更加清晰：

1.  加载冻结的 `LLaDA_base` 和训练好的 `SoftPrompt_Module`。
2.  创建全掩码序列 `x₁`。
3.  设置 `r=0`, `t=1`。
4.  **进行一次前向传播，得到平均流的预测logits**:
    *   `u_logits = LLaDA_student(x₁, r=0, t=1)`。
5.  **直接从该logits生成文本**:
    *   对 `u_logits` 在词汇表维度上进行`argmax`（贪心解码）或`sample`（随机采样），直接得到最终的完整文本序列 `x₀`。

## 6. 实施要点与伪代码 (离散空间优化版)

```python
# main_training_loop_discrete.py
import torch
import torch.nn.functional as F
# 假设使用 PyTorch 2.0+ 的 functorch
from torch.func import jvp

# 1. 初始化模型
LLaDA_teacher = load_pretrained_llada().eval().requires_grad_(False)
LLaDA_base = load_pretrained_llada().eval().requires_grad_(False)
soft_prompt_module = SoftPrompt_Module(...) # 可训练
optimizer = torch.optim.AdamW(soft_prompt_module.parameters(), lr=1e-4)

# 2. 训练循环
for batch in dataloader:
    x0 = batch['input_ids']
    
    # 采样时间和噪声
    t, r = sample_times() # e.g., t in [0,1], r in [0,t)
    xt = mask_data(x0, t) # xt 包含 [MASK] token
    
    # 定义学生模型函数，用于JVP计算
    # 它接收 token_ids 和时间标量，返回 logits
    def student_fn(current_xt, current_t):
        prompts = soft_prompt_module(r, current_t)
        # LLaDA_base 内部需要有逻辑来接收和使用 prompts
        return LLaDA_base(current_xt, current_t, prompts=prompts)

    # 教师提供瞬时流 v (logits)
    with torch.no_grad():
        v_logits = LLaDA_teacher(xt, t)

    # JVP 计算 du/dt (在logits空间)
    # primal_outputs 是 u_logits
    u_logits, du_dt_logits = jvp(
        lambda _xt, _t: student_fn(_xt, _t),
        (xt, t),
        # 切向量：xt 的方向是 v_logits，t 的方向是 1.0
        # 注意：xt是离散的，它的切向量是定义在嵌入空间的，这里用logits作为其在语义空间变化方向的代表
        # 这是一个简化但有效的处理。更严谨的做法是在embedding空间计算。
        # 但为了简单起见，我们假设JVP可以直接在logits上操作，代表概率流方向。
        (LLaDA_base.get_input_embeddings()(xt.new_zeros(xt.shape)).normal_(), torch.ones_like(t)) # 一个更严谨的切向量
        # 或者用一个更概念化的切向量
        # tangent_xt_embedding = (v_logits.softmax(-1) - xt_one_hot.float()) # 代表从当前状态到目标的概率流
        # tangent_t = torch.ones_like(t)
    )
    # ！！！注意：JVP在离散输入上的定义需要小心处理。
    # 一个更实际的做法是，JVP的切向量应该作用在模型的连续输入（即embedding）上。
    # tangent_for_xt = LLaDA_base.get_input_embeddings()(torch.argmax(v_logits, -1)) - LLaDA_base.get_input_embeddings()(xt)
    # (u_logits, du_dt_logits) = jvp(..., tangent=(tangent_for_xt, torch.ones_like(t)))
    # 为了让Claude能直接上手，我们先用一个概念上清晰的伪代码
    # 假设我们有一个函数能正确处理这个JVP
    u_logits, du_dt_logits = calculate_jvp_for_discrete_model(student_fn, xt, t, v_logits)


    # 构建目标 u_tgt (在logits空间)
    with torch.no_grad():
        u_tgt_logits = v_logits - (t.view(-1, 1, 1) - r.view(-1, 1, 1)) * du_dt_logits
        
    # 计算损失 (KL散度)
    log_probs_student = F.log_softmax(u_logits, dim=-1)
    probs_target = F.softmax(u_tgt_logits, dim=-1)
    
    # 只在被掩码的位置计算损失
    mask_positions = (xt == MASK_TOKEN_ID)
    loss = F.kl_div(
        log_probs_student[mask_positions], 
        probs_target[mask_positions], 
        reduction='batchmean',
        log_target=False
    )
    
    # 优化
    optimizer.zero_grad()
    loss.backward() # 梯度只会流向 soft_prompt_module
    optimizer.step()

# 辅助函数伪代码
def calculate_jvp_for_discrete_model(fn, xt, t, v_logits):
    """
    这是一个需要仔细实现的函数。
    它应该在模型的embedding空间计算JVP。
    """
    # 1. 获取xt的embedding
    xt_embed = LLaDA_base.get_input_embeddings()(xt)
    
    # 2. 定义切向量
    # v_logits代表了目标方向，我们可以用它来定义embedding空间的变化方向
    # 例如，用v_logits的softmax加权平均所有词的embedding，得到目标embedding
    vocab_embeds = LLaDA_base.get_input_embeddings().weight
    v_probs = F.softmax(v_logits, dim=-1)
    target_embed = torch.einsum('blv,vd->bld', v_probs, vocab_embeds)
    tangent_xt = target_embed - xt_embed # 从当前embedding指向目标embedding的方向
    tangent_t = torch.ones_like(t)

    # 3. 定义一个新的函数，输入为embedding
    def student_fn_embed(xt_embedding, current_t):
        # 这个函数内部直接接收embedding作为输入
        return LLaDA_base.forward_from_embedding(xt_embedding, current_t, prompts=...)

    # 4. 计算JVP
    u_logits, du_dt_logits = jvp(
        lambda _xte, _t: student_fn_embed(_xte, _t),
        (xt_embed, t),
        (tangent_xt, tangent_t)
    )
    return u_logits, du_dt_logits