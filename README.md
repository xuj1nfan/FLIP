# 法律推理机制诊断与 GRPO 实验

这是 `code_artifact.md` 的服务器实验代码。两种任务共用结构化推理协议，以模型生成的最终标签是否正确作为奖励；标准随机训练、随机选题、常规特征筛选、机制筛选和瓶颈续写可使用相同训练器。**目前没有运行模型或取得任何实验结果。**

## 环境与数据

服务器安装 Python 3.10+、适配 CUDA 的 PyTorch，然后 `pip install -r requirements.txt`。模型采用 Hugging Face causal LM，建议使用中文指令模型；`--model` 可为本地路径或模型名，`--adapter` 指向本项目训练生成的 LoRA 目录。本机无 GPU，示例数据仅供检查格式，不是实验数据。

每行一个 JSON 对象，参考 `examples/data.jsonl`：`id` 全局唯一；`task` 为任务名；`text` 为案情或 NLI 前提和假设；`labels` 是该题允许的完整标签列表；`answer` 是其中一个标签。使用彼此独立的训练、探针拟合集、验证和测试集；罪名数据需自行预处理为单被告且避免答案泄漏，NLI 保持封闭前提。固定标签文字及顺序。轨迹若未遵守 `[final] 标签` 格式则视为错误；槽位格式为 `[facts] yes: ...`、`[rule] no: ...`、`[exception] unknown: ...`。

以下命令从项目根目录运行，路径仅为示例。所有模型调用参数应在各组保持一致。先用独立数据收集基线轨迹并训练标准 GRPO：

```bash
python -m legal_rl.rollouts --model MODEL --data data/train.jsonl --output runs/base_train.jsonl --samples 4
python -m legal_rl.rollouts --model MODEL --data data/probe.jsonl --output runs/base_probe.jsonl --samples 1 --temperature 0
python -m legal_rl.rollouts --model MODEL --data data/valid.jsonl --output runs/base_valid.jsonl --samples 1 --temperature 0
python -m legal_rl.train --model MODEL --data data/train.jsonl --output runs/grpo --steps 1000 --group-size 4
python -m legal_rl.rollouts --model MODEL --adapter runs/grpo --data data/train.jsonl --output runs/grpo_train.jsonl --samples 4
python -m legal_rl.rollouts --model MODEL --adapter runs/grpo --data data/probe.jsonl --output runs/grpo_probe.jsonl --samples 1 --temperature 0
python -m legal_rl.rollouts --model MODEL --adapter runs/grpo --data data/valid.jsonl --output runs/grpo_valid.jsonl --samples 1 --temperature 0
python -m legal_rl.analyze transitions --base runs/base_valid.jsonl --grpo runs/grpo_valid.jsonl --output runs/transitions.jsonl
```

`transitions` 按任务统计基线错误、GRPO 正确的配对样本中各槽位的有效状态翻转率。依据验证集结果人工选择 target 和 control 槽位；仅凭状态差异无法证明其因果作用或识别“仅措辞变化”。探针按任务使用各节点最后一层隐藏状态拟合线性分类器，目标是模型最终**预测**标签，使用独立探针数据拟合，base 和 GRPO 各自拟合：

```bash
python -m legal_rl.diagnose --model MODEL --rollouts runs/base_probe.jsonl --fit-probe runs/base_probe.pkl --output runs/unused.jsonl
python -m legal_rl.diagnose --model MODEL --rollouts runs/base_valid.jsonl --probe runs/base_probe.pkl --output runs/base_diagnostic.jsonl
python -m legal_rl.diagnose --model MODEL --adapter runs/grpo --rollouts runs/grpo_probe.jsonl --fit-probe runs/grpo_probe.pkl --output runs/unused.jsonl
python -m legal_rl.diagnose --model MODEL --adapter runs/grpo --rollouts runs/grpo_valid.jsonl --probe runs/grpo_probe.pkl --output runs/grpo_diagnostic.jsonl
python -m legal_rl.intervene --model MODEL --data runs/base_diagnostic.jsonl --slot exception --output runs/base_flip.jsonl
python -m legal_rl.intervene --model MODEL --adapter runs/grpo --data runs/grpo_diagnostic.jsonl --slot exception --output runs/grpo_flip.jsonl
```

对 control 槽位重复翻转。`ECI` 是探针对最终预测标签在起点与结尾的概率比，可大于 1；各节点概率和最大变化槽位也会保存。`CCR` 是标签候选序列归一化概率在屏蔽结论位置对中间推理的注意力前后的 KL。该注意力干预依赖模型支持 Transformers 的 4D additive attention mask 和 eager 注意力，正式跑前应在目标架构上验证。翻转只改 yes/no 状态并丢弃该行旧理由，然后续写；`C` 为正确标签概率变化绝对值、`E` 为标签分布总变差、`Q` 为生成标签是否改变。缺失标签/不可翻转状态会跳过，需报告有效样本数；槽位合理性需要人工核验。

## 训练收益、筛选及消融

用**训练集**重复诊断/翻转，以免筛选触碰测试集；训练前后的多次采样用于估计 `gain` 和 `U=4p(1-p)`：

```bash
python -m legal_rl.prepare --data runs/train_flip.jsonl --before runs/base_train.jsonl --after runs/grpo_train.jsonl --output runs/features.jsonl
python -m legal_rl.analyze correlate --data runs/features.jsonl --output runs/correlations.jsonl
python -m legal_rl.analyze select --data runs/features.jsonl --method random --count 500 --output runs/random.jsonl
python -m legal_rl.analyze select --data runs/features.jsonl --method conventional --count 500 --output runs/conventional.jsonl
python -m legal_rl.analyze select --data runs/features.jsonl --method mechanism --count 500 --output runs/mechanism.jsonl
python -m legal_rl.train --model MODEL --data runs/mechanism.jsonl --output runs/mechanism_model --steps 1000
python -m legal_rl.train --model MODEL --data runs/mechanism.jsonl --output runs/branch_model --steps 1000 --branch
```

`prepare` 按可配置阈值划分 discovery、justification 和 mixed；阈值和评分权重应在验证数据确定后固定。综合分数默认相加 `U + alignment`，低 ECI 样本再加 `C + E + Q`；缺项按 0 处理。要实现高 ECI 不做翻转的省算力分流，应先诊断并按 ECI 将数据分开，只对低 ECI 数据运行 `intervene`，再将结果与高 ECI 数据合并。可用 `--eci-threshold` 调整筛选阈值。

可选的梯度特征使用正确轨迹的教师强制损失梯度，与独立验证集平均梯度计算余弦：

```bash
python -m legal_rl.alignment --model MODEL --data runs/features.jsonl --validation runs/base_valid.jsonl --output runs/features_aligned.jsonl
```

这是 LoRA 梯度代理，**不是** GRPO 单题实际更新梯度；只输出有正确轨迹的样本。若无需该代理可跳过，相关性分析会仅在有 `alignment` 的行上计算。`prepare` 从正确轨迹提取瓶颈槽位之前的前缀，`--branch` 对有此前缀的样本只生成和优化后缀，其余样本从头采样；正确最终标签仅是前缀的弱验证，正式实验应人工或用独立标注确认前缀。训练日志记录每步题目、组奖励和生成 token 数，比较四组时自行控制并报告总生成 token、诊断及筛选成本；固定步数不等于固定计算预算。

训练器使用结果奖励、组内标准化优势、逐 token 裁剪策略比率和 LoRA；默认每组一次更新，不含参考模型 KL。所有组应采用相同超参数并重复多个随机种子。最终在独立测试集运行 `rollouts`，按任务、种子汇总准确率、格式有效率及计算成本。相关性仅是探索性 Spearman 分析，报告样本量、置信区间及多重比较处理；不要将诊断关联直接表述成因果结论。
