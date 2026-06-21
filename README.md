# GPT-2 Fine-Tuning for Tweet Sentiment Analysis

本项目基于 GPT-2 (124M) 预训练语言模型，针对三分类（消极 Negative / 中性 Neutral / 积极 Positive）的 Tweet 情感分析任务，系统地探索和评估了多种在小规模数据条件下的高效微调策略。

## 核心亮点 (Key Highlights)
- **从 41.8% 到 100% 的飞跃**：仅替换原生分类头效果有限（41.8%），但通过局部参数解冻，测试集准确率达到了惊人的 **100%**。
- **结构性改进 (Masked Average Pooling)**：引入掩码平均池化构建全局句级语义表示，完美解决了传统 GPT-2 仅依赖最后一个 Token 导致长句语义偏置的问题，带来 **+53.4%** 的强力性能提升。
- **轻量级适配 (Layer Unfreezing)**：仅选择性解冻最后一层 Transformer Block (Block 11) 和 Final LayerNorm，在保持海量预训练语言知识的同时，实现了高效的下游任务对齐，有效防止了小样本场景下的过拟合与灾难性遗忘。
- **数据增强 (Data Augmentation)**：利用大模型改写与同义词扩增将训练集扩展至 5,000 条，使基线模型泛化能力大幅提升。

## 实验方案与结果对比 (Experiment Results)

| 微调策略 (Strategy) | 测试集准确率 (Test Accuracy) | 特点与机制分析 (Mechanism) |
| :--- | :---: | :--- |
| **仅分类头 (Baseline)** | **41.8%** | 局限于句尾单个 Token，长句全局语义严重缺失。 |
| **Masked Average Pooling** | **95.2%** | 全局句级语义聚合，对句长鲁棒，性能发生强提升。 |
| **解冻最后一层 (Unfreeze Last Layer)** | **100.0%** | 高度定制化情感语义，保留原模型泛化能力，表现最佳。 |
| **池化 + 解冻组合** | **95.4%** | 池化的均值操作轻微稀释了注意力机制对关键情感词的放大效应。 |
| **数据增强 (冻结基线)** | **62.0%** | 丰富了句式与多样性，在小样本场景下证明了其鲁棒性。 |

## 项目结构 (Repository Structure)
- `分类头.py` - 仅微调线性分类器的基线实验
- `池化.py` - 引入 Masked Average Pooling 的结构改进实验
- `解冻一层.py` - 解冻最后一层 Transformer Block 的轻量级微调实验
- `数据增强.py` - 基于 5000 条扩增数据的微调实验
- `池化+解冻+增强.py` - 多策略组合消融实验
- `LLM_.pdf` - 本项目的完整系统性实验与消融分析学术报告书

## 核心结论 (Key Takeaways)
1. 在资源受限的小样本场景下，**结构优化（Pooling）与局部参数微调（Layer Unfreezing）相结合**是预训练大模型高效定制的最佳途径。
2. 实验意外发现，自回归结构（GPT-2）对输入格式非常敏感（如句首大写、末尾标点）。在小样本微调中，模型易将高频格式模式误当作“伪特征”进行利用，这为未来数据增强中引入格式扰动提供了重要启示。

---
*详细的数学推导、损失函数曲线及消融实验细节，请参阅项目中的 [LLM_.pdf](./LLM_.pdf) 报告书。*
*权重以及数据请参考（https://drive.google.com/drive/folders/1HKdYt2ZpIkNZwL0FokNSDSqR4jvZSCYs?usp=sharing）。*
