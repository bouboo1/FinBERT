模型组任务
- 实现算法框架、把模型跑起来跑出可以使用的结果。
- 详细描述技术方案（算法流程、评价指标、参数设置），给出伪代码（标注关键部分注释）、模型公式
- 绘制表格、图像：数据集构成、训练结果、参数设置、模型图
- 如果有的话：可以继续优化的优化方案（不一定要实现，只是画饼的一个作用）【比如集成模型、加入其他效率硬件等】

# 数据集
链接：https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news

数据集 

| label | 	content|
|  :----: |  :---- |
|neutral	|According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .
|negative	|The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility ; contrary to earlier layoffs the company contracted the ranks of its office workers , the daily Postimees reported .
|positive	|Foundries division reports its sales increased by 9.7 % to EUR 63.1 mn from EUR 57.5 mn in the corresponding period in 2006 , and sales of the Machine Shop division increased by 16.4 % to EUR 41.2 mn from EUR 35.4 mn in the corresponding period in 2006 .

label列表示情感属性，分为neutral、negative、positive三种；content部分是金融相关新闻报道或阐述

# 模型介绍
对比模型有：BiLSTM、BERT-CNN、BERT-wwm、FinBERT。这几个模型基本没怎么优化或者修改过，用的都是最基础的，主要起对比作用

本次实验用的模型是：**FinBERT-RCNN-Attack**模型

**FinBERT**：这是一个基于BERT的金融领域预训练语言模型。与一般的BERT模型类似，FinBERT在大规模金融文本数据上进行了预训练，以捕捉金融领域的语义信息和语境关系。

**RCNN**：这是一个基于循环神经网络（RNN）和卷积神经网络（CNN）的模型。RCNN结合了这两种类型的神经网络，以在文本分类任务中提取特征。（RNN->LSTM, CNN->Conv1D)

**Attack**：Attack指的是对抗训练。对抗训练是一种用于提高模型鲁棒性的技术，通过向模型输入添加微小的、人类不易察觉的扰动来训练模型。在文本分类任务中，对抗训练可以帮助模型更好地应对对抗性攻击、提高泛化能力等。


所有原理与原论文中基本一致，这里简单总结一下代码逻辑，具体参考论文

#### 1. **数据预处理**
- **切分数据集**：见split_data.py，将原有数据集根据三种标签的比例拆分为了训练集和测试集，必须要保证训练集和测试集都有比例合适的三种标签，否则在训练和测试过程中可能会因为学习不充分/过度造成模型效果不佳。
- **分词:** 将文本数据进行分词，填充，截断到最大长度128(这里有可能会对结果造成影响），并将标签转换为整数方便后期计算（`neutral->0, positive->1, negative:->2`）。
- **TensorDataset:** 创建一个包含 `input_ids`、`attention_mask` 和 `labels` 的 `TensorDataset`。

#### 2. **数据加载器创建**
- **训练加载器:** Batch大小为16，启用打乱。（如果数据没有打乱，模型可能会记住数据的顺序而不是学习到数据的真正特征。通过在每个epoch开始时打乱数据，可以确保模型在每个batch中都能接触到不同的样本，从而更好地泛化到其他未见过的数据。）
- **测试加载器:** Batch大小为16，不启用打乱。

#### 3. **RCNN模型部分**（FinBERT直接用现成的）
`RCNN` 类是一个结合BERT、双向LSTM和卷积层的自定义神经网络架构：

- **初始化:**
  - **FinBERT模型:** 预训练的FinBERT。
  - **LSTM层:** 隐藏层大小为768的双向LSTM。
  - **卷积层:** 1D卷积层，输入通道数为1536（由于LSTM是双向的），输出通道数为768。
  - **Dropout层:** 概率为0.5的Dropout层。
  - **全连接层:** 输入大小为768，输出大小为3（标签数）的线性层。

- **前向传播:**
  - **FinBERT输出:** 提取FinBERT的最后隐藏状态。
  - **LSTM层:** 对BERT输出应用双向LSTM。
  - **卷积层:** 对LSTM输出应用1D卷积（先要进行维度变换）。
  - **激活:** 对卷积输出应用ReLU激活函数。
  - **池化:** 应用全局最大池化来获取每个特征图的最大值。
  - **Dropout:** 对池化后的输出应用Dropout。
  - **分类:** 使用全连接层得到最终的logits。

#### 4. **Attack对抗训练函数**
在训练过程中引入扰动，以提高模型的鲁棒性（见论文citation19）

- **前向传播:** 计算FinBERT的前向传播以获得序列输出。
- **梯度保留:** 保留序列输出的梯度。
- **扰动:** 计算基于梯度符号的小扰动（缩放因子为epsilon）。
- **扰动输出:** 将扰动添加到序列输出中。
- **最终前向传播:** 使用扰动后的序列输出重新计算前向传播。

#### 5. **训练和评估**
- **损失函数:** 使用交叉熵损失（CrossEntropyLoss）。
- **优化器:** 使用学习率为5e-5的AdamW优化器。
- **训练:** 进行3个epoch的训练。


#### 6. **模型和指标总结**
- **RCNN层:**
  - FinBERT用于上下文嵌入。
  - 双向LSTM用于序列特征提取。
  - 1D卷积层用于捕捉局部特征。
  - 全局最大池化用于降维。
  - Dropout用于正则化。
  - 全连接层用于分类。

- **对抗训练:** 通过基于梯度的小扰动来增强模型的鲁棒性。

- **指标:** 每个epoch结束后计算精确度、召回率、F1分数和准确率。

该详细流程充分利用了BERT和LSTM/卷积层的优势，同时通过对抗训练提升了模型对输入数据扰动的鲁棒性。

模型参数如下：

数据预处理相关参数：
- max_len：定义输入文本的最大长度。这里设置为128。
- batch_size：数据加载器中的批次大小。在训练和测试数据加载器中均设置为32。

Our Model相关参数：
- hidden_size：BERT模型和LSTM的隐藏层大小，设为768。
- num_labels：分类任务的标签数量，设为3（neutral, positive, negative）。
- dropout：Dropout层的丢弃率，设为0.1。

对抗训练参数：
- epsilon：对抗训练中扰动的大小，设为1e-5。

优化器和损失函数：
- lr：AdamW优化器的学习率，设为5e-5。
- criterion：交叉熵损失函数。

训练参数：
- num_epochs：训练的总轮数，实验中为3。（确实比较少，但是模型太慢了，GPU资源也有限）

# 结果
评估指标见论文介绍
|   Model   | Precision | F1-score | Recall |
| :-------: | :-------: | :------: | :----: |
|  BiLSTM   |  74.90%   |  74.71%  | 74.95% |
| BERT-CNN  |  75.22%   |  75.11%  | 75.15% |
| BERT-wwm  |   7.92%   |  12.36%  | 28.14% |
|  FinBERT  |  78.62%   |  77.55%  | 78.45% |
| Our Model |  85.65%   |  85.65%  | 85.77% |

可以看到比FinBERT提升了7%左右

# 消融实验
评估指标见论文介绍
| Model | Precision| F1-score| Recall|
|:-:|:-:|:-:|:-:|
|FindBERT-RCNN|84.64%|84.57%|84.54%|
|FindBERT-Attack|80.92%|79.36%|80.25%|
|FinBERT|78.62%|77.55%|78.45%|
|Our Model|86.65%|86.12%|86.51%|

可以看到对应的RCNN和Attack部分都有性能贡献

# 优化：

在**RCNN层:**增加了一个卷积层和激活函数，以获取更深层的特征提取能力，最后准确率有了一定的提升：

|   Model   | Precision | F1-score | Recall |
| :-------: | :-------: | :------: | :----: |
| NEW_Model |  92.17%   |  91.83%  | 92.04% |


## 关于目前进度的说明：

- 首先原论文没有给出数据集来源，我们使用的数据集是“三分类”，如果原数据集是二分类的话（没有neutral），Accuracy肯定会更高。FinBERT是研究好的成熟大模型，原始准确率就78.45%的话，基本可以判定是数据集造成的问题。FinBERT论文参考群里扔的**1908.10063v1.pdf**
- 模型参数一定不是最好的，因为已经出现了过拟合的现象
- FinBERT不适用于中文，如果想应用于中文领域，只能用BERT-Chinese
- BERT-wwm效果为什么会这么差有待研究
- 对比模型的代码我没有仔细检查，如果有问题还请指正，理论上我们只需要他们的结果就可以了
- **数据集换了！！！！抛弃原有中文数据集！！！！换英文的！！！！** 



## TODO
1. 伪代码
3. 模型图
4. 如果还有时间的话，优化再想想