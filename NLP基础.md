# NLP基础

# 一、基本概念

​	自然语言处理（Natural Language Processing，简称NLP）是人工智能和语言学领域的一个分支，它涉及到计算机和人类（自然）语言之间的相互作用。它的主要目标是让**计算机能够理解、解释和生成人类语言的数据**。NLP结合了计算机科学、人工智能和语言学的技术和理论，旨在填补人与机器之间的交流隔阂。

## 1. 自然语言处理的基本介绍

在定义NLP之前，先了解几个相关概念：

- 语言（Language）：是人类用于沟通的一种结构化系统，可以包括声音、书写符号或手势。

- 自然语言（Natural Language）：是指自然进化中通过使用和重复，无需有意计划或预谋而形成的语言。

- 计算语言学（Computational Linguistics）：是语言学和计算机科学之间的跨学科领域，它包括：

  a.计算机辅助语言学（Computer-aided Linguistics）：利用计算机研究语言的学科

  b.自然语言处理（NLP）：使计算机能够解决以自然语言表达的数据问题的技术

​	NLP的研究范围广泛，包括但不限于**语言理解**（让计算机理解输入的语言）、**语言生成**（让计算机生成人类可以理解的语言）、**机器翻译**（将一种语言翻译成另一种语言）、**情感分析**（分析文本中的情绪倾向）、**语音识别和语音合成**等。

​	在中文环境下，自然语言处理的定义和应用也与英文环境相似，但需要考虑中文的特殊性，如**中文分词、中文语法和语义分析**等，因为中文与英文在语言结构上有很大的不同，这对NLP技术的实现提出了特殊的挑战。自然语言处理使计算机不仅能够理解和解析人类的语言，还能在一定程度上模仿人类的语言使用方式，进行有效的沟通和信息交换。

## 2. NLP常用数据集

- Daily Dialog英文对话经典benchmark数据集;Paper: arxiv.org/abs/1710.0395...;数据集地址: yanran.li/dailydialog.h...;
- WMT-1x翻译数据集，官网: statmt.org/wmt18/transl...;
- 50万中文闲聊数据: drive.google.com/file/d...;
- 日常闲聊数据: github.com/codemayq/chi...;
- 中国古诗词数据集，数据集地址: github.com/congcong0806;

## 3. NLP基础概念

下面总结一些NLP中常用的概念名词，便于理解任务。

​    （1）词表/词库（Vocabulary）：文本数据集中出现的所有单词的集合。

​    （2）语料库（Corpus）：用于NLP任务的文本数据集合，可以是大规模的书籍、文章、网页等。

​    （3）词嵌入（Word Embedding）：将单词映射到低维连续向量空间的技术，用于捕捉单词的语义和语法信息。

​    （4）停用词（Stop Words）：在文本处理中被忽略的常见单词，如"a"、"the"、"is"等，它们通常对文本的意义贡献较 小。

​    （5）分词（Tokenization）：将文本分割成一个个单词或标记的过程，为后续处理提供基本的单位。

​    （6） 词频（Term Frequency）：在给定文档中，某个单词出现的次数。

​    （7）逆文档频率（Inverse Document Frequency）：用于衡量一个单词在整个语料库中的重要性，是将词频取倒数并取 对数的值。

​    （8） TF-IDF（Term Frequency-Inverse Document Frequency）：一种常用的文本特征表示方法，综合考虑了词频和逆文档频率。

​    （9） 词袋模型（Bag of Words）：将文本表示为一个单词的集合，忽略了单词的顺序和语法结构。

   （10）N-gram：连续的N个单词构成的序列，用于捕捉文本中的局部特征和上下文信息。

   （11）**序列**：指的是一个按顺序排列的元素集合。这些元素可以是字符、单词、句子，甚至更抽象的结构。序列的每个元素都有特定的顺序和位置，这意味着它们不能随意重排，否则会影响其意义或功能。

**序列的常见类型**

1. **字符序列**：
   - 一个字符串就是一个字符序列，每个字符按顺序排列。
   - 例子：`"hello"` 是一个由 `h`、`e`、`l`、`l`、`o` 组成的字符序列。
2. **单词序列**：
   - 一句话可以看作是一个单词序列，每个单词按照一定顺序排列。
   - 例子：`"I love NLP"` 是一个由 `I`、`love`、`NLP` 组成的单词序列。
3. **时序数据**：
   - 在时间序列中，元素是按时间顺序排列的，常用于预测问题。
   - 例子：股票价格数据可以看作是随时间变化的数值序列。
4. **语音序列**：
   - 在语音处理任务中，语音信号可以被分解为按时间顺序排列的帧序列（特征向量序列）。
5. **其他序列**：
   - 序列还可以表示一些更抽象的结构，比如DNA序列（由碱基组成的序列）、事件序列等。

------------------------------------------------

## 5. NLP的基本流程

**（1）语料获取**

在进行NLP之前，人们需要得到文本语料。文本语料的获取一般有以下几种方法。
		(1）利用已经建好的数据集或第三方语料库，这样可以省去很多处理成本。

​		(2）获取网络数据。很多时候要解决的是某种特定领域的应用，仅靠开放语料库无法满足需求,这时就需要通过爬虫技术获取需要的信息。

​		(3）与第三方合作获取数据。通过购买的方式获取部分需求文本数据。

**（2）语料预处理**
		(1）去除数据中非文本内容。

​		(2）**中文分词**。常用的中文分词软件有很多，如**jieba**、FoolNLTK、HanLP、THULAC、NLPIR、LTP等。

​		(3）词性标注。词性标注指给词语打上词类标签，如名词、动词、形容词等,常用的词性标注方法有基于规则的算法、基于统计的算法等。

​		(4）去停用词。停用词就是句子中没必要存在的词，去掉停用词后对理解整个句子的语义没有影响。

**（3）文本向量化（特征工程）**

​	文本数据经过预处理去除数据中非文本内容、中文分词、词性标注和去停用词后，基本上是干净的文本了。但此时还是无法直接将文本用于任务计算，需要通过某些处理手段，预先将文本转化为特征向量。一般可以调用一些模型来对文本进行处理，常用的模型有词袋模型(Bag of Words Model）、独热表示、TF-IDF 表示、n元语法(n-gram)模型和 Word2Vec模型等。

**（4）模型构建**
	根据需求选择合适的模型进行构建，备选模型也需准备。模型包括机器学习模型（如SVM、Naive Bayes）和深度学习模型（如TextCNN、RNN、transformer等），复杂模型通常需更长训练时间，但不一定能显著提高精度。

**（5）模型训练**

​	训练过程中需注意过拟合和欠拟合问题，避免梯度消失或爆炸。模型调优是一个复杂且冗长的过程，需权衡精度与泛用性，并应对数据分布变化进行重新训练。

**（6）模型评价**
	模型训练完成后，还需要对模型的效果进行评价。模型的评价指标指主要有准确率(Accuracy)、精确率(Precision)、召回率、F1值、ROC曲线、AUC线等。



# 二、NLP中的特征工程

​	特征是数据中抽取出来的对结果预测有用的信息。

​	在自然语言处理（NLP）中，**特征工程**是指**将文本数据转换为适合机器学习模型使用的数值表示的过程**。文本是一种非结构化数据，机器学习模型无法直接处理，因此必须通过特征工程来提取有用的信息。

​	**通过特征工程能让机器学习到文本数据中的一些特征，比如词性、语法、相似度等**。

## 1. 词向量的引入

​	词向量（Word Vector）是对词语义或含义的数值向量表示，包括字面意义和隐含意义。 词向量可以捕捉到词的内涵，将这些含义结合起来构成一个稠密的浮点数向量，这个稠密向量支持查询和逻辑推理。

​	词向量也称为词嵌入，其英文均可用 Word Embedding，是自然语言处理中的一组语言建模和特征学习技术的统称，其中来自词表的单词或短语被映射为实数的向量，**这些向量能够体现词语之间的语义关系**。从概念上讲，它涉及从每个单词多维的空间到具有更低维度的连续向量空间的数学嵌入。当用作底层输入表示时，单词和短语嵌入已经被证明可以提高 NLP 任务的性能，例如文本分类、命名实体识别、关系抽取等。

​	词嵌入实际上是一类技术，单个词在预定义的向量空间中被表示为实数向量，每个单词都映射到一个向量。举个例子，比如在一个文本中包含“猫”“狗”“爱情”等若干单词，而这若干单词映射到向量空间中，“猫”对应的向量为（0.1 0.2 0.3），“狗”对应的向量为（0.2 0.2 0.4），“爱情”对应的映射为（-0.4 -0.5 -0.2）（本数据仅为示意）。像这种将文本X{x1,x2,x3,x4,x5……xn}映射到多维向量空间Y{y1,y2,y3,y4,y5……yn }，这个映射的过程就叫做词嵌入。

![](https://github.com/ljgit1316/Picture_resource/blob/main/NLP_Pic/13D1498B4DAD6BA24F3EC6C7AF6A0DD6.png)

​	此外，词嵌入还可以做类比，比如：v(“国王”)－v(“男人”)＋v(“女人”)≈v(“女王”)，v(“中国”)＋v(“首都”)≈v(“北京”)，当然还可以进行算法推理。

## 2. 传统NLP中的特征工程

### 2.1 独热编码 one - hot

​	**独热编码（One-Hot Encoding）** 是一种常见的特征表示方法，通常用于将离散的类别型数据转换为数值型表示，以便输入到机器学习模型中。它的特点是将每个类别表示为一个向量，在该向量中，只有一个元素为1，其余元素全部为0。

**One-Hot Encoding 的工作原理**

假设你有一个包含以下类别的分类任务：

- 红色（red）
- 绿色（green）
- 蓝色（blue）

要将这些类别转换为 One-Hot 编码，我们会为每个类别创建一个独特的二进制向量：

- 红色：`[1, 0, 0]`
- 绿色：`[0, 1, 0]`
- 蓝色：`[0, 0, 1]`

例如，如果输入数据是“红色”，在使用 One-Hot 编码后，它将被表示为 `[1, 0, 0]`。

**在NLP当中**

- Time flies like an arrow.
- Fruit flies like a banana.

构成词库{time, fruit, flies, like, a, an, arrow, banana}

banana的one-hot表示就是：[0，0，0，0，0，0，0，1]，"like a banana” 的one-hot表示就是：[0，0，0，1，1，0，0，1]。

**示例**

<img src="https://github.com/ljgit1316/Picture_resource/blob/main/NLP_Pic/8a7e92cfbeb8f16d3b42c5e96378408b.png" style="zoom: 50%;float:left" />

### 2.2 词频-逆文档频率(TF-IDF)

**（1）词频**

​	将文本中的每个单词视为一个特征，并将文本中每个单词的出现次数除以该单词在所有文档中的出现次数，以调整单词的权重。

注意：在计算词频（TF）时，**分母是文档中的总词数**，而不考虑每个词是否重复。这意味着无论词是否重复，分母始终是文档中所有词的数量总和。

<img src="https://github.com/ljgit1316/Picture_resource/blob/main/NLP_Pic/4fbe22df-789b-484d-9866-8541f17499fb.png" style="zoom: 67%;" />

举个例子，如果词 "cat" 在一篇包含 100 个词的文章中出现了 5 次，那么它的词频为：

<img src="https://github.com/ljgit1316/Picture_resource/blob/main/NLP_Pic/25d6d5fa-6753-408b-9d7f-0947486e1a1c.png" style="zoom:67%;" />

- 短语，句子或者文档的词频表示就是其组成单词‘one-hot’表示向量的总和。
- “Fruit flies like time flies a fruit” ，DF表示为：[1,2,2,1,1,0,0,0]，TF表示为：[0.14,0.29,0.29,0.14,0.14,0,0,0]
- 根据词库{time, fruit, flies, like, a, an, arrow, banana}验证上面的表示Fruit flies like a banana.



**（2）逆文档频率（Inverse Document Frequency, IDF）**

​	逆文档频率用来衡量一个词在整个文档集合（语料库）中的重要性。它的目的是降低那些在很多文档中频繁出现的词的权重，例如“the”、“is”这种常见词，或者低频罕见词tetrafluoroethylene(四氟乙烯)。计算公式如下：

<img src="https://github.com/ljgit1316/Picture_resource/blob/main/NLP_Pic/1fe67527-3116-4b14-9e35-816f84ffd977.png" style="zoom:67%;" />

其中，`D` 表示文档集合，`t` 是要计算的词。`+1` 是为了避免分母为 0 的情况。

例如，如果有 1000 篇文档，而词 "cat" 仅在 10 篇文档中出现过，那么它的 IDF 计算如下：

<img src="https://github.com/ljgit1316/Picture_resource/blob/main/NLP_Pic/ada3b3f5-9afb-4917-bfbf-6b062283aa18.png" style="zoom:67%;" />

**（3）TF-IDF 计算**

​	最终，TF-IDF 是将 TF 和 IDF 相乘得出的结果，公式如下：

<img src="https://github.com/ljgit1316/Picture_resource/blob/main/NLP_Pic/ff23ce7d-9065-42eb-acd4-8da919916206.png" style="zoom: 80%;" />

通过这个方法，一个词在特定文档中出现的频率越高（TF高），并且在整个语料库中出现得越少（IDF高），它的 TF-IDF 值就越高。这样可以使模型更加关注那些在某篇文档中特别重要但不常见的词。

**特性：**

<img src="https://github.com/ljgit1316/Picture_resource/blob/main/NLP_Pic/44c70e17-6adb-4cdb-b3e7-3e142e7712e7.png" style="zoom:50%;float:left" />

**结论：**

- 文档频率和样本语义贡献程度呈反相关
- 文档频率和逆文档频率呈反相关
- 逆文档频率和样本语义贡献度呈正相关

### 2.3 n-grams

**n-grams** 是特征工程中的一种技术，它通过将文本中的连续 n 个词（或字符）组合起来，形成一个短语来捕捉文本中的局部上下文信息。n 可以为 1、2、3 等，具体取决于希望捕捉的上下文范围。

**什么是 n-grams？**

- **1-gram（Unigram）**：每个单独的词作为一个单位。例如，"I love NLP" 的 1-gram 是 `["I", "love", "NLP"]`。
- **2-grams（Bigram）**：相邻的两个词组合成一个短语。例如，"I love NLP" 的 2-grams 是 `["I love", "love NLP"]`。
- **3-grams（Trigram）**：相邻的三个词组合成一个短语。例如，"I love NLP" 的 3-grams 是 `["I love NLP"]`。

**n-grams 的作用**

使用 n-grams 可以捕捉词之间的局部上下文关系。例如，1-gram 只关心词的独立出现频率，而 bigram 和 trigram 能捕捉到词之间的顺序关系。例如，bigram `"love NLP"` 表示词 "love" 和 "NLP" 是一起出现的，这种信息在建模中会比仅仅知道 "love" 和 "NLP" 出现频率更有价值。

**n-grams 的示例**

假设句子为 "I love NLP and machine learning"：

- **1-gram**（Unigram）: `["I", "love", "NLP", "and", "machine", "learning"]`
- **2-grams**（Bigram）: `["I love", "love NLP", "NLP and", "and machine", "machine learning"]`
- **3-grams**（Trigram）: `["I love NLP", "love NLP and", "NLP and machine", "and machine learning"]`

通过这些 n-grams，模型可以捕捉到词与词之间的局部依赖关系。

​	将 **n-grams** 与 **TF-IDF** 相结合是文本特征工程中非常常见的做法，它不仅能够捕捉词与词之间的局部关系，还能通过 TF-IDF 来衡量这些短语在整个语料库中的重要性。结合的过程基本上是先生成 n-grams，然后对这些 n-grams 计算 TF-IDF 权重。

**结合 n-grams 与 TF-IDF 的步骤：**

1. **生成 n-grams**：首先从文本中生成 n-grams（n 可以是 1, 2, 3 等）。这些 n-grams 就像是词的组合，通常使用 `CountVectorizer` 或类似的工具生成。
2. **计算词频 (TF)**：统计每个 n-gram 在文本中出现的频率。
3. **计算逆文档频率 (IDF)**：计算 n-gram 在所有文档中出现的频率，稀有的 n-grams 会得到较高的权重，而常见的 n-grams 权重较低。
4. **计算 TF-IDF**：将每个 n-gram 的 TF 和 IDF 相乘，得到 TF-IDF 权重，表示该 n-gram 对特定文本的重要性。

注意：当使用 **2-grams** 时，`I love` 和 `love NLP` 被看作是两个单独的特征，总共有两个特征（总特征数 = 2）。

**举例说明**

假设我们有以下两个文本：

1. `"I love NLP"`
2. `"NLP is fun"`

我们的词汇表为 `{"I", "love", "NLP", "is", "fun"}`，并且我们想要计算 2-grams 的 TF-IDF 值。

**步骤 1：生成 2-grams**

对于 `"I love NLP"`，生成的 2-grams 是：

- `["I love", "love NLP"]`

对于 `"NLP is fun"`，生成的 2-grams 是：

- `["NLP is", "is fun"]`

所以我们的 n-grams 词汇表为 `{"I love", "love NLP", "NLP is", "is fun"}`。

**步骤 2：计算词频（TF）**

在每个文本中，计算每个 2-gram 的出现次数：

- 文本 1 (

  ```
  "I love NLP"
  ```

  ) 的 2-grams 词频：

  - `I love`: 1/2 
  - `love NLP`: 1/2

- 文本 2 (

  ```
  "NLP is fun"
  ```

  ) 的 2-grams 词频：

  - `NLP is`: 1/2
  - `is fun`: 1/2

**步骤 3：计算逆文档频率（IDF）**

<img src="https://github.com/ljgit1316/Picture_resource/blob/main/NLP_Pic/dcaf8966-d560-4172-bbda-4f50499297e7.png" style="zoom:67%;float:left" />

**步骤 4：计算 TF-IDF**

现在，我们计算每个 2-gram 的 TF-IDF 值：

- **文本 1 ("I love NLP")**：
  - `I love` 的 TF = 0.5，IDF = 0.693，所以 TF-IDF(`I love`) = 0.5×0.693=0.3465
  - `love NLP` 的 TF =0.5，IDF = 0.693，所以 TF-IDF(`love NLP`) = 0.5×0.693=0.3465
- **文本 2 ("NLP is fun")**：
  - `NLP is` 的 TF =0.5，IDF = 0.693，所以 TF-IDF(`NLP is`) = 0.5×0.693=0.3465
  - `is fun` 的 TF =0.5，IDF = 0.693，所以 TF-IDF(`is fun`) = 0.5×0.693=0.3465



**传统NLP中的特征工程缺点**

<img src="https://github.com/ljgit1316/Picture_resource/blob/main/NLP_Pic/efc1cafe-491d-40bd-a55b-5e2d2202ec0c.png" style="zoom: 50%;" />

## **3. 深度学习中NLP的特征输入**

​	深度学习使用分布式单词表示技术（也称**词嵌入**表示)，通过查看所使用的单词的周围单词(即上下文)来学习单词表示。这种表示方式将词表示为一个粘稠的序列，在保留词**上下文信息**同时，避免维度过大导致的计算困难。

### 3.1 稠密编码（特征嵌入)

​	稠密编码（Dense Encoding），在机器学习和深度学习中，**通常指的是将离散或高维稀疏数据转化为低维的连续、密集向量表示**。这种编码方式在特征嵌入（Feature Embedding）中尤为常见。

​	稠密向量表示：不再以one-hot中的一维来表示各个特征，而是把每个核心特征（词，词性，位置等)都嵌入到d维空间中，并用空间中的一个向量表示。通常空间维度d远小于每个特征的样本数(40000的词表，100/200维向量)。嵌入的向量(每个核心特征的向量表示)作为网络参数与神经网络中的其他参数一起被训练。

**特征嵌入（Feature Embedding）**

​	特征嵌入，也成为词嵌入，是稠密编码的一种表现形式，目的是将离散的类别、对象或其他类型的特征映射到一个连续的向量空间。通过这种方式，嵌入后的向量可以捕捉不同特征之间的语义关系，并且便于在后续的机器学习模型中使用。

**特点：**

- **低维度**：相比稀疏表示（如独热编码），稠密编码的维度更低，能够减少计算和存储成本。
- **语义相似性**：嵌入向量之间的距离（如欧氏距离或余弦相似度）可以表示这些对象之间的语义相似性。
- **可微学习**：嵌入表示通常通过神经网络进行学习，并且通过反向传播算法进行优化。

### 3.2 词嵌入算法

#### **3.2.1 Embedding Layer**  

- 定义

  ​		Embedding Layer是与特定自然语言处理上的神经网络模型联合学习的单词嵌入。该嵌入方法将清理好的文本中的单词进行one hot编码（独热编码），向量空间的大小或维度被指定为模型的一部分，例如50、100或200维。向量以小的随机数进行初始化。Embedding Layer用于神经网络的前端，并采用反向传播算法进行监督。

- 实现流程

  已知一句话的前几个字，预测下一个字是什么，于是有了**NNLM 语言模型**搭建的网络结构图：

  <img src="https://github.com/ljgit1316/Picture_resource/blob/main/NLP_Pic/12b12316-da8d-47e9-bce7-65dbc633ce63.png" style="zoom: 50%;" />

  具体怎么实施呢？先用最简单的方法来表示每个词，one-hot 表示为︰

  **dog=(0,0,0,0,1,0,0,0,0,...)；cat=(0,0,0,0,0,0,0,1,0,...) ；eat=(0,1,0,0,0,0,0,0,0,...)**

  <img src="https://github.com/ljgit1316/Picture_resource/blob/main/NLP_Pic/aad7055d-fe44-4bcd-aeb7-89e1cadfab59.png" style="zoom: 50%;" />

  可是 one-hot 表示法有诸多的缺陷，还是稠密的向量表示更好一些，那么怎么转换呢？加一个**矩阵映射**一下就好！

  <img src="https://github.com/ljgit1316/Picture_resource/blob/main/NLP_Pic/5e92abe8-9ab3-4d7b-a8c4-0607b68f0850.png" style="zoom:50%;" />

  映射之后的向量层如果单独拿出来看，还有办法找到原本的词是什么吗？

  One-hot表示法这时候就作为一个索引字典了，可以通过映射矩阵对应到具体的词向量。

  ![](https://github.com/ljgit1316/Picture_resource/blob/main/NLP_Pic/438c41de-6b51-4d9d-b664-c6f0375bfeac.png)

##### 词嵌入层的使用

​	词嵌入层首先会根据输入的词的数量构建一个词向量矩阵，例如: 我们有 100 个词，每个词希望转换成 128 维度的向量，那么构建的矩阵形状即为: 100*128，输入的每个词都对应了一个该矩阵中的一个向量。

在 PyTorch 中，我们可以使用 **nn.Embedding** 词嵌入层来实现输入词的向量化。接下来，我们将会学习如何将词转换为词向量，其步骤如下:

1. 先将语料进行分词，构建词与索引的映射，我们可以把这个映射叫做词表，词表中每个词都对应了一个唯一的索引；
2. 然后使用 nn.Embedding 构建词嵌入矩阵，词索引对应的向量即为该词对应的数值化后的向量表示。

例如，我们的文本数据为: "北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。"，接下来，我们看下如何使用词嵌入层将其进行转换为向量表示，步骤如下：

1. 首先，将文本进行分词；
2. 然后，根据词构建词表；
3. 最后，使用嵌入层将文本转换为向量表示。

nn.Embedding 对象构建时，最主要有两个参数:

1. num_embeddings 表示词的数量
2. embedding_dim 表示用多少维的向量来表示每个词

```
nn.Embedding(num_embeddings=10, embedding_dim=4)
```

接下来，我们就实现下刚才的需求：

```python
import torch
import torch.nn as nn
import jieba

txt = '今天天气非常好，不如我们去旅游吧，这样更能放松你的心情'

# 1.文本分词
words = jieba.lcut(txt)
# print(words)

# 2.构建词表
idex_to_word = {}  # 存储索引对应的词语
word_to_idex = {}  # 存储词语对应的索引
unique_words = list(set(words))  # 去重
for idx, word in enumerate(unique_words):
    idex_to_word[idx] = word
    word_to_idex[word] = idx
# print(idex_to_word)
# print(word_to_idex)

# 3.词嵌入层
# 第一个参数是词表长度，第二个参数是词向量维度
embed = nn.Embedding(len(word_to_idex), 5)

# 4.将文本转成词向量表示
for word in words:
    idx = word_to_idex[word]
    word_vector = embed(torch.tensor([idx]))
    print(word_vector)
    
    
"""
['今天天气', '非常', '好', '，', '不如', '我们', '去', '旅游', '吧', '，', '这样', '更能', '放松', '你', '的', '心情']


tensor([[ 0.5621, -0.9307,  1.2265,  0.4046, -1.4312]],
       grad_fn=<EmbeddingBackward0>)
tensor([[0.7797, 0.2190, 1.5007, 0.5279, 0.5697]],
       grad_fn=<EmbeddingBackward0>)
tensor([[ 0.0063,  1.0967, -0.3440, -1.2293, -0.5154]],
       grad_fn=<EmbeddingBackward0>)
tensor([[ 0.2250,  0.0096, -0.4750, -1.9148, -0.9661]],
       grad_fn=<EmbeddingBackward0>)
tensor([[ 0.5041,  0.0884, -2.1984, -0.4958,  0.6268]],
       grad_fn=<EmbeddingBackward0>)
tensor([[-0.1371,  0.7041,  0.3322,  0.4439, -0.5457]],
       grad_fn=<EmbeddingBackward0>)
tensor([[-0.1197, -0.7853, -0.1545,  0.4717,  1.1422]],
       grad_fn=<EmbeddingBackward0>)
tensor([[-1.1719,  0.0518,  0.2756,  0.8014,  0.3466]],
       grad_fn=<EmbeddingBackward0>)
tensor([[-1.0846,  0.8710, -1.1377, -0.7931, -0.9636]],
       grad_fn=<EmbeddingBackward0>)
tensor([[ 0.2250,  0.0096, -0.4750, -1.9148, -0.9661]],
       grad_fn=<EmbeddingBackward0>)
tensor([[ 0.9522,  1.1583,  0.6976, -0.7136,  0.9717]],
       grad_fn=<EmbeddingBackward0>)
tensor([[0.3139, 1.3496, 0.8864, 0.1544, 0.4284]],
       grad_fn=<EmbeddingBackward0>)
tensor([[-0.9089,  1.4530, -1.6523,  1.4659,  1.0345]],
       grad_fn=<EmbeddingBackward0>)
tensor([[-0.2186,  0.1666, -0.7460, -1.6811,  0.3335]],
       grad_fn=<EmbeddingBackward0>)
tensor([[-1.4077,  1.4323,  0.4079,  2.0198, -0.0598]],
       grad_fn=<EmbeddingBackward0>)
tensor([[-0.4625, -0.1678, -2.0173, -0.0423, -1.4953]],
       grad_fn=<EmbeddingBackward0>)
进程已结束,退出代码0
"""
```

**归纳：**

- **Embedding类的本质**是一个大小为 `(num_embeddings, embedding_dim)` 的矩阵，每一行是某个词汇的嵌入向量。

<img src="https://github.com/ljgit1316/Picture_resource/blob/main/NLP_Pic/7203a708-d83b-4f08-b6d2-a8871d705880.png" style="zoom: 80%;" />

- **通过索引**，可以高效地从这个矩阵中提取对应词汇的向量表示，因为 `nn.Embedding` 在内部通过索引直接查找矩阵中的行，这种操作非常快速且方便。

##### 词嵌入层**实例练习**

假设现在有语料库sentences = ["i like dog", "i love coffee", "i hate milk", "i do nlp"] 通过**词嵌入层算法**和**NNLM模型**得到以下结果

**[['i', 'like'], ['i', 'love'], ['i', 'hate'], ['i', 'do']] -> ['dog', 'coffee', 'milk', 'nlp']**

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sentences = ["i like dog", "i love coffee", "i hate milk", "i do nlp"]
# 创建词表和对应的映射关系
word_list = " ".join(sentences).split()
# 去重操作
word_list = list(set(word_list))
# 建立映射关系
word_dict = {w: i for i, w in enumerate(word_list)}
num_dic = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)  # 词表大小,分类数量
# 设置模型超参数
n_dim = 2  # 词向量维度
n_hidden_neuron = 2  # 神经元的个数
n_step = 2  # 步长数，一个词的包含的词数量


# 创建输入的样本和目标值
def make_batch(sentences):
    input_batch = []
    target_batch = []
    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]
        input_batch.append(input)
        target_batch.append(target)
    return input_batch, target_batch


# 创建模型
class NNLM(nn.Module):
    def __init__(self, n_step, n_class, n_dim, n_hidden_neuron):
        super(NNLM, self).__init__()
        # 定义嵌入层，单词索引映射为嵌入向量
        self.embed = nn.Embedding(n_class, n_dim)
        self.linear1 = nn.Linear(n_step * n_dim, n_hidden_neuron)
        self.linear2 = nn.Linear(n_hidden_neuron, n_class)

    def forward(self, x):
        # 通过嵌入层得到的形状是 (batch_size, n_step,n_dim)->(batch_size, n_step * n_dim)
        x = self.embed(x)
        x = self.linear1(x.view(-1, x.size(1) * x.size(2)))
        x = torch.tanh(x)
        x = self.linear2(x)
        return x


# 初始化模型
model = NNLM(n_step, n_class, n_dim, n_hidden_neuron)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 准备输入和目标数据
input_batch, target_batch = make_batch(sentences)
input_batch = torch.LongTensor(input_batch)
target_batch = torch.LongTensor(target_batch)

# 开始训练
for epoch in tqdm(range(5000)):
    optimizer.zero_grad()
    output = model(input_batch)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()

# 预测
predict = model(input_batch).max(1, keepdim=True)[1]

print([sen.split()[:2] for sen in sentences], '->', [num_dic[n.item()] for n in predict.squeeze()])
"""
[['i', 'like'], ['i', 'love'], ['i', 'hate'], ['i', 'do']] -> ['dog', 'coffee', 'milk', 'nlp']
"""
```

**说明：**

```python
(1) predict = model(input_batch).max(1, keepdim=True)[1]
```

1. **`model(input_batch)`**：

- 这部分是将 `input_batch` 输入到模型中，经过 `forward` 函数计算，得到网络的输出。
- 输出是一个二维张量，形状为 `(batch_size, n_class)`，其中每行代表一个样本的预测得分（未归一化的类别概率），每列对应词汇表中的一个单词类别。

   2.**`.max(1, keepdim=True)`**：

- `max` 是一个 PyTorch 的函数，返回沿指定维度的最大值及其索引。
- **`1`**：表示在第 1 维（即每行）上寻找最大值。第 0 维是批次（样本），第 1 维是类别的得分（对于每个单词的概率分布）。
- **`keepdim=True`**：表示保持原始维度，即结果张量的形状与原来的张量在没有被压缩的维度上保持一致。

返回的结果是一个包含两个元素的元组：

- **第一个元素**：每行的最大值（在预测任务中通常是没有用到的）。
- **第二个元素**：每行最大值的索引（即预测的类别）。

  3.**`[1]`**：

- 通过 `[1]` 取出最大值对应的索引，即上一步中返回的最大值的索引部分。这个索引对应预测的类别编号，也就是词汇表中的单词索引。

```python

(2) [number_dict[n.item()] for n in predict.squeeze()]
```

1. **`predict.squeeze()`**：

- `predict` 是模型预测的结果，形状是 `(batch_size, 1)`。`predict.squeeze()` 用于去掉维度为 1 的那一维，使其从 `(batch_size, 1)` 变为 `(batch_size)`。

- `squeeze()` 函数会去掉张量中所有大小为 1 的维度。在这里，`predict` 的维度是 `(batch_size, 1)`，经过 `squeeze()` 后变成一维张量，形状为 `(batch_size)`，即包含 `batch_size` 个预测的类别索引。

   2.**`n.item()`**：

- `n` 是张量中的一个元素（预测的类别索引），`n.item()` 将这个单元素张量转换为普通的 Python 标量类型（整数）。

- `item()` 是 PyTorch 张量的方法，用于从单个元素的张量中提取值。它确保索引 `n` 是一个纯 Python 的整数，可以用于字典查找。

  3.**`number_dict[n.item()]`**：

- `number_dict` 是一个字典，它将单词索引映射回对应的单词，即 `{index: word}` 的映射。

- `number_dict[n.item()]` 用来通过索引 `n.item()` 从字典 `number_dict` 中查找对应的单词，将模型预测出的索引转换为实际的单词。

神经语言模型构建完成之后，就是训练参数了，这里的参数包括：

- **词向量矩阵C；**
- 神经网络的权重；
- 偏置等参数

##### 预训练模型

在自然语言处理（NLP）中，预训练模型是指在大量通用文本数据集上训练好的模型，**它可以捕捉到语言的广泛特征**，例如**词汇、句法、语境相关性**等。这些模型训练完成后，可以在特定的下游任务上进行微调（fine-tuning），以适应特定的应用场景，如情感分析、问答系统、文档摘要等。通过这种方式，**预训练模型可以显著提高NLP任务的性能，尤其是在标注数据有限的情况下。**

#### 3.2.2 word2vec

- 定义

  word2vec是一种高效训练词向量的模型，基本出发点是上下文相似的两个词，它们的词向量也应该相似，比如香蕉和梨在句子中可能经常出现在相同的上下文中，因此这两个词的表示向量应该就比较相似。 

- 分类

  word2vec一般分为CBOW（Continuous Bag-of-Words）与 Skip-Gram 两种模型：

   1、CBOW：根据**中心词周围的词来预测中心词**，有negative sample和Huffman两种加速算法；

   2、Skip-Gram：根据**中心词来预测周围词**；

##### 3.2.2.1 Skip-gram 模型

- 定义

  Skip-gram 模型是一种根据目标单词来预测上下文单词的模型。具体来说，给定一个中心单词，Skip-gram 模型的任务是预测在它周围窗口大小为 n 内的上下文单词。

  **Skip-gram 模型在处理大规模语料库时效果比 CBOW 模型更好**。

  ![](https://github.com/ljgit1316/Picture_resource/blob/main/NLP_Pic/fbf43b71-4fcf-4ce2-a23d-016b5c712d05.png)

- 代码实现

  ```python
  import torch
  import torch.nn as nn
  import numpy as np
  from tqdm import tqdm
  import matplotlib.pyplot as plt
  
  # 语料库，包含训练模型的句子
  sentences = ["i like dog", "i like cat", "i like animal",
               "dog cat animal", "apple cat dog like", "cat like fish",
               "dog like meat", "i like apple", "i hate apple",
               "i like movie book music apple", "dog like bark", "dog friend cat"]
  
  # 定义数据类型
  dtype = torch.FloatTensor
  
  # 拆分单词
  word_list = " ".join(sentences).split()
  # 去重
  word_no_repeat = list(set(word_list))
  # 建立单词索引
  word_to_indx = {w: i for i, w in enumerate(word_list)}
  indx_to_word = {i: w for i, w in enumerate(word_list)}
  # 词表大小
  word_len = len(word_list)
  
  # 创建skip-gram数据集
  skip_grams = []
  for i in range(1, len(word_list) - 1):
      center = word_to_indx[word_list[i]]
      context = [word_to_indx[word_list[i - 1]], word_to_indx[word_list[i + 1]]]
      # 将中心词和上下文词组合成skip-gram
      for w in context:
          skip_grams.append([center, w])
  
  # 定义超参数
  embedding_size = 2  # 词嵌入维度
  batch_size = 2  # 训练批次大小
  
  
  # 定义模型
  class SkipGram(nn.Module):
      def __init__(self, vocab_size, embedding_size):
          super(SkipGram, self).__init__()
          # 定义词嵌入层矩阵W，随机初始化，大小为（词汇表大小，词嵌入维度）
          self.W = nn.Parameter(torch.rand(vocab_size, embedding_size)).type(dtype)
          # 定义输出层矩阵W2，随机初始化，大小为（词嵌入维度，词汇表大小）
          self.W2 = nn.Parameter(torch.rand(embedding_size, vocab_size)).type(dtype)
  
      def forward(self, x):
          weight = torch.matmul(x, self.W)
          out_put = torch.matmul(weight, self.W2)
          return out_put
  
  
  # 定义模型
  model = SkipGram(word_len, embedding_size)
  # 定义损失函数和优化器
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  
  
  # 准备输入和目标数据，随机批量生成数据
  def random_batch(data, size):
      random_label = []
      random_input = []
      # 随机选择size个数据
      random_indx = np.random.choice(range(len(data)), size, replace=False)
      # 根据随机索引生成输入和标签
      for i in random_indx:
          random_input.append(np.eye(word_len)[data[i][0]])
          random_label.append(data[i][1])
      return random_input, random_label
  
  
  # 训练模型
  for epoch in tqdm(range(10000)):
      input, label = random_batch(skip_grams, batch_size)
      input = np.array(input)
      input_batch = torch.Tensor(input)
      label_batch = torch.LongTensor(label)
  
      # 梯度清零
      optimizer.zero_grad()
      # 前向传播
      output = model(input_batch)
      # 计算损失
      loss = criterion(output, label_batch)
      if epoch % 1000 == 0:
          print("Epoch:", epoch, "Loss:", loss.item())
      # 反向传播
      loss.backward()
      # 更新参数
      optimizer.step()
  
  cont=0
  # 可视化
  for i, label in enumerate(word_list):
      W, W2 = model.parameters()
      x, y = float(W[i][0]), float(W[i][1])
      plt.scatter(x, y)
      plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
      break
  plt.show()
  
  ```

##### 3.2.2.2 CBOW模型

- 定义

  连续词袋模型（CBOW）是一种根据上下文单词来预测目标单词的模型。具体来说，给定一个窗口大小为 n 的上下文单词序列，连续词袋模型的任务是**预测中间的目标单词**。

  CBOW模型∶使用文本的中间词作为目标词（标签），**去掉了隐藏层。用上下文各词的词向量的均值**代替NNLM拼接起来的词向量。

  <img src="https://github.com/ljgit1316/Picture_resource/blob/main/NLP_Pic/a3079555-6a27-4d80-bbc8-b7576cb4a226.png" style="zoom: 50%;" />

  **CBOW对小型数据库比较合适。**

  **输入∶上下文的语义表示**

  **输出∶中间词是哪个词**

- 代码实现

  ```python
  import torch
  import torch.nn as nn
  import numpy as np
  from tqdm import tqdm
  import matplotlib.pyplot as plt
  
  # 语料库，包含训练模型的句子
  sentences = ["i like dog", "i like cat", "i like animal",
               "dog cat animal", "apple cat dog like", "cat like fish",
               "dog like meat", "i like apple", "i hate apple",
               "i like movie book music apple", "dog like bark", "dog friend cat"]
  
  # 定义数据类型
  dtype = torch.FloatTensor
  
  # 拆分单词
  word_list = " ".join(sentences).split()
  # 去重
  word_no_repeat = list(set(word_list))
  # 建立单词索引
  word_to_indx = {w: i for i, w in enumerate(word_no_repeat)}
  indx_to_word = {i: w for i, w in enumerate(word_no_repeat)}
  # 词表大小
  word_len = len(word_list)
  
  # 创建CBOW数据集
  cbow_data = []
  for i in range(1, len(word_list) - 1):
      context = [word_to_indx[word_list[i - 1]], word_to_indx[word_list[i + 1]]]
      target = word_to_indx[word_list[i]]
      cbow_data.append([context, target])
  # 定义超参数
  embedding_size = 2
  batch_size = 2
  
  
  # 定义cbow模型
  class CBOW(nn.Module):
      def __init__(self, vocab_size, embedding_size):
          super(CBOW, self).__init__()
          self.embedding = nn.Embedding(vocab_size, embedding_size)
          self.linear = nn.Linear(embedding_size, vocab_size)
  
      def forward(self, x):
          x = self.embedding(x)  # x:(batch_size, n_step, n_dim)
          # 上下文词向量求加权平均,在n_step维度求平均
          x = torch.mean(x, dim=1)
          x = self.linear(x)
          return x
  
  
  # 定义模型
  model = CBOW(word_len, embedding_size)
  # 定义损失函数和优化器
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  
  
  # 定义随机批量生成函数
  def random_batch(data, size):
      random_inputs = []
      random_labels = []
      random_index = np.random.choice(range(len(data)), size, replace=False)
      for i in random_index:
          random_inputs.append(data[i][0])
          random_labels.append(data[i][1])
      return random_inputs, random_labels
  
  
  # 开始训练
  for epoch in tqdm(range(10000)):
      # 随机批量生成
      input, label = random_batch(cbow_data, batch_size)
      # 输入数据类型转换
      input_batch = torch.LongTensor(input)
      label_batch = torch.LongTensor(label)
      # 梯度清零
      optimizer.zero_grad()
      # 前向传播
      output = model(input_batch)
      # 计算损失
      loss = criterion(output, label_batch)
      if epoch % 1000 == 0:
          print("Epoch:", epoch, "Loss:", loss.item())
      # 反向传播
      loss.backward()
      # 更新参数
      optimizer.step()
  
  
  # 可视化
  for i,label in enumerate(word_to_indx):
      w=model.embedding.weight.data.numpy()
      x, y = float(w[i][0]), float(w[i][1])
      plt.scatter(x, y)
      plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
  plt.show()
  ```

##### 3.2.2.3  gensim  API调用

- 定义

  Word2vec是一个用来产生词向量的模型。是一个将单词转换成向量形式的工具。通过转换，可以把对[文本内容](https://so.csdn.net/so/search?q=文本内容&spm=1001.2101.3001.7020)的处理简化为向量空间中的向量运算，计算出向量空间上的相似度，来表示文本语义上的相似度。

- 参数说明

  | 参数            | 说明                                                         |
  | --------------- | ------------------------------------------------------------ |
  | **sentences**   | **可以是一个list，对于大语料集，建议使用BrownCorpus,Text8Corpus或lineSentence构建。** |
  | **vector_size** | **word向量的维度，默认为100。大的size需要更多的训练数据，但是效果会更好。推荐值为几十到几百。** |
  | alpha           | 学习率                                                       |
  | **window**      | **表示当前词与预测词在一个句子中的最大距离是多少。**         |
  | **min_count**   | **可以对字典做截断。词频少于min_count次数的单词会被丢弃掉，默认值为5。** |
  | max_vocab_size  | 设置词向量构建期间的RAM限制。如果所有独立单词个数超过这个，则就消除掉其中最不频繁的一个。每一千万个单词需要大约1GB的RAM。设置成None则没有限制。 |
  | sample          | 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0，1e-5)  |
  | seed            | 用于随机数发生器。与初始化词向量有关。                       |
  | workers         | 参数控制训练的并行数。                                       |
  | **sg**          | **用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。** |
  | hs              | 如果为1则会采用hierarchica·softmax技巧。如果设置为0（default），则negative sampling会被使用。 |
  | negative        | 如果>0，则会采用negative samping，用于设置多少个noise words。 |
  | cbow_mean       | 如果为0，则采用上下文词向量的和，如果为1（default）则采用均值。只有使用CBOW的时候才起作用。 |
  | hashfxn         | hash函数来初始化权重。默认使用python的hash函数。             |
  | epochs          | 迭代次数，默认为5。                                          |
  | trim_rule       | 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。可以设置为None（min_count会被使用）或者一个接受()并返回RULE_DISCARD，utils。RULE_KEEP或者utils。RULE_DEFAULT的函数。 |
  | sorted_vocab    | 如果为1（default），则在分配word index 的时候会先对单词基于频率降序排序。 |
  | batch_words     | 每一批的传递给线程的单词的数量，默认为10000                  |
  | min_alpha       | 随着训练的进行，学习率线性下降到min_alpha                    |

- 常用方法

  `model.wv`: 这个对象包含了所有单词的词嵌入向量。常用的方法有：

  - `model.wv[word]`：返回某个特定单词的向量。

  - `model.wv.most_similar(word)`：获取与某个单词最相似的词。

  - `model.wv.similarity(word1, word2)`：计算两个词之间的相似度。

  - model.save("word2vec.model")

    model = Word2Vec.load("word2vec.model")

    保存和加载模型

    <img src="https://github.com/ljgit1316/Picture_resource/blob/main/NLP_Pic/e0906ae4-57ea-4783-a13f-52c5f8b4e78b.png" style="zoom:50%;float:left" />

- 代码示例

  ```python
  import numpy as np
  from gensim.models import Word2Vec
  from matplotlib import pyplot as plt
  
  # 语料库，包含训练模型的句子
  sentences = ["i like dog", "i like cat", "i like animal",
               "dog cat animal", "apple cat dog like", "cat like fish",
               "dog like meat", "i like apple", "i hate apple",
               "i like movie book music apple", "dog like bark", "dog friend cat"]
  # 每个句子分成单词(word2vec api 要求输入的语料库为二维列表[['i','like','dog'],['i','like','cat']])
  token_list = [s.split() for s in sentences]
  print(token_list)
  # 定义word2vec模型
  model = Word2Vec(token_list, vector_size=2, window=1, min_count=0, sg=0)
  
  # 获取词汇表
  word_list = list(model.wv.index_to_key)
  print(word_list)
  # 可视化嵌入结果
  for i, label in enumerate(word_list):
      w = model.wv
      x, y = float(w[i][0]), float(w[i][1])
      plt.scatter(x, y)
      plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
  # plt.show()
  
  print(model.wv.most_similar("like"))
  print(model.wv.similarity("i", 'like'))
  ```

- 实例练习

  ```python
  import numpy as np
  from gensim.models import Word2Vec
  
  """
  # 示例文本数据
  sentence = "Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers."
  
  需要把这句话通过gensim  API调用的方法，转成句向量
  """
  sentence = "Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing (NLP) where words or phrases from the vocabulary are mapped to vectors of real numbers."
  # 拆分词汇
  token_list = [sentence.split()]
  # 定义w2v模型
  model = Word2Vec(token_list, vector_size=2, min_count=0, window=4, sg=0)
  # 加权平均计算句子向量
  def sentence_vector(sentence, model):
      word_list = sentence.split()
      # 获取每个词向量（空间位置）
      word_vector = [model.wv[word] for word in word_list]
      print(word_vector)
      return np.mean(word_vector, axis=0)
  
  
  print(sentence_vector(sentence, model))
  
  ```

  

