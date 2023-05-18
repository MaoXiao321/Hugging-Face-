# Hugging Face教程

环境配置
```
conda create -n transformer python=3.10 # 3.7以上
conda install -c conda-forge huggingface_hub
pip install datasets evaluate transformers[sentencepiece]
pip install torch
```

# chapter1
## Section1: pipline支持的task
模型库：https://huggingface.co/models
>feature-extraction：特征抽取，将文本转化为一个向量，这就是embedding呀<br>
>fill-mask：完形填空，BERT, BART等模型的预训练策略<br>
>ner (named entity recognition)：命名实体识别<br>
>question-answering：问答，一般是抽取式问答<br>
>sentiment-analysis：情感分析，也是文本分类<br>
>summarization：摘要，一般是抽取式摘要text-generation：文本生成，一般指GPT类模型<br>
>translation：翻译，seq2seq。<br>
>zero-shot-classification：零样本分类<br>
## Section2: 模型受限的情况
使用这些模型的时候，需要注意这些模型可能存在种族歧视、性别歧视或同性恋问题

# chapter2
如何使用tokenzier和预训练模型得到一个或多个文本的输出
## Section2: pipline的内部结构（前处理、model、后处理）
（1）第一步，用Tokenizer将输入转成模型可处理的格式<br>
（2）第二步，输入经过模型得到输出<br>
在Transformers库中，有许多模型架构，他们一般有基础Transformer架构加上不同的head模块组成，例如：<br>
>*Model (retrieve the hidden states)：只输出隐状态<br>
*ForCausalLM：常规语言模型，典型的有GPT系列<br>
*ForMaskedLM：掩码语言模型，典型的有BERT、RoBERTa、DeBERTa<br>
*ForMultipleChoice：多项选择模型<br>
*ForQuestionAnswering：问答模型，一般是抽取式问答<br>
*ForSequenceClassification：序列分类模型<br>
*ForTokenClassification：token分类模型，如命名实体识别和关系抽取<br>
（3）第三步，后处理模型输出，得到概率值<br>
## Section3:模型的创建、加载、保存、推理
```
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
```
## Section4: 介绍Tokenizer的使用。介绍预处理中的分词、token-id
```
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```
## Section5: 介绍批处理、补齐、注意力掩码、截断
## Section6: 用Tokenizer搞定预处理pipline，介绍相关参数设置
```
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, 
padding=True, # 填充
truncation=True, # 截断
return_tensors="pt" # 指定张量结果
)
output = model(**tokens)
```

# chapter3
微调transformer预训练模型(以文本分类为例)。Hugging Face的datasets Hub：https://huggingface.co/datasets
## Section2: 数据集加载与批处理
## Section3: 用Trainer类完成一个模型的微调和评估，Accelerate库做分布式
## Section4: 使用原生pytorch完成一个模型的微调和评估

# chapter4
使用模型库中的各种预训练模型。把自己的模型上传到hub。模型库：https://huggingface.co/models

# chapter5
用datasets加载本地文件和远端数据。
## Sectio2: datasets加载本地文件和远端数据
## Sectio3: 数据集预处理、数据集划分、保存与加载
## Sectio4: datasets加载大数据
## Sectio5: 创建自己的datasets，提交到hub
## Sectio6: 基于FAISS的语义检索引擎

# chapter6
微调tokenizer

# chapter7
主流NLP任务模型微调
## Sectio2: 微调Token分类（命名实体识别、词性标注、分块）模型
## Sectio3: 微调掩码语言模型
## Sectio4: 微调文本翻译（序列到序列）模型
## Sectio5: 微调生成式文本摘要（序列到序列）模型
## Sectio6: 从头训练GPT类语言模型
