<!-- #region -->
<center><img src='./Figs/cs-logo.png' width=200></center>



<h6><center></center></h6>

<h1>
<hr style=" border:none; height:3px;">
<center>Introduction to BERT</center>
<hr style=" border:none; height:3px;">
</h1>
In this lab, we will use BERT model for some NLP tasks.

## Why BERT?

[BERT](https://arxiv.org/abs/1810.04805) (Bidirectional Encoder Representations from Transformers) was first introduced towards the end of 2018, and quickly became a hot topic in NLP. The main reasons are :
- It demonstrates a very sophisticated knowledge of language, achieving human-level performance on certain tasks.
- It can be applied to a variety of tasks.
- It offers the benefits of pre-training and fine-tuning. BERT has been **pre-trained** on a very large text corpus, and we can leverage its understanding of language by taking the pre-trained model and fine-tuning it on your own application (e.g., classification, entity recognition, question answering, etc.). This can allow to achieve highly accurate results on other target tasks with minimal design work.
- It also put into emphasis the benefits of the principle of [self-supervised learning](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html).

## Before we start

In this lab, and given how computationally expensive BERT is, you can't run the experiments on your local machines. So it is recommended to use Mydocker or [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true).


## This lab

In this lab, you will try to get a high-level understanding of BERT. The lab is based on the awesome [Transformers](https://huggingface.co/transformers/) library from huggingface 🤗, and will consist of the following step:


- A small lecture to understand what is a Transformer : [illustrated-transformer](http://jalammar.github.io/illustrated-transformer/)
- Intro to BERT: in this document ([link](intro_to_bert.pdf)), we present a high-level overview of BERT, and go through the necessary components that we need for the rest of
the lab. Please take a bit of time and read the [intro to bert.pdf](intro_to_bert.pdf).
- [BERT lab1](BERT_vocabulary.ipynb): investigating the BERT vocabulary.
- [BERT lab2](sentence_classification.ipynb): Applying BERT to sentence classification with CoLA dataset. The task is to predict if a sentence is grammatically correct or not.
- [BERT lab3](document_classification.ipynb): Applying BERT to sentence classification Wikipedia Personal Attacks. The task is to predict if a comment contains a personal attack or not.
- [BERT lab4](question_answering.ipynb): Applying BERT to question answering. The task if to predict the span of the answer given a reference text containing it.

## Going deeper. Some resources:

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT](https://arxiv.org/abs/1810.04805)
- [RoBERTa](https://arxiv.org/abs/1907.11692)
- [CamemBERT](https://arxiv.org/abs/1911.03894)
- [Contextual Word Representations: A Contextual Introduction](https://arxiv.org/abs/1902.06006)

### Blogs & Tutorials
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [A Visual Guide to Using BERT for the First Time](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Transformers from scratch](http://peterbloem.nl/blog/transformers)
- [Transfer Learning in Natural Language Processing - NAACL 2019](https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/edit?ts=5c8d09e7#slide=id.g5888218f39_364_0)
<!-- #endregion -->
