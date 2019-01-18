# Resources
* [arXiv](https://arxiv.org/abs/1411.4555)
* [Youtube, TensorFlow KR 논문읽기 모임](https://www.youtube.com/watch?v=BrmCnoYhQb4&t=0s&index=42&list=PL0oFI08O71gKjGhaWctTPvvM7_cVzsAtK)
* [Stanford University CS231n_1, Spring 2017, Lecture 10 | Recurrent Neural Networks](https://youtu.be/6niqTuYFZLQ?t=3347)


# Abstract

* describing the content of an image is a fundamental problem in artificial intelligence that **connects computer vision and natural language processing**.
* a generaive model based on a **deep recurrent architecture** that ... to generate natural sentences describing an image.

# 1. Introduction

* a description must **capture** not only **the objects** contained in an image, but it also must express **how these objects relate >to each other** as well as their **attributes and the activities** they are involved in.
* To express the above semantic knowledge, **a language model** is needed in addition to visual understanding.

* a single joint model that takes an image I as input, and is trained to maximize the likelihood p(S|I) of producing a target sequence of words


<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/intro_0.PNG?raw=true" width="50%" height="50%">
* An “encoder” RNN reads the source sentence and transforms it into a rich fixed-length vector representation, which in turn in used as the initial hidden state of a “decoder” RNN that generates the target sentence.

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/intro_1.PNG?raw=true" width="50%" height="50%">
* replacing the encoder RNN by a deep convolution neural network(CNN). CNNs can produce a rich representation of the input images by embedding it to a fixed-length vector which can be used by another tasks.
* use last hidden layer as an input to the RNN decoder that generates sentences.
* We call this model the Neural Image Capion, or NIC.

### Contributions:
1) end-to-end system for the problem.
2) combines state-of-art sub-networks for vision and language models.
3) yields significantly better performance compared to state-of-the-art approaches
  \* Pascal dataset(BLEU score): 25 to 59 (human performance is 69), Flickr30k: 56 to 66, SBU: 19 to 28)
  
  
# 2. Related Work
1) Mainly for video. systems composed of visual primitive recognizers combine with structured formal language.
    \- heavily hand-designed, relatively brittle and have been demonstrated only limited domain.
2) Systems dealing with image description were made after some advances in recognition of objects.
    \- These are also limited in their expressivity.
3) The idea of co-embedding of images and text in the same vector space. Descriptions are retrieved which lie close to the image in the embedding space.
    \- do not attempt to generate novel descriptions.
  * the above approaches cannot describe previously unseen compositions of objects, even though the individual objects might have been observed in the training data.
4) Simillar recurrent NN for was introduced.  These networks use sentences as RNN input whereas Show and Tell use the visual input to the RNN model directly.
  * As a result of these seemingly insignificant differences, our system achieves substantially better results on the established benchmarks.

# 3. Model

> Machine translation models make use of a recurrent neural network which **encodes the variable length input** into a fixed dimensional vector, and uses this representation to **“decode” it to the desired output sentence.** <br />
> Thus, it is natural to use the same approach where, given **an image** (instead of an input sentence in the source language), one applies the same principle of “translating” it into its description.

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/1.PNG?raw=true" width="50%" height="50%">
<br />

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/theta.PNG?raw=true" width="1%" height="1%"> : parameters of our model  
_I_ : image  
_S_ : correct transcription (unbounded length)

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/2.PNG?raw=true" width="50%" height="50%">

* It is common to apply the chain rule to model the joint probability over S_0, ... S_N, where N is the length of this particular example.
* we **optimize the sum of the log probabilities as described in (2)** over the whole training set using **stochastic gradient descent.**

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/3.PNG?raw=true" width="50%" height="50%">
RNN은 위와 같은 구조를 가지고 있기 때문에 (2)와 같은 상황을 다루기에 적합한 모델이다.
NIC에서는 LSTM을 사용하였다. images의 representation을 위해서 CNN을 사용하였다. 그리고 단어들은 임베딩 모델로 represent 되었다.



### 3.1 LSTM-based Sentence Generator

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/4.PNG?raw=true" width="50%" height="50%">
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/5.PNG?raw=true" width="50%" height="50%">
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/LSTM_cs231n.PNG?raw=true" width="50%" height="50%">
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/LSTM_cs231n.PNG?raw=true" width="50%" height="50%">
\* LSTM에 대한 구체적인 설명은 cs231n 링크로 대체하겠습니다. <br />[Stanford University CS231n_1, Spring 2017, Lecture 10 | Recurrent Neural Networks](https://youtu.be/6niqTuYFZLQ?t=3347)

#### Training
* The LSTM model is **trained to predict each word of the sentence** after it has **seen the image as well as all preceding words** as defined by **p(S<sub>t</sub>|I, S<sub>0</sub> , ..., S<sub>t-1</sub>)**.
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/6.PNG?raw=true" width="50%" height="50%">
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/7.PNG?raw=true" width="50%" height="50%">

* 위 그림과 함께 LSTM에 과정에 대한 기본적인 설명이 나옵니다.
* 각각의 단어를 Dictionary 사이즈와 같은 차원의 one-hot vecotr S<sub>t</sub>로 represent 했습니다.
* S<sub>0</sub>는 start word 이고, S<sub>N</sub>은 stop word입니다.
* We empirically verified that feeding the image at each time step as an extra input yields inferior results, as the network can explicitly exploit noise in the image and overfits more easily.

**Loss:**

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/8.PNG?raw=true" width="50%" height="50%"><b />

Our loss is **the sum of the negative log likelihood** of the correct word at each step **all the parameters of the LSTM, the top layer of the image embedder CNN and word embeddings W<sub>e</sub>**

#### Inference

>주어진 이미지로부터 문장을 생성하는 것에는 많은 방법이 있다고 합니다.<br />
> **Sampling:** we just sample **the first word according to p1**, then provide the corresponding embedding **as input** and sample p2, **continuing like this** until we sample the special end-of-sentence token or some maximum length. <br />
> **BeamSearch:** iteratively consider the set of the k best sentences up to time t as candidates to generate sentences of size t + 1, and keep only the resulting best k of them.
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/Beamsearch.PNG?raw=true" width="50%" height="50%">
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/Beamsearch2.PNG?raw=true" width="50%" height="50%">
\*출처: https://www.oreilly.com/learning/caption-this-with-tensorflow, https://www.youtube.com/watch?v=UXW6Cs82UKo

# 4. Experiments

### 4.1. Evaluation Mertrics

* The most reliable (but time consuming) is to **ask for raters to give a subjective score** on the usefulness of each desciption given the image.
* In this paper, we used this to reinforce that some of the automatic metrics indeed correlate with this subjective score.
* we set up an **Amazon Mechanical Turk experiment**. Each image was rated by **2 workers**.
* **BLEU score:** a form of precision of word n-grams between generated and reference sentences
* **Perplexity:** geometric mean of the inverse probability for each predicted world. But they didn't report it.
* We report two such metrics - METEOR and Cider - hoping for much more discussion and research to arise regarding the choice of metric.
* transforming the description generation task into a ranking task is unsatisfactory.

### 4.2. Datasets

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/9.PNG?raw=true" width="50%" height="50%">

### 4.3 Results

>we wanted to answwer questions such as
>1) how data size affects generalization
>2) what kinds of transfer learning it would be able to achieve
>3) how it would deal with weakly labeled example
> * performed experiments on five different datasets.

#### 4.3.1 Training Details

* overfitting과의 싸움이 가장 힘들었다. 질이 높은 데이터셋이 100,000장보다 적어서 힘들었다. training set sizes가 커지면 좋아질 것이다.
* overfitting을 피하기 위해서 pretrained model(e.g., on ImageNet)의 weights로 intialize를 했다.
* W<sub>e</sub>도 직접 initalize 하려고 했는데, 큰 이점이 없어서 uninitialized 채로 두었다.
* Dropout과 ensembling이 조금 BLEU를 높였다.
* fixed learning rate and no momentum.
* All weights were randomly initialized except for the CNN weights.
* used 512 dimensions for the embeddings and the size of the LSTM memory.
* Descriptions were preprocessed with basic tokenization, keeping all words that appeared at least 5 times in the training set.

#### 4.3.2 Generation Results

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/10.PNG?raw=true" width="50%" height="50%">

논문으로 대체하겠습니다.

#### 4.3.3 Transfer learning, Data Size and Label Quality

* transfer learning and data size
  * Flickr30k로 트레이닝하고 Flickr8k로 테스트했더니 BLEU가 4 points 향상되었다.
  * MSCOCO는 Flickr30k보다 5배 많은데, 구조가 다르다보니 BLEU는 10 points 하락하였다. 그럼에도 불구하고 descriptions은 잘 되었다.
  * PASCAL은 공식적인 training set이 없고 Flickr과 MSCOCO와 독립적이다. 역시 데이터셋이 작은 Flickr30k부터의 transfer learning이 더 결과가 안좋았다.

#### 4.3.4 Generation Diversity Discussion

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/11.PNG?raw=true" width="50%" height="50%">

* an obvious question is whether the model generates **novel captions**, and whether the generated captions are both **diverse and high quality**.
* 위 그림은 Beam search로 찾은 문장들이다. 상위 15개의 생성된 문장들의 스코어가 58로 인간과 비슷하다.
* 최적의 후보를 선택하면 80%가 예제 문장들이다. 데이터의 양이 적기 때문에 놀랄 일이 아니다.
* 하지만 상위 15개의 문장을 살펴보면 반 정도가 새로 생성된 문장이다(여전히 BLEU score가 높다). diversity와 quality가 높음을 보여준다.


#### 4.3.5 Ranking Results

ranking이 unsatisfactory way라고 생각하지만 많은 paper들이 쓴다. 그래서 했는데 잘한다.

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/12.PNG?raw=true" width="50%" height="50%">

#### 4.3.6 Human Evaluation

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/13.PNG?raw=true" width="50%" height="50%">
* This shows that BLEU is not a perfect metric, as it does not capture well the difference between NIC and human descriptions assessed by raters.
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/human.PNG?raw=true" width="50%" height="50%">


#### 4.3.7 Analysis of Embeddings

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/embedding.PNG?raw=true" width="50%" height="50%">

임베딩도 잘 된 것을 알 수 있다.

****************************************************************************************************************************************

## 추가자료/Show and Attend and Tell
