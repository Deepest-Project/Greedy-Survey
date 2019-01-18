# Resources
* [arXiv](https://arxiv.org/abs/1411.4555)
* [Youtube, TensorFlow KR 논문읽기 모임](https://www.youtube.com/watch?v=BrmCnoYhQb4&t=0s&index=42&list=PL0oFI08O71gKjGhaWctTPvvM7_cVzsAtK)


# Abstract

* describing the content of an image is a fundamental problem in artificial intelligence that **connects computer vision and natural language processing**.
* a generaive model based on a **deep recurrent architecture** that ... to generate natural sentences describing an image.

# Introduction

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
  
  
# Related Work
1) Mainly for video. systems composed of visual primitive recognizers combine with structured formal language.
    \- heavily hand-designed, relatively brittle and have been demonstrated only limited domain.
2) Systems dealing with image description were made after some advances in recognition of objects.
    \- These are also limited in their expressivity.
3) The idea of co-embedding of images and text in the same vector space. Descriptions are retrieved which lie close to the image in the embedding space.
    \- do not attempt to generate novel descriptions.
* the above approaches cannot describe previously unseen compositions of objects, even though the individual objects might have been observed in the training data.
4) Simillar recurrent NN for was introduced.  These networks use sentences as RNN input whereas Show and Tell use the visual input to the RNN model directly.
* As a result of these seemingly insignificant differences, our system achieves substantially better results on the established benchmarks.

# Model

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

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/5.PNG?raw=true" width="50%" height="50%">
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/6.PNG?raw=true" width="50%" height="50%">
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/7.PNG?raw=true" width="50%" height="50%">
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/8.PNG?raw=true" width="50%" height="50%">

### LSTM-based Sentence Generator

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/4.PNG?raw=true" width="50%" height="50%">

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/LSTM_cs231n.PNG?raw=true" width="50%" height="50%">

\* LSTM에 대한 구체적인 설명은 cs231n 링크로 대체하겠습니다. <br />[Stanford University CS231n, Spring 2017, Lecture 10 | Recurrent Neural Networks](https://youtu.be/6niqTuYFZLQ?t=3347)

#### Training
* The LSTM model is trained to predict each word of the sentence after it has seen the image as well as all preceding words as defined by p(S<sub>t|I, S<sub>0, ..., S<sub>t-1.


#### conditional class probabilities:


#### class-specific confidence score:


<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/9.PNG?raw=true" width="50%" height="50%">
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/10.PNG?raw=true" width="50%" height="50%">
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/11.PNG?raw=true" width="50%" height="50%">
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/12.PNG?raw=true" width="50%" height="50%">
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/13.PNG?raw=true" width="50%" height="50%">

### Design


### Training

#### leaky rectified linear activation:

#### loss function:

#### to remedy this issue, they set Lamda coord, noobj


### Limiations of YOLO


# Experiments and Results


### VoC 2007 Error Analysis

### Generalization results on Picasso and People-Art

# Discussion
