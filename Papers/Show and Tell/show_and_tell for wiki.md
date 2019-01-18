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

> Machine translation models make use of a recurrent neural network which encodes the variable length input into a fixed dimensional vector, and uses this representation to “decode” it to the desired output sentence. <br />
> Thus, it is natural to use the same approach where, given an image (instead of an input sentence in the source language), one applies the same principle of “translating” it into its description.

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/1.PNG?raw=true" width="50%" height="50%">

$$\theta$$ : parameters of our model
*I* : image
*S* : correct transcription (unbounded length)


<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/2.PNG?raw=true" width="50%" height="50%">
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/3.PNG?raw=true" width="50%" height="50%">
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/4.PNG?raw=true" width="50%" height="50%">
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/5.PNG?raw=true" width="50%" height="50%">
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/6.PNG?raw=true" width="50%" height="50%">
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/7.PNG?raw=true" width="50%" height="50%">
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/8.PNG?raw=true" width="50%" height="50%">

 #### confidence: 
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/confidence.PNG?raw=true" width="20%" height="20%">

#### bounding boxes:
* Each bounding box consists of **5 predictions: x, y, w, h, and confidence.**

#### conditional class probabilities:
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/conditional-class-probability.PNG?raw=true" width="20%" height="20%"><br />
* Each grid cell also predicts **C conditional class probabilites**.

#### class-specific confidence score:
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/class-specific-confidence-scores.PNG?raw=true" width="50%" height="50%"><br />
* At test time we **multiply** the conditional class probabilities and the individual box confidence predictions.

the predictions are encoded as an **S x S x (B * 5 + C) tensor.**

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/9.PNG?raw=true" width="50%" height="50%">
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/10.PNG?raw=true" width="50%" height="50%">
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/11.PNG?raw=true" width="50%" height="50%">
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/12.PNG?raw=true" width="50%" height="50%">
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/Show%20and%20Tell/13.PNG?raw=true" width="50%" height="50%">

### Design
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/architecture.PNG?raw=true" width="80%" height="80%">

* **GoogleLeNet** + 2 fully connected layers
* replace inception modules with **1 x 1 reduction layers** followed by 3 x 3 convolutional layers. 

### Training

* **pretrain first 20 convolutional layers** on the ImageNet 1000-class competetion dataset.
* achieve a single crop top-5 accuracy of 88% on the ImageNet 2012 validation set.
* add **four convolutional layers** and **two fully connected layers** with randomly initialized weights.
* increase the input **resolution** of the network from 224 x 224 to 448 x 448.
* **normalize** the bounding **box width and height** by the image width and height so that they fall between 0 and 1.
* parametrize the bounding box x and y coordinates to be offsets of a particular grid cell location so they are also bounded between 0 and 1.

#### leaky rectified linear activation:

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/activation.PNG?raw=true" width="40%" height="40%">

#### loss function:
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/loss.PNG?raw=true" width="100%" height="100%">

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/obj_i.PNG?raw=true" width="5%" height="5%"> : if object appears in cell i

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/obj_ij.PNG?raw=true" width="5%" height="5%"> : jth bounding box predictor in cell i is "responsible" for that prediction.

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/noobj_ij.PNG?raw=true" width="5%" height="5%"> : jth bounding box predictor in cell i with no object.
* optimize for **sum-squared error**. It weights **localization error equally with classification error** which may not be ideal.
* many gird cells do not contain any object. -> pushing the "confidence" scores of these cells <br />
-> **overpowering the gradient** form cells that do contain object.

#### to remedy this issue, they set Lamda coord, noobj



> Sum-squared error also **equally weights errors in large boxes and small boxes**.
>
> To partially address this we predict **the square root of the bounding box width and height** instead of the width and height directly
>

> At the training time we only want one bounding box predictor to be responsible for predicting an object based on which prediction has **the highest current IOU** with the ground truth.
>
> Assign one predictor to be "responsible" for prediction an object based on which prediction has the highest current IOU with the ground truth. <br />
>\* non-maximal suppression adds 2-3% in mAP.[detail](https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.g137784ab86_4_1318)

### Limiations of YOLO


# Experiments and Results


### VoC 2007 Error Analysis

### Generalization results on Picasso and People-Art

# Discussion
