# Resources
* [arXiv](https://arxiv.org/abs/1506.02640)
* [Youtube, TensorFlow KR 논문읽기 모임](https://www.youtube.com/watch?v=eTDcoeqj1_w&t=396s&list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS&index=18)

# Abstract
**기존 Object detection 구조의 한계:**
* Prior work on object detection **repurposes classifiers** to perform detection.<br />
* Prior works are **not real-time object detection**.

**논문에서 제시하는 해결 방법:**
* Instead, we frame object detection as **regression problem** to spatially separated bounding boxes and associated class probabilities.<br />
* **A single neural network** predicts **bounding boxes** and **class probabilities** directly from full images in one evaluation..

# Introduction
## 기존 시스템의 방법론과 단점:
> Curnent detection sytems repurpose classifiers perform dectection.
>
> To detect an object...
>
> 1) propose region of interest by using region proposal network.
> 2) run classifier on these proposed boxes.
> 3) refine the bounding box, eliminate duplicate detections, and rescore the box based on other objects in the scene.
>
> These complex pipelines are **slow and hard to optimize** because each individual componet must be trained separately.(R-CNN)
>
> \* Fast R-CNN, Faster R-CNN은 첨부자료 참고.

## YOLO의 차별성
> We reframe object detection as a single regression problem, **straght from image pixels** to **bounding box coordinates and class probabilities.**<br />
> ... simultaneously predicts multiple bounding boxex and class probabilities .......<br />
> ... trains full images and directly optimizes detection performance. 

## 이로 인한 장점 세가지
> **First,** YOLO is extremly fast.<br />
> Base network runs at 45 fps and Fast version runs at more than 150 fps.

> **Second,** YOLO reasons globally about the image when making predictions.<br />
> ... sees the entire image during training and test time so it encodes **contextual information** about classes as well as **their appearance.**

> **Third,** YOLO learns **genealizable representations** of objects.


# Detail
## Unified Detection

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/grid.PNG?raw=true" width="50%" height="50%">

* ... divides the input image in to S x S grid.<br />
* if center of an object falls into a grid cell, that grid cell is respensible for detecting that object. <br />
* Each grid cell predicts B bounding boxes and confidence score for those boxes.

#### Confidence: 
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/confidence.PNG?raw=true" width="20%" height="20%">

#### bounding boxes
* Each bounding box consists of **5 predictions: x, y, w, h, and confidence.**

#### conditional class probabilities
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/class-specific-confidence-scores.PNG?raw=true" width="50%" height="50%"><br />
* Each grid cell also predicts **C** conditional class probabilites.

* the predictions are encoded as an **S x S x (B * 5 + C) tensor.**


### Design
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/architecture.PNG?raw=true" width="50%" height="50%">

* **GoogleLeNet** + 2 fully connected layers
* replace inception modules with **1 x 1 reduction layers** followed by 3 x 3 convolutional layers. 

### Training

* **pretrain first 20 convolutional layers** on the ImageNet 1000-class competetion dataset.
* achieve a single crop top-5 accuracy of 88% on the ImageNet 2012 validation set.
* add **four convolutional layers** and **two fully connected layers** with randomly initialized weights.
* increase the input resolution of the network from 224 x 224 to 448 x 448.
* normalize the bounding box width and height by the image width and height so that they  fall between 0 and 1.
* parametrize the bounding box x and y coordinates to be offsets of a particular grid cell location so they are also bounded between 0 and 1.

#### leaky rectified linear activation:

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/activation.PNG?raw=true" width="50%" height="50%">

* optimize for **sum-squared error**. It weights **localization error equally with classification error** which may not be ideal.
* many gird cells do not contain any object. -> pushing the "confidence" scores of these cells <br />
-> **overpowering the gradient** form cells that do contain object.

#### to remedy this issue, they set Lamda coord, noobj

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/coord.PNG?raw=true" width="5%" height="5%"> = 5.0
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/noobj.PNG?raw=true" width="5%" height="5%"> = 0.5

> Sum-squared error also **equally weights errors in large boxes and small boxes**.
>
> To partially address this we predict **the square root of the bounding box width and height** instead of the width and height directly
>

> At the trainin time we only want one bounding box predictor to be responsible for predicting an object based on which prediction has **the highest current IOU** with the ground truth.
>
> Assign one predictor to be "respensible" for prediction an object based on which prediction has the highest current IOU with the ground truth. 

#### loss fuction
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/loss.PNG?raw=true" width="50%" height="50%">

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/obj_i.PNG?raw=true" width="3%" height="3%"> if object appears in cell i

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/obj_ij.PNG?raw=true" width="3%" height="3%"> jth bounding box predictor in cell i is "responsible for that prediction.

# Experiments and Results

자세한 설명은 논문으로 대체하겠습니다.

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/errorAnalysis.PNG?raw=true" width="50%" height="50%">

* Correct: correct class and IOU> .5
* Localization: correct class, .1 < IOU < .5
* Similar: class is similar, IOU > .1
* Other: class is wrong, IOU > .1
* Background: IOU < .1 for any object

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/generalization.PNG?raw=true" width="50%" height="50%">

### 
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/qualitativeResults.PNG?raw=true" width="50%" height="50%">

# Discussion
YOLO의 다음 버젼들을 모르는 상태에서 성능개선 방안을 생각해본다면?
