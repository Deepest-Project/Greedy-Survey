# Resources
* [arXiv](https://arxiv.org/abs/1506.02640)
* [Youtube, TensorFlow KR 논문읽기 모임](https://www.youtube.com/watch?v=eTDcoeqj1_w&t=396s&list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS&index=18)

# Abstract
기존 Object detection 구조의 한계:
> Prior work on object detection repurposes classifiers to perform detection.

> Prior works are not **real-time object detection**.

논문에서 제시하는 해결 방법:
> Instead, we frame object detection as **regression problem** to spatially separated bounding boxes and associated class probabilities.

> A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation..

# Introduction
기존 시스템의 방법론과 단점:
> Curnent detection sytems repurpose classifiers perform dectection.

> To detect an object...,

> 1) propose region of interest by using region proposal network.

> 2) run classifier on these proposed boxes.

> 3) refine the bounding box, eliminate duplicate detections, and rescore the box based on other objects in the scene.

> These complex pipelines are **slow and hard to optimize** because each individual componet must be trained separately.(R-CNN)

> \* Fast R-CNN, Faster R-CNN은 첨부자료 참고.

YOLO의 차별성
> We reframe object detection as a single regression problem, straght from image pixels to bounding box coordinates and class probabilities.

> ... simultaneously predicts multiple bounding boxex and class probabilities .......

> ... trains full images and directly optimizes detection performance. 

이로 인한 장점 세가지.

> First, YPLP is extremly fast.

> * Base network runs at 45 fps and Fast version runs at more than 150 fps.

> Second, YOLO reasons globally about the image when making predictions.

> ... sees the entire image during training and test time so it encodes **contextual information** about classes as well as their appearance.

> Third, YOLO learns **genealizable representations** of objects.
...

# Detail
### Unified Detection

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/grid.PNG?raw=true" width="50%" height="50%">

> ... divides the input image in to S x S grid.
> if center of an object falls into a grid cell, that grid cell is respensible for detecting that object. 
> Each grid cell predicts B bounding boxes and confidence score for those boxes.

##### Confidence: 
<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/confidence.PNG?raw=true">

##### bounding boxes

> Each bounding  box consists of 5 predictions: x, y, w, h, and confidence.
> Each grid cell also predicts C conditional class probabilites.

<img src="https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/class-specific-confidence-scores.PNG?raw=true">



#architecture
![](https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/architecture.PNG?raw=true)



#coord
![](https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/coord.PNG?raw=true)

#errorAnalysis
![](https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/errorAnalysis.PNG?raw=true)

#generalization
![](https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/generalization.PNG?raw=true)

#loss
![](https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/loss.PNG?raw=true)

#noobj
![](https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/noobj.PNG?raw=true)

#obj_i
![](https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/obj_i.PNG?raw=true)

#obj_ij
![](https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/obj_ij.PNG?raw=true)

#qualitativeResults
![](https://github.com/Deepest-Project/Greedy-Survey/blob/ys/Papers/You%20Only%20Look%20Once/qualitativeResults.PNG?raw=true)


# Experiments and Results

# Discussion
