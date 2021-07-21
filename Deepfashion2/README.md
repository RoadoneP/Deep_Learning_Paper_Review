Deepfashion2
============

![img](https://ifh.cc/g/Z2EPsx.jpg)

dataset
-------

> It contains 13 popular clothing categories from both commercial shopping stores and consumers.

'short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear', 'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress'

1.	train.zip (10 GB)

2.	validation.zip (2 GB)

train
-----

I used [Mask-RCNN]('https://arxiv.org/abs/1703.06870').

> R-CNN은 CNN에 Region Proposal 을 추가하여 물체가 있을법한 곳을 제안하고, 그 구역에서 object detection을 하는 것이다. R-CNN 계열 모델은 R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN 까지 총 4가지 종류가 있다.

I'll use use YOLO and Deeplabs model later.

-	10epochs ![img](https://ifh.cc/g/ERv8iO.jpg)
-	100epochs ![img](https://ifh.cc/g/afN103.jpg)

Description
-----------

```
$(Deepfashion2)
|-- dataset
|   |-- train
|   |   |-- images
|   |   |-- train.json
|   |-- validation
|   |   |-- images
|   |   |-- valid.json
|
|-- source
|-- tools
|-- lib
|-- main.py
|-- model_test.py
|-- requirement.txt
```
