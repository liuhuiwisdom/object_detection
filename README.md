This is a Tensorflow implementation of the object detector described by the paper "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" by Ren et al. (NIPS2015) and "Deep Residual Learning for Image Recognition" by He et al. (CVPR2016). Given an image, it predicts the bounding box and label of each object in the image. It uses a Region Proposal Network (RPN) to find a set of rectangular cadidate regions, and uses a Fast R-CNN to classify these regions. To improve the efficiency, the RPN and Fast R-CNN modules share their convolutional layers. These convolutional layers are inherited from VGG16, ResNet50, ResNet101 or ResNet152 model, and these models can be obtained by using Caffe-to-Tensorflow. 

References
----------
* [Fast R-CNN.](https://arxiv.org/abs/1504.08083) Ross Girshick. ICCV 2015.
* [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.](https://arxiv.org/abs/1506.01497) Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. NIPS 2015.
* [Deep Residual Learning for Image Recognition.](https://arxiv.org/abs/1512.03385) 
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. CVPR 2016.
* [Instance-aware Semantic Segmentation via Multi-task Network Cascades.](https://arxiv.org/abs/1512.04412) Jifeng Dai, Kaiming He, Jian Sun. CVPR 2016.
* [Faster R-CNN in Caffe (Matlab version)](https://github.com/ShaoqingRen/faster_rcnn)
* [Faster R-CNN in Caffe (Python version)](https://github.com/rbgirshick/py-faster-rcnn)
* [Microsoft COCO dataset](http://mscoco.org/)
* [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
* [Caffe to Tensorflow](https://github.com/ethereon/caffe-tensorflow)

