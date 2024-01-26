# UrbanCarDetector
Comparison of Deep Learning Models for Urban Car Detection
__________________________________________________________

The goal of the project was to compare the performance of six object detection models for detecting cars within a complex urban environment. 1,593 image tiles were created from RGB drone imagery at a spatial resolution of 3.65 cm. Some vehicles are clustered together while others are not, or partially covered by trees and shadows. The models tested for comparison all use a standard set of hyperparameters and backbone models. Except for YOLOv3 which uses darknet53 and MMDetection which uses cascade_rcnn. The ArcGIS.Learn module from the ArcGIS Developer library was using for coding with the pytorch and fastai backends. Training object detection models is available within ArcGIS Pro but it was necessary to use the python libraries to ensure that the same training/testing split was used for evaluating each model. In addition to the model comparison the resnet50 backbone model was selected based on a grid search of the available resnet family of models (18-152) using FasterRCNN.

![image](https://github.com/DanGeospatial/UrbanCarDetector/assets/87085567/b3f3a892-e09e-45d9-8d90-ed800b5d9d39)

__________________________________________________________

Tile sizes of [256,416,512], strides of [half,quarter,0] and a minimum polygon overlap ratio of 0.5 were compared when optimizing the tiles. Using a tile size of 416 provided approximately 98% the average precision of 512 but at only about 60% of the training time. For this reason a tile size of 416 and stride of 224 were chosen after comparing options using FasterRCNN. Larger tile sizes can provide more context of surrounding objects but requires more GPU memory. The minimum polygon overlap ratio was used to try to eliminate tiles that only contained very small or no portions of vehicles. 

Overall, the DETReg model had the highest average precision score (0.955) out of any model compared with YOLOv3 (0.915), MMDetection (0.932) and FasterRCNN (0.9348) producing very similar results. RetinaNet and SingleShotDetector produced substantially worse results than the other models. 

![image](https://github.com/DanGeospatial/UrbanCarDetector/assets/87085567/771e0421-644a-4da2-974d-260d09c97c16)

__________________________________________________________

Next, when we look at the total run time to train each model we get a very different picture. While DETReg had the highest average precision score it also took the longest to train. SingleShotDetector did very poorly overall after taking more than 80 minutes to run while providing a low average precision score. YOLOv3 and FasterRCNN both took less than 50 minutes to train on this dataset while producing among the highest precision. 

![image](https://github.com/DanGeospatial/UrbanCarDetector/assets/87085567/d58bb4bc-b9e6-4821-b42d-ddff7487f1ec)

__________________________________________________________

DETReg produced the highest average precision score and required the largest number of epochs whereas FasterRCNN provided a balance between accuracy and processing time. After resnet50 adding more layers did not improve model.
