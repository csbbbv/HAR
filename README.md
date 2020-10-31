
# HAR
## 基于多传感器数据融合的人体行为识别系统 


工程基于以下数据集进行设计：
[数据集-Kaggel](https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones)

用自采集数据集（约3000组特征序列）构建训练集和测试集，模型搭载在树莓派3B上实时运行，acc 96%

## 参考：
  [1D卷积网络](https://blog.csdn.net/bhneo/article/details/83092557);[文献综述](https://github.com/jindongwang/activityrecognition)

## Requirements：

  *python 3.7.3
  
  *torch 1.1.0 
    
  *torchvision 0.3.0 
   
  *Pillow
   
  *Sense-HAT-B（使用九轴传感器）
 
## sensor:
   *accelerometer
     
   *gyroscope
    
## Usage

  * classi.py : load your own dataset and train your model 
  
  * ICM20948.py : run in raspberry pi to get sensor's data
  
  * MakeANewDataset.py ： create Feature Engineering
  
  * run.py : run in raspberry pi,scratch datas,calculate your feature and classification.
  
  * visdo.py ： Monitoring the training process in tensorboard
  
  * visualble.py ： Draw your model as a flowchart
    

## Usage in Raspberry Pi
 
 1.登录
 
>username:pi

>passwd :yahboom

2.run

>  cd ~/pycharmproject/censorcal/model23

>  sudo python 422.py

## 未完成部分

**目前每个动作的特征序列上共有561个特征值，许多特征是无用的，可以用特征工程知识进行特征选择，在数据预处理时做好特征清理工作，提高计算效率**

我认为有两个方向进行改进，一是做成让用户自建小样本数据集进行训练，二是尝试训练出兼容不同设备的模型。面向游戏设备的姿态检测也是较有趣的研究方向，当然它对精度和速度有更高的要求，可以尝试加入更多的传感器并压缩算法。

### 一：

#### 1）在保证正确率情况下，用尽量小的样本进行训练。目前的训练的特征序列仍需要2000+组，对个人采集来说时间还是太长。目前已有许多小样本甚至零样本学习模型，也可以用预训练的模型移植到设备上再让用户Funtuning,或许也可以实现小样本训练

#### 2）做出面向用户的自采集自训练的交互界面


### 二：

#### 1）目前较流行的基于时序的模型或许可以做到，如LSTM网络等等，需做更多研究。

<br>
<br>
<br>
<br>
<br>
<br>
 *如有问题或遗失材料，可联系我：936357225@qq.com*
