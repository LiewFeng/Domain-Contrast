## Domain Contrast

Code of Domain Contrast for Domain Adaptive Object Detection, accepted in IEEE Transactions on Circuits and Systems for Video Technology(TCSVT)，2021.

The code is built based on the [faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0). Please follow original project respository to set up the environment.


### Data Preparation

* **PASCAL_VOC 07+12**: Please refer [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) for constructing PASCAL VOC Datasets.
* **Clipart, Comic, WaterColor**: Please refer [Cross Domain Detection ](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets).
* **SIM10k**: Please refer website [SIM10k](https://fcav.engin.umich.edu/sim-dataset)
* **Cityscape**:Please refer website [Cityscape](https://www.cityscapes-dataset.com/), see dataset preparation code in [DA-Faster RCNN](https://github.com/yuhuayc/da-faster-rcnn/tree/master/prepare_data)
* **Transferred Datasets**: We use [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to generate transferred images.We trained CycleGAN with a learning rate of 2e-4 for the first ten epochs and a linear decaying rate to zero over the next ten epochs.

All codes are written to fit for the Data format of Pascal VOC. After downloading/generating the data, creat softlinks in the folder data/.


### Pretrained Model
In our experiments, we used two pre-trained models on ImageNet, i.e., VGG16 and ResNet101. Please download these two models from:

* **VGG16:** [Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)

* **ResNet101:** [Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0)  [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)

Download them and put them into the data/pretrained_model/.

### Train and Test
All specific hyperparameters are in the shell scripts. Run with the following commands and you will get the results.
**Pascal2Clipart**: 

```
bash pascal2clipart.sh
```

**Pascal2Comic**: 



```
bash pascal2comic_vgg16.sh
```

```
bash pascal2comic_resnet101.sh
```

**Pascal2Watercolor**: 



```
bash pascal2watercolor.sh
```

**SIM10K2Cityscape**: 



```
bash sim10k2city.sh
```

### Results



|         Task       |  Backbone  | mAP  |
|:------------------:|:----------:|:----:|
| Pascal2Clipart     | Resnet101  | 43.2 |
| Pascal2Comic       | VGG16      | 36.9 |
| Pascal2Watercolor  | Resnet101  | 53.7 |
| SIM10K2Cityscape   | VGG16      | 41.6 |
