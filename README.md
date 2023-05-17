# Adaptive Distillation for Fast and Accurate N:M Sparse Fine-tuning


## Requirements

- python 3.7+
- pytorch 1.13.0
- torchvision 0.14.0

## Data Preparation

- The ImageNet dataset should be prepared as follows:

```text
ImageNet
├── train
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 1)
│   ├── ...
├── val
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 1)
│   ├── ...
```

## Model Preparation
Since AdaDistill includes feature maps distillation, if you want to use AdaDistill on other torchvision models, you need to modify the forward code in the torchvision model library. Otherwise, you will meet the following bug：
```bash
output, _ = model(image)
ValueError: too many values to unpack (expected 2)
```
We provide two examples of modifications:
```python
# You can find your torchvision model code in this Path
# {Your_env_name}/lib/python3.7/site-packages/torchvision/models/resnet.py
def _forward_impl(self, x: Tensor) -> Tensor:
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    f = x  # modified 
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x,f # modified

# {Your_env_name}/lib/python3.7/site-packages/torchvision/models/mobilenetv2.py
def _forward_impl(self, x: Tensor) -> Tensor:
    x = self.features(x)
    f = x # modified
    x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x, f # modified


```

## Re-produce our results



- ResNet-32 on CIFAR-10

```bash
cd ResNet
# Set the path and hyperparameter in cifar.sh
sh cifar.sh
```

- ResNet-50 on ImageNet

```bash
cd ResNet
# Set the path and hyperparameter in imagenet.sh
sh imagenet.sh
```

- ResNext50_32x4d on ImageNet

```bash
cd torch_models
# fill in the path and hyperparameter 
torchrun --nproc_per_node=4  main.py --data-path ImageNet_PATH --output-dir /output/ --model resnext50_32x4d  --lr 0.1 --wd 0.0001 --lr-scheduler cosineannealinglr --opt  sgd  --label-smoothing 0.1  --epochs 120 -b 64 --usenm --N 2 --M 4 --amp --resume_pretrain Pretrained_Model_PATH --resume_teacher Teacher_Model_PATH --Tau 1 --a 0 --b 0.5 --c 0.5 
```

- MobileNet-V2 on ImageNet

```bash
cd torch_models
# fill in the path and hyperparameter 
torchrun --nproc_per_node=4 main.py --data-path ImageNet_PATH --output-dir /output/ --model mobilenet_v2  --lr 0.045 --lr-scheduler steplr --lr-step-size 1 --lr-gamma 0.98 --wd 0.00004 --epochs 300 -b 64 --N 2 --M 4 --usenm --amp --Tau 1 --a 0 --b 0.5 --c 0.5 --resume_pretrain Pretrained_Model_PATH --resume_teacher Teacher_Model_PATH 
```

- EfficientNet-B0 on ImageNet

```bash
cd torch_models
# fill in the path and hyperparameter 
torchrun --nproc_per_node=8  main.py --data-path ImageNet_PATH --output-dir /output/ --model efficientnet_b0 --label-smoothing 0.1 --opt RMSprop --lr 0.08 --lr-scheduler cosineannealinglr --wd 0.00001 --epochs 400 -b 256 --amp --lr-warmup-method linear --lr-warmup-epochs 16 --mixup-alpha 0.2 --usenm --N 2 --M 4 --resume_pretrain Pretrained_Model_PATH --resume_teacher Teacher_Model_PATH --a 0 --b 0.5 --c 0.5 --Tau 1 
```


Besides, we provide our trained models and experiment logs at [Google Drive](https://drive.google.com/drive/folders/10sJ9hCn6ezJeYk91qPeEOT6fQ44ntceT?usp=share_link). To test, run:

- ResNet on CIFAR-10

```bash
cd ResNet
python main_cifar.py --data_path CIFAR-10_PATH  --job_dir /output/ --arch resnet32_cifar10 --data_set cifar10 --pretrained_model Pretrained_Model_PATH  --test-only
```

- ResNet on ImageNet

```bash
cd ResNet
python main_imagenet.py --data_path ImageNet_PATH --job_dir /output/ --arch resnet50 --data_set imagenet  --pretrained_model Pretrained_Model_PATH --test-only
```


- Other models on ImageNet

```bash
cd torch_models
# fill in the path and hyperparameter 
torchrun --nproc_per_node=4 main.py --data-path ImageNet_PATH --output-dir /output/ --model {model_name}  --resume_pretrain Pretrained_Model_PATH --test-only
```
Here {model_name} is one of mobilenet_v2, resnext50_32x4d or efficientnet_b0.