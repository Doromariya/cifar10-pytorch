# cifar10-pytorch
Partial implementation of [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) by K. He _et al._ 2015

## Dependencies
* Python 2.7
* [PyTorch](http://pytorch.org/)
* [tensorboard](https://www.tensorflow.org/)
* [tensorboardX]()

## Demo
```
python train.py --model resnet20
```

## [The CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)<br>
* 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
* 32x32 tiny images
* 50,000 training images and 10,000 test images

## Models
* ResNet
* VGG
