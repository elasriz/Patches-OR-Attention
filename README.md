#  Attention Or Patches ? What we really need ? 🤷🏻
![images](images/patches_are_all_you_need.png?raw=true)

### Authors
Zakariae EL ASRI - Nicolas GREVET 


## Objectif

In this project, we provide the details of reimplementing a ConvMixer (a new architecture on Convolutional Neural Networks) on a new dataset. Then, we verify if the ConvMixers performs better than a Transformer-base model on this new dataset

A tutorial is availible in this google colab: https://colab.research.google.com/drive/1kkSIgFQIAP-0Yt2spVbQBwZv2MsBtWuM?usp=sharing


## Abstract

For many years, the mainstream architecture in computer vision was CNNs, until the time when Vision Transformer (ViT), a transformer-based model shown promising performance. On later works, it was improved to outperform CNNs in many vision tasks. Where image resolutions are very large, the quadratic computation complexity of selfattention was a major bottleneck for vision. To tackle this problem, ViTs introduced the use of patch embeddings, which group together small regions of the image into single input features. This raises the idea that gains of vision transformers are due, in part, to patch representation as input. The question is to determine which factor is more important, the patch representation or the self-attention?

In this sense, Trockman et al. [1] presented a new idea in computer vision. The authors present a new architecture named ConvMixer that destroy the pyramid architecture on CNNs and replace it by an isotropic one using patches.

The paper show that the new architecture outperforms ViT, for similar parameter counts and dataset sizes.


## The model: ConvMixer

This model destroys the historical triangular architecture of ConvNets that increases feature sizes and decreases resolution. Instead, It use an isotropic architecture similar to transformers, where the main computations are performed with convolutions instead of self-attention. The architecture is very simple. It has a patch embedding stage followed by repeated convolutional blocks.

<img src="images\convmixer.png"  />


## Experiment
### Dataset:
In this project, we use the Imagenette [2] dataset. It's very similar to Imagenet, but much less expensive to deal with.
The Imagenette dataset consists of a subset of 10 classes from Imagenet (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute). 
It contains two versions: '320 px' and '160 px' and have a 70/30 train/valid split.

### Training

We trained the model on Imagenette-160 classification with the folowing specifications:
* Data augmentation: Random Horizontal Flip and Random Resized Crop
* kernel size and a patch size : 7
* ConvMixer layer depth : 12
* hidden dimensions : 256
* Activation function: GeLu
* One Cycle Learning Rate Policy
* Optimizer: AdamW

### Baselines:
* ViT, with 6 transform layers with 8 heads in the Multi-Head Attention block.
* Resnet-18:

## Results

We can observe the trend of the convergence of the losses vs. the number of epochs (max 30), it shows that our model isn't overfitting to the training data

<img src="images\convmixer_training.png"  />

Our main objective is not to perform the best accuracy but to compare thethree models to see that patches are a major factor in such architectures.
we see that the ConvMixer model is clearly performing better
with the same range in the number of parameters (between 0.8M and 1.6M).

<img src="images\comparaison.png" width="640" height="512" />


## Files


* ``Project_Patches_or_Attention.ipynb``   :              A tutorial from Google Collab
* ``Patches or Attention _ Project Deep Learning.pdf``  : The final Report for the project.







## Further Reading

This model is closely based on (TD3) paper and an implementation for Minitaur:

[1] Trockman, Asher and J. Zico Kolter. “Patches Are All You Need?” ArXiv abs/2201.09792 (2022)
[2] Jeremy Howard. imagenette, 2019a. URL https://github.com/fastai/imagenette/.



