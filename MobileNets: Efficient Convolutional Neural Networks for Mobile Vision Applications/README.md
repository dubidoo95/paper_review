# 1. Introduction

Convolutional neural networks는 점점 복잡해지고 정교해지고 있다. 하지만 이는 size와 speed 면에서 효율성을 떨어뜨리는 측면도 있다. robotics, self-driving cars, argumented reality같은 분야에서는 platform의 제약때문에 그런 networks를 이용해서 tasks를 수행할 수 없다. 이에 본 논문에서는 효율적인 network architecture와 적은 수의 hyper-parameters를 이용해 매우 작고 latency가 작은 모델을 고안하였다.<br>
![image](https://user-images.githubusercontent.com/110075956/224365534-8b623839-5752-423d-b33c-30384321f3ab.png)


# 2. Prior Work

최근 작고 효율적인 neural networks를 만들고자 하는 노력이 이어져왔다. 다른 논문들은 대개 small networks에만 포커스를 두고 speed에는 관심을 갖지 않지만 본 논문에서는 latency를 최적화하는 것에 우선적으로 포커스를 두고 small networks를 고려하였다.<br>
MobileNets은 Inception models에서 처음 도입되고 사용된 depthwise separable convolutions로 구축하였다. Flattened networks는 factorized convolutions로 network를 구성하였는데 이는 factorized networks의 잠재력을 보여주었고, Xception network는 어떻게 depthwise separable filters를 확장하는지 보여주었다. Squeezenet은 bottleneck을 사용해 매우 작은 network를 구축하였다.

# 3. MobileNet Architecture

# 3.1. Depthwise Separable Convolution

Depthwise separable convolution은 standard convolution을 depthwise convolution과 $1\times1$ pointwise convolution으로 분해한 것을 말한다. MobileNet model은 이 depthwise separable convolution에 기반을 두고 있다. 우선 depthwise convolution은 각 input channels에 대해 하나의 filter를 적용한다. 이후 pointwise convolution이 depthwise convolution의 outputs을 합하는 구조이다. standard convolution을 두 개의 layers로 나눈 형태로 연산량과 model size의 극적인 감소를 보인다고 한다. <br>
![image](https://user-images.githubusercontent.com/110075956/224698771-b1bd3a12-7039-44ef-be44-c08c96069544.png)<br>
standard convolution의 computational cost는 $D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F$이고 depthwise convolution의 computational cost는 $D_K \cdot D_K \cdot M \cdot D_F \cdot D_F$, pointwise convolution의 경우엔 $M \cdot N \cdot D_F \cdot D_F$이다. 따라서 standard convolution과 depthwise separable convolution의 computational cost를 비교해보면, $\frac{D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F}{D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F}$ = $\frac{1}{N} + \frac{1}{D^2_K}$가 된다. 따라서 depthwise separable convolution의 연산량은 standard convolution의 약 1/8~1/9라고 한다.

# 3.2. Network Structure and Training

![image](https://user-images.githubusercontent.com/110075956/224714484-82f42ebe-8fe8-4084-b768-9d37ff908a2c.png)<br>
첫 layer는 full convolution으로 구성하고 나머지는 depthwise separable convolution으로 구성한다. 마지막 fully connected layer를 제외한 모든 layers에는 batchnorm과 ReLU를 덧붙인다. 마지막 layer에는 classification을 위한 softmax를 적용한다. <br><br>
![image](https://user-images.githubusercontent.com/110075956/224714816-c5fd16e2-2f1b-4229-be98-a4254c548f23.png)<br>
batchnorm과 ReLU를 덧붙인 모습은 위 그림과 같다. depthwise와 pointwise convolution을 하나의 seperate layer로 셌을 때 MobileNet은 총 28layers로 이루어진 network가 된다.<br><br>
MobileNet는 network가 간단하다는 장점 외에도 general matrix multiply(GEMM) functions에 최적화 되어있다는 장점이 있다. 일반적인 convolution은 GEMM으로 연산하기 위해는 memory의 reordering 과정이 필요한데 $1\times1$ convolution의 경우엔 이 reordering 과정이 필요하지 않다. 따라서 MobileNet은 연산에 들어가는 시간을 획기적으로 줄일 수 있었다.<br>
![image](https://user-images.githubusercontent.com/110075956/224721553-d51eeb54-ed6a-4983-8e45-0c9d63d6b454.png)<br>
optimizer로는 RMSprop을 사용하였고 모델이 작아 overfitting의 위험이 적기 때문에 regularization과 data augmentation은 많이 사용하지 않았다. parameters 수 또한 매우 적기 때문에 weight decay도 사용하지 않거나 매우 작은 값을 사용하였다.

# 3.3. Width Multiplier: Thinner Models

이미 작고 가벼운 모델을 더 작고 빠르게 만들기 위해 width multiplier를 도입하였다. width multiplier는 각 layer를 균일하게 얆게 만드는 것으로, input channels $N$과 output channels $M$은 각각 $\alpha N$, $\alpha M$이 되고, depthwise seperable convolution의 computational cost는 $$D_K \cdot D_K \cdot \alpha M \cdot D_F \cdot D_F + \alpha M \cdot \alpha N \cdot D_F \cdot D_F$$가 된다. $\alpha$는 일반적으로 0.25, 0.5, 0.75, 1.0을 사용하며 $\alpha = 1$일 때 baseline MobileNet이고 $\alpha < 1$일 때 reduced MobileNet이다. width multiplier는 computational cost와 parameters의 수를 $\alpha ^2$만큼 줄이는 효과가 있다.

# 3.4. Resolution Multiplier: Reduced Representation

# 4. Experiments

# 4.1. Model Choices

# 4.2. Model Shrinking Hyperparameters

# 4.3. Fine Grained Recognition

# 4.4. Large Scale Geolocalizaion

# 4.5. Face Attributes

# 4.6. Object Detection

# 4.7. Face Embeddings

# 5. Conclusion
