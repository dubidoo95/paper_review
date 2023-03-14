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

이미 작고 가벼운 모델을 더 작고 빠르게 만들기 위해 width multiplier를 도입하였다. width multiplier는 각 layer를 균일하게 얆게 만드는 것으로, input channels $N$과 output channels $M$은 각각 $\alpha N$, $\alpha M$이 되고, depthwise seperable convolution의 computational cost는 $$D_K \cdot D_K \cdot \alpha M \cdot D_F \cdot D_F + \alpha M \cdot \alpha N \cdot D_F \cdot D_F$$가 된다. $\alpha$의 값은 (0,1]사이로 일반적으로 0.25, 0.5, 0.75, 1.0을 사용하며 $\alpha = 1$일 때 baseline MobileNet이고 $\alpha < 1$일 때 reduced MobileNet이다. width multiplier는 computational cost와 parameters의 수를 $\alpha ^2$만큼 줄이는 효과가 있다.

# 3.4. Resolution Multiplier: Reduced Representation

computational cost를 줄이기 위한 방법으로 resolution multiplier $\rho$를 도입하였다. 이는 이를 input image에 적용하면 depthwise separable convolutions의 computational cost는 다음과 같아진다.
$$D_K \cdot D_K \cdot \alpha M \cdot \rho D_F \cdot \rho D_F + \alpha M \cdot \alpha N \cdot \rho D_F \cdot \rho D_F$$
$\rho$의 값은 (0,1]사이로 일반적으로 input resolution이 224, 192, 160, 128이 되도록 조절하고 resolution multiplier는 computational cost를 $\rho ^2$만큼 낮추는 효과가 있다. 일련의 기법들을 적용하면 computational cost와 parameters의 수는 다음과 같이 감소한다.<br>
![image](https://user-images.githubusercontent.com/110075956/224956972-37f504cf-2bed-465c-acfa-27b824e7bf4f.png)

# 4. Experiments

이번 장에선 먼저 depthwise convolutions의 효과를 보았다. 이후 width multiplier와 resolution multiplier의 효과를 보았다. 마지막으로 다양한 tasks에 MobileNet을 적용시키고 결과를 확인하였다.

# 4.1. Model Choices

![image](https://user-images.githubusercontent.com/110075956/225021945-71cb0992-71b4-4a3c-a2cd-57f1fd33091d.png)<br>
full convolutions 대신 depthwise seperable convolutions를 쓴 모델의 정확도는 약 1% 정도 감소하였고 mult-adds와 parameters는 대폭 감소하였다. <br><br>
![image](https://user-images.githubusercontent.com/110075956/225022164-fd9ca8ba-e156-4ed3-a149-79d4a1f093ca.png)<br>
$14 \times 14 \times 512$의 5개 layers를 제거한 모델은 비슷한 computation cost와 parameters를 가지고 있으면서도 3% 더 얇게 만들었다.

# 4.2. Model Shrinking Hyperparameters

![image](https://user-images.githubusercontent.com/110075956/225022646-bc3086ff-14f0-437c-994f-e7a8f951675f.png)<br>
width multiplier가 작아짐에 따라 accuracy가 서서히 줄어들고 $\alpha = 0.25$일 때 확 낮아짐을 볼 수 있다.<br><br>
![image](https://user-images.githubusercontent.com/110075956/225028317-b78b70f6-ec0f-4a42-9e74-52a9cfa8c217.png)<br>
resolution Multiplier가 작아짐에 따라 accuracy가 서서히 줄어드는 것을 볼 수 있다.<br><br>
![image](https://user-images.githubusercontent.com/110075956/225030221-5f5db51b-31b9-4f20-83c0-c48860192a70.png)<br>
Figure 4는 width multiplier $\alpha in {1, 0.75, 0.5, 0.25}$, resolution $in {224, 192, 160, 128}$일 때 accuracy와 computation 간의 trade-off를 보여주는 그래프이다. 전반적으로 log-linear 형태를 띤다.<br><br>
![image](https://user-images.githubusercontent.com/110075956/225031895-b102a2c4-6d28-4a18-bfb9-48df2840ee9f.png)<br>
Figure 5는 위 조건에서 accuracy와 parameters 수의 trade-off를 보여주는 그래프이다. <br><br>
![image](https://user-images.githubusercontent.com/110075956/225037633-e54953f4-1748-4d60-b62f-dbe0b86d0ff6.png)<br>
MobileNet은 VGG16보다 무려 32배 작고 27배 계산을 덜 하였다. GoogleNet과 정확도가 비슷하지만 GoogleNet보다 작고 계산을 덜 하였다.<br><br>
![image](https://user-images.githubusercontent.com/110075956/225038275-4b747697-8ac1-4ebf-8272-6d6831c97602.png)<br>
width multiplier $\alpha = 0.5$로 설정하고 resolution을 $160 \times 160$으로 줄였을 때 Squeezenet과 AlexNet보다 훨씬 작고 가벼웠지만 오히려 정확도가 더 높은 모습을 보였다.

# 4.3. Fine Grained Recognition

![image](https://user-images.githubusercontent.com/110075956/225044210-bbfd0d9f-54cd-4325-9866-edfef8117dca.png)<br>
noisy web data를 이용해 pretrain하고 Stanford Dogs dataset으로 fine grained recognition을 학습시켰다. computation과 size를 상당히 감소시키고도 state of art에 근접하는 성능을 보였다.

# 4.4. Large Scale Geolocalizaion

PlaNet은 해당 사진이 지구 어디서 찍혔는지를 분류하는 model이다. 본 논문에서는 MobileNet architecture를 이용해 PlaNet을 re-train하였고 parameters의 수는 약 $\frac{1}{4}$, mult-adds는 약 $\frac{1}{10}$ 감소시키면서도 정확도 측면에선 아주 약간의 감소만 있었다. 더불어 비슷한 tasks를 수행하는 Im2GPS에 비해서는 월등한 성능을 보였다.

# 4.5. Face Attributes

# 4.6. Object Detection

# 4.7. Face Embeddings

# 5. Conclusion
