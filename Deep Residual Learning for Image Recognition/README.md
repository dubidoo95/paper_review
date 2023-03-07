# 1. Introduction
![image](https://user-images.githubusercontent.com/110075956/222880416-b0e49ab5-ce42-4dbb-8b69-ffb291935ff1.png)<br>
convolutional neural networks의 깊이는 매우 중요하고 깊은 networks는 좋은 성능을 이끌어낸다. 하지만 networks가 깊어질수록 한가지 문제를 수반하는데, 바로 vanishing/exploding gradients라는 것이다. 이 문제를 normalized initialization, intermediate normalization layers, stochastic gradient descent(SGD) 등으로 해결해왔지만 이내 degradation이라는 새로운 문제에 직면하였다. network의 깊이가 증가할수록 training error가 오히려 높아지는 모습을 보였다.<br><br>

![image](https://user-images.githubusercontent.com/110075956/222884756-15a484ff-3ef9-4575-8c91-60a5ed7ec8b1.png)<br>
본 논문에서는 이 degradation 문제를 해결하기 위해 deep residual learning framework를 도입하였다. 각 layers가 직접 쌓이는 대신, residual mapping을 하도록 설계하였다. 기본 mapping을 $H(x)$라 한다면, 또다른 nonlinear mapping $F(x) = H(x) - x$을 쌓았다. 이 때 original mapping을 다시 쓴다면 $H(x) = F(x) + x$로 쓸 수 있다. 그리고 $F(x) + x$는 위 그림에서 나오는 것처럼 "shortcut connections"를 이용하여 만들 수 있다.<br><br>
이 방법을 이용해 ImageNet, CIFAR-10의 dataset으로 network를 학습시켜본 결과 기존의 "plain" networks보다 optimize가 잘되고 train error 또한 낮게 나옴을 확인하였다. 그리고 ILSVRC 2015 classification competition에서 1등을 거머쥐었다. 이 network는 일반화 성능 또한 좋아서 ImageNet detection, ImageNet localization, COCO detection, COCO segmentation 등 다양한 competitions에서 우수한 성적을 거두었다.

# 2. Related Work

VLAD와 VLAD를 확률적 표현으로 만든 Fisher Vector 모두 image recognition에서 좋은 성능을 보이는 residual representations다. vector quantization의 경우엔 residual vectors를 encoding하는 것이 original vectors를 encoding하는 것보다 효과적인 모습을 보인다. <br>
low-level vision과 computer graphics에서 Partial Differential Equations(PDE)를 풀 때 Multigrid methods를 많이 사용한다. Multigirid methods란 system을 여러 개의 하위 문제로 재구성하여 문제를 해결하는 방식이다. Multigrid의 대안은 두 scales간의 residual vectors를 나타내는 변수에 의존하는 계층 기반 pre-conditioning으로, 이는 standard solvers보다 훨씬 빨리 수렴한다는 장점이 있다. 이 방식은 좋은 reformulation 혹은 preconditioning이 optimization을 단순화시킬 수 있음을 입증한다.<br><br>
Shortcut connections의 초기 연구는 network의 input에서 output으로 연결된 linear layer를 하나 추가하는 것이었다. 이후 몇개의 intermediate layers가 vanishing/exploding gradients를 해결하기 위해 auxiliary classifiers에 직접 연결되었고 다른 논문들에서는 layer responses, gradients, propagated errors를 centoring하기 위해 shortcut connections를 활용하였다. <br>
"highway networks"는 gating functions가 있는 shortcut connections를 제공한다. 이 gates는 data 의존적이고 parameters를 가지고 있다. gated shortcut이 닫혔을 때 highway networks는 non-residual functions를 가진다. 반면 우리가 사용한 shortcut connections는 parameter-free이고 shortcut이 닫히지 않아 항상 residual functions를 가진다. 모든 정보는 항상 통과하고 residual functions는 학습된다. 더불어 highway networks는 100개 이상의 layers를 가진, extremely increased depth에서도 accuracy를 올려준다는 것이 입증되지 않았다.

# 3. Deep Residual Learning

# 3.1. Residual Learning

$H(x)$는 이들 layers 중 첫 번 layer에 대한 입력을 나타내는 x와 함께 몇 개의 stacked layers에 적합한 underlying mapping이다. 만약 multiple nonlinear layers가 complicated functions를 점근적으로 근사할 수 있다면 그것은 residual function, $H(x) - x$를 점근적으로 근사할 수 있는 것과 같다. 따라서 stacked layers를 $H(x)$로 근사하기보다 $F(x): H(x) - x$를 근사하고, original function은 $F(x) + x$가 된다. deeper model이 shallower counterpart보다 training이 높았던 문제가 multiple nonlinear layers를 근사하기 어렵기 때문이었는데 이 residual learning reformation을 이용해 단순히 multiple nonlinear layers의 weights를 0으로 유도하여 identity mappings에 접근할 수 있다.

# 3.2. Identity Mapping by Shortcuts

본 논문에서는 building block을 다음과 같이 정의한다.
$$y = F(x, {W_i}) + x$$
여기서 $x$와 $y$는 각각 input, output vectors를 나타내며 $F(x, {W_i})$는 residual mapping을 뜻한다. activation function으로는 ReLU를 사용하고 단순화하기 위해 bias를 생략한다. shortcut connection과 element-wise addition이 $F + x$ 연산을 수행하고 이후 second nonlinearity를 더한다.<br>
위 공식은 추가 parameters도 없고 계산이 복잡하지도 않다. 이는 실용적이며 plain networks와 residual networks의 비교를 가능하게 만들었다. 본 논문에서는 parameters, depth, width, computational cost를 동일하게 통제하고 plain/residual networks를 비교하였다. <br>
$x$와 $F$의 차원은 같아야하는데 만약 다를 때엔 linear projection $W_s$을 이용함으로써 두 차원을 동일하게 맞출 수 있다.
$$y = F(x, {W_i}) + W_sx$$

# 3.3. Network Architectures

본 논문에서는 plain networks와 residual networks를 각각 여러개 만들어서 그 성능을 비교하였다. <br>
![image](https://user-images.githubusercontent.com/110075956/223146322-56cb5e13-5317-4244-a65d-a6f326f73b97.png)<br>
plain networks는 Fig.3의 중간과 같은 모습으로 VGG networks와 유사한 모습으로 구현하였다. convolutional layers는 대부분 $3\times3$ filters를 사용하였다. 같은 output feature map size를 갖는 layers는 같은 수의 filters를 가지며, layer당 time complexity를 유지하기 위해 map size가 절반으로 줄어들 때 filters의 수는 두 배로 증가시켰다. <br>
residual networks는 위의 plain networks를 기반으로 shortcut connections을 추가한 것이다. input과 output의 차원이 같으면 identity shortcut을 사용하고 차원이 달라질 때는 두 가지 옵션을 고려했는데, shortcut이 identity mapping을 수행할 때는 zero로 padding했고 그 외엔 Eqn.2의 방법을 사용하였다.

# 3.4. Implementation

ImageNet에 대한 구현은 다른 논문에서 사용된 방식을 따랐다. scale augmentation, crop, horizontal flip, standard color augmentation 등이 사용되었다. 각 convolution과 activation 사이엔 batch normalization을 넣었다. optimizer로는 256 mini-batch size의 SGD를 사용하였고 learning rate는 0.1부터 시작하여 error가 발생할 때마다 10으로 나누었다. models는 $60 \times 104$ iterations만큼 학습되었고 weight decay는 0.0001, momentum은 0.9로 설정하였다. dropout은 사용하지 않았다. 

# 4. Experiments

# 4.1. ImageNet Classification

1000개의 classes로 이루어진 ImageNet 2012 classification dataset을 가지고 모델 평가를 진행하였다. training images는 1.28M, validation images는 50k, test images는 100k개가 있으며 top-1과 top-5 error rates를 기록하였다. <br><br>
![image](https://user-images.githubusercontent.com/110075956/223344153-968a4b80-4263-4bd6-b27b-9836c795aa14.png)<br>
plain nets의 경우 더 깊은 34-layer nets가 18-layer nets보다 더 높은 validation error를 기록하였다. 이는 34-layer nets가 18-layer보다 training error가 높기 때문이다. 저자는 이 현상을 vanishing gradients 때문이 아닌, 단지 매우 낮은 convergence rates를 가지는 것이라 하였다.<br><br>
다음으로 18-layer와 34-layer residual nets를 평가하였다. 이 두 nets는 위에서 사용한 plain nets에 shortcut connection을 더한 것으로, 34-layer ResNet이 18-layer ResNet보다 좋은 성능을 보였고, 34-layer ResNet은 34-layer plain net에 비해 3.5%p의 train error를 줄였으며 18-layer ResNet은 18-layer plain net과 accuracy는 비슷했지만 훨씬 빠르게 수렴하는 모습을 보였다. <br><br>
다음으로 projection shortcuts를 확인해보았다. <br>
![image](https://user-images.githubusercontent.com/110075956/223380368-7be6f13c-a0ad-41a9-91b5-d31ca0a4a57b.png)<br>
위 table3에서 A는 increasing dimension에 zero-padding shortcuts을, B는 increasing dimension에 projection shortcuts과 나머지엔 identity shortcuts을, C는 모두 projection shortcuts을 사용한 것이다. train error를 보았을때 C>B>A로 나왔는데 A에 사용된 zero-padding은 residual learning을 하지 않기 때문이고, C가 B보다 더 좋은 성능을 보인 것은 더 많은 projection shortcuts으로 인해 더 많은 parameters가 사용되었기 때문이다. 다만 A, B, C간에 큰 차이를 보이지 않는 것으로 보아 projection shortcuts은 degradation problem을 해결하는 데 필수적이기 않기 때문에 이후 실험에서는 B 방법을 사용하였다.<br><br>
![image](https://user-images.githubusercontent.com/110075956/223383053-ab8c29ae-df15-49a4-8dfe-6fc509c44f90.png)<br>
더 깊은 layers는 "bottleneck"으로 block을 만들었다. Figure 5의 오른쪽 그림처럼 $1 \times 1$, $3 \times 3$, $1 \times 1$ convolutions로 구성되어 있으며 $1 \times 1$ layers는 dimensions를 줄이고 늘리는 역할을 한다. <br>
Table 4에서 보이는 것처럼 50/101/152-layer ResNets 모두 34-layer보다 유의미한 차이로 더 높은 정확도를 보였다. degradation problem이 보이지 않았고 모든 평가 지표에 대해 depth의 이점을 가졌다.<br><br>
![image](https://user-images.githubusercontent.com/110075956/223396552-e0182aad-4068-447f-9a21-a1b23e5170f9.png)<br>
152-layer ResNet의 single-model top-5 validaton error는 4.49%를 기록하였고 6개의 서로 다른 depth를 가진 모델을 결합한 networks의 top-5 error는 3.57%를 기록하였다.


# 4.2. CIFAR-10 and Analysis

# 4.3. Object Detection and PASCAL and MS COCO
