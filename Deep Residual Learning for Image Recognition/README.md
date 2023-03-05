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



# 3.2. Identity Mapping by Shortcuts

# 3.3. Network Architectures

# 3.4. Implementation

# 4. Experiments

# 4.1. ImageNet Classification

# 4.2. CIFAR-10 and Analysis

# 4.3. Object Detection and PASCAL and MS COCO
