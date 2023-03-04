# 1. Introduction
![image](https://user-images.githubusercontent.com/110075956/222880416-b0e49ab5-ce42-4dbb-8b69-ffb291935ff1.png)<br>
convolutional neural networks의 깊이는 매우 중요하고 깊은 networks는 좋은 성능을 이끌어낸다. 하지만 networks가 깊어질수록 한가지 문제를 수반하는데, 바로 vanishing/exploding gradients라는 것이다. 이 문제를 normalized initialization, intermediate normalization layers, stochastic gradient descent(SGD) 등으로 해결해왔지만 이내 degradation이라는 새로운 문제에 직면하였다. network의 깊이가 증가할수록 training error가 오히려 높아지는 모습을 보였다.<br><br>

![image](https://user-images.githubusercontent.com/110075956/222884756-15a484ff-3ef9-4575-8c91-60a5ed7ec8b1.png)<br>
본 논문에서는 이 degradation 문제를 해결하기 위해 deep residual learning framework를 도입하였다. 각 layers가 직접 쌓이는 대신, residual mapping을 하도록 설계하였다. 기본 mapping을 $H(x)$라 한다면, 또다른 nonlinear mapping $F(x) = H(x) - x$을 쌓았다. 이 때 original mapping을 다시 쓴다면 $H(x) = F(x) + x$로 쓸 수 있다. 그리고 $F(x) + x$는 위 그림에서 나오는 것처럼 "shortcut connections"를 이용하여 만들 수 있다.<br><br>
이 방법을 이용해 ImageNet, CIFAR-10의 dataset으로 network를 학습시켜본 결과 기존의 "plain" networks보다 optimize가 잘되고 train error 또한 낮게 나옴을 확인하였다. 그리고 ILSVRC 2015 classification competition에서 1등을 거머쥐었다. 이 network는 일반화 성능 또한 좋아서 ImageNet detection, ImageNet localization, COCO detection, COCO segmentation 등 다양한 competitions에서 우수한 성적을 거두었다.

# 2. Related Work

# 3. Deep Residual Learning

# 4. Experiments
