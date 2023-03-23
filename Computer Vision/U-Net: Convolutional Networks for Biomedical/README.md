# 1. Introduction

지난 수년간 deep convolutional networks의 비약적인 발전이 있어 왔고 많은 visual recognition tasks에 사용되어 왔다. 
하지만 이는 image가 single class label을 가지는 classification tasks에 국한되었고 biomedical image processing과 같은 localization이 필요한 visual tasks에는 제약이 있었다.
Ciresan el al.이 localization 가능한 network를 소개하였고 EM segmentation에서도 우승했지만, 이는 속도가 느리고 localization accuracy와 the use of context 간에 trade-off가 있다는 결점이 있었다.<br><br>
본 논문에서는 "fully convolutional network"라는 것을 이용해 더 적은 training images를 가지고도 더 정확한 segmentations가 가능함을 보였다.<br>
![image](https://user-images.githubusercontent.com/110075956/222347539-228cd750-4ffa-4e04-9001-7733514f4e20.png)
전반적인 구조는 위와 같다.<br><br>
왼쪽 부분은 contracting path라 하여 context를 추출하며 feature map의 size가 줄어드는 구조를, 오른쪽 부분은 expanding path라 하여 더 정확한 localization을 가능케 한다.
main idea는 pooling operators 대신 upsampling operators를 사용한 일련의 layers를 통해 contracting path를 보충하는 것이다. 이를 통해 output의 resolution을 높일 수 있었다.<br><br>
upsampling part에 있는 많은 수의 feature channels는 network가 higher resolution layers로 context information을 전파하도록 한다. 
그 결과, contracting path와 expanding path는 대칭 구조(U-shaped architecture)를 가진다.
또한 fully connected layers를 사용하지 않고 convolutional layers만 사용하였기 때문에 full context가 그대로 담긴다.<br>
![image](https://user-images.githubusercontent.com/110075956/222359773-4843036a-09f8-4bb8-adca-e999ff419bec.png)
overlap-tile strategy를 사용해 임의의 큰 images를 원활하게 분할하였다. image의 경계부분을 예측하는 데 있어, 누락된 context는 input image를 mirroring 함으로써 사용되었다. 
이를 이용해 GPU memory에 제한되지 않고 large images에 network를 적용할 수 있었다.<br><br>
![image](https://user-images.githubusercontent.com/110075956/222367862-3bde935c-949d-495d-b15d-517406912e7f.png)
cell segmentation은 같은 class를 가진, 붙어있는 object를 분리하는 문제이다. 이를 해결하기 위해 저자는 인접한 cell 간에 separating background labels가 loss function에서 큰 가중치를 얻는 weighted loss의 사용을 제안한다.

# 2. Network Architecture

contracting path는 일반적인 convolutional network처럼, 3x3 convolutions, ReLU(rectified linear unit)로 구성된 layer 두 세트와 stride 값으로 2를 가진 2x2 max pool의 반복된 구조를 가진다. 각 downsampling step에서 feature channels의 수는 2배로 증가한다. <br>
expansive path는 contracting path에 대칭적인 모습을 보이며 max pool 대신 2x2 "up-convolution"을 가진다. 각 upsampling step에서 feature channels의 수는 2배 감소한다.<br>
마지막 layer에서는 1x1 convolution을 가지고, 최종적으로 23개의 convolutional layers가 사용되었다.

# 3. Training

input images와 그에 해당하는 segmentation maps가 network를 훈련하는 데 사용되었고 optimizer로는 SGD(Stochastic Gradient Descent)를 사용하였다. convolution layers에 padding을 사용하지 않았기 때문에 output size가 input size보다 작았으며 overhead를 최소화하고 GPU 사용을 최대화하기 위해 batch size를 크게 하기 보다는 input tiles를 크게 설정하였다. 따라서 이전 training samples가 current optimizer step에 많은 영향을 미치게 하기 위해 높은 momentum(0.99)을 적용하였다.<br><br>
최종 출력물에는 cross entropy loss function과 결합한 soft-max를 사용한다.
soft-max는 $$p_k(x) = {e^{a_k(x)}}/{\sum_{k'=1}^{K} e^{a_k(x)}}$$로 정의된다. 그리고 이를 토대로 나온 cross entropy loss는 $$E = \sum_{x \in \Omega}w(x)log(p_{l(x)}(x))$$이다.<br><br>
또한 각 ground truth segmentation의 weight map을 계산하여 적용시켰다. weight map을 구하는 공식은 $$w(x) = w_c(x) + w_0 \cdot e^{-\frac{{d_1(x) + d_2(x)}^2}{2\sigma^2}}$$이다. 

## 3.1. Data Augmentation

본 논문에서는 training data가 얼마 되지 않았기 때문에 data augmentation을 사용하였다. 3x3 grid에 random displacement vectors를 사용해 smooth deformations를 만들었다. displacements는 10 pixels strandard deviation을 가진 Gaussian distribution으로부터 샘플링되었다. 이후 bicubic interpolation으로 per-pixel displacements를 계산하였고, contracting path의 drop out layers를 통해 암시적인 data augmentation을 수행하였다.

# 4. Experiments

학습은 총 3개의 segmentation tasks, EM segmentation challenge와 ISBI cell tracking challenge 2014, 2015의 datasets를 이용해 진행하였다.<br><br>
![image](https://user-images.githubusercontent.com/110075956/222651418-e66d2c81-d6ef-433a-bb51-629dd203656e.png)
![image](https://user-images.githubusercontent.com/110075956/222651689-533cc886-f033-4d2c-ab7f-a15900ea675e.png)

EM segmentation challenge와 ISBI cell tracking challenge 모두 다른 networks보다 더 좋은 성능을 보였다.

# 5. Conclusion

U-Net architecture는 biomedical segmentation applications에 매우 좋은 성능을 입증하였다. elastic deformations를 이용하여 매우 적은 images만 있어도 학습이 가능했고 학습 시간도 상당히 짧았다.  
