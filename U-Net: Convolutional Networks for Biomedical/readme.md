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

# 3. Training

## 3.1. Data Augmentation

본 논문에서는 training data가 얼마 되지 않았기 때문에 data augmentation을 사용하였다. 

# 4. Experiments

# 5. Conclusion
