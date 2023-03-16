# 1. Introduction

지난 10년간 visual recognition tasks 분야에 대해 상당한 발전이 있었지만 object detection 분야의 발전은 더뎠다. neocognitron은 그 분야의 초기 모델로 생물학적 구조에서 따온 hierarchy와 shift-invariance를 특징으로 갖는 모델이다. 하지만 이 모델은 supervised training algorithm이 부족했다. <br>
본 논문에서는 CNN이 object detection 분야의 성능을 어떻게 끌어올렸는지 소개하였다. 이를 위해 두 가지 문제에 집중하였는데, 하나는 objects를 localizing하는 것이고 나머지 하나는 high-capacity model을 적은 양의 data로 학습시키는 것이다. <br>
image classification과 달리 object detection은 image 내 objects를 localizing하는 것이 필요하다. 오랫동안 이를 위해 sliding-window detector를 사용하였다. 하지만 매우 큰 receptive fields와 strides를 가진 모델에 적용하기 어려웠고 "recognition using resions"라는 새로운 개념을 도입하였다.<br>
![image](https://user-images.githubusercontent.com/110075956/225571645-10500c83-5dd2-433a-88c3-5732f4070123.png)<br>
본 논문에서 sliding-window를 사용한 OverFeat와 성능을 비교하였다. ILSVRC2013 detection dataset을 이용해 학습시켜본 결과 R-CNN이 OverFeat보다 약 7%p 더 높은 정확도를 보인 것으로 나타났다.<br>
다음으로 마주한 문제는 object detection의 data가 매우 적어 CNN을 학습시키기 어렵다는 것이었다. 이 문제를 unsupervised pre-training으로 해결하였다. 먼저 대용량의 dataset(ILSVRC)을 이용해 supervised pre-train시켰고 작은 dataset(PASCAL)을 이용해 domain-specific 미세조정을 하였다. 이 미세조정은 mAP를 약 8%p 증가시켰다.

# 2. Object Detection with R-CNN

이 system은 세 개의 modules로 구성된다. 하나는 category-independent region proposals를 만드는 것이고 두번째는 각 region으로부터 fixed-length feature vector를 추출하는 convolutional neural network고, 나머지 하나는 class-specific linear SVMs이다. 

# 2.1. Module Design

R-CNN은 특정한 region proposal method에 구애받지 않지만 selective search를 사용하여 이전의 detection work와 통제된 비교를 가능하게 하였다.<br><br>
각 resion proposal로부터 4096차원의 feature vector를 추출하였고 이 features는 다섯 개의 convolutional layers와 두 개의 fully connected layers를 통해 $227 \times 227$의 평균 차감된 RGB images를 forward propagating 함으로써 계산된다. <br>
가장 먼저 input image data를 CNN에서 쓸 size(여기서는 $227 \times 227$)로 변환한다. 많은 변환 가능한 임의의 모양을 가진 regions 중에서 가장 간단한 것을 선택한다. tight bounding box에 있는 모든 pixels를 필요한 size로 맞춘다. 

# 2.2. Test-time Detection

# 2.3. Training

# 2.4. Results on PASCAL VOC 2010-12

# 2.5. Results on ILSVRC2013 Detection

# 3. Visualization, Ablation, and Modes of Error

# 3.1. Visualizing Learned Features

# 3.2. Ablation Studies

# 3.3. Network Architectures

# 3.4. Detection Error Analysis

# 3.5. Bounding-box Regression

# 3.6. Qualitative Results

# 4. The ILSVRC2013 Detection Dataset

# 4.1. Dataset Overview

# 4.2. Region Proposals

# 4.3. Training Data

# 4.4. Validation and Evaluation

# 4.5. Ablation Study

# 4.6. Relationship to OverFeat

# 5. Semantic Segmentation

# 6. Conclusion
