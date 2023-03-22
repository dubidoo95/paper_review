# 1. Introduction

지난 10년간 visual recognition tasks 분야에 대해 상당한 발전이 있었지만 object detection 분야의 발전은 더뎠다. neocognitron은 그 분야의 초기 모델로 생물학적 구조에서 따온 hierarchy와 shift-invariance를 특징으로 갖는 모델이다. 하지만 이 모델은 supervised training algorithm이 부족했다. <br>
본 논문에서는 CNN이 object detection 분야의 성능을 어떻게 끌어올렸는지 소개하였다. 이를 위해 두 가지 문제에 집중하였는데, 하나는 objects를 localizing하는 것이고 나머지 하나는 high-capacity model을 적은 양의 data로 학습시키는 것이다. <br>
image classification과 달리 object detection은 image 내 objects를 localizing하는 것이 필요하다. 오랫동안 이를 위해 sliding-window detector를 사용하였다. 하지만 매우 큰 receptive fields와 strides를 가진 모델에 적용하기 어려웠고 "recognition using resions"라는 새로운 개념을 도입하였다.<br>
![image](https://user-images.githubusercontent.com/110075956/225571645-10500c83-5dd2-433a-88c3-5732f4070123.png)<br>
본 논문에서 sliding-window를 사용한 OverFeat와 성능을 비교하였다. ILSVRC2013 detection dataset을 이용해 학습시켜본 결과 R-CNN이 OverFeat보다 약 7%p 더 높은 정확도를 보인 것으로 나타났다.<br>
다음으로 마주한 문제는 object detection의 data가 매우 적어 CNN을 학습시키기 어렵다는 것이었다. 이 문제를 unsupervised pre-training으로 해결하였다. 먼저 대용량의 dataset(ILSVRC)을 이용해 supervised pre-train시켰고 작은 dataset(PASCAL)을 이용해 domain-specific fine tuning을 하였다. 이 fine tuning은 mAP를 약 8%p 증가시켰다.

# 2. Object Detection with R-CNN

이 system은 세 개의 modules로 구성된다. 하나는 category-independent region proposals를 만드는 것이고 두번째는 각 region으로부터 fixed-length feature vector를 추출하는 convolutional neural network고, 나머지 하나는 class-specific linear SVMs이다. 

# 2.1. Module Design

R-CNN은 selective search를 사용하여 region proposals를 만든다.<br><br>
만들어진 각 resion proposals로부터 4096차원의 feature vector를 추출하였고 이 features는 다섯 개의 convolutional layers와 두 개의 fully connected layers를 통해 $227 \times 227$의 평균 차감된 RGB images를 forward propagating 함으로써 계산된다. <br>
가장 먼저 input image data를 CNN에서 쓸 size(여기서는 $227 \times 227$)로 변환한다. 많은 변환 가능한 임의의 모양을 가진 regions 중에서 가장 간단한 것을 선택한다. tight bounding box에 있는 모든 pixels를 필요한 size로 맞춘다. warping하기 전 주변 배경을 $p$ pixels(여기서는 $p=16$)만큼 남긴다.<br>
![image](https://user-images.githubusercontent.com/110075956/226237133-4616d255-1a1c-470d-a8b9-0d6eaaf1eca6.png)

# 2.2. Test-time Detection

test 단계에서 selective search를 통해 2000개의 region proposals를 만들었다. 각 proposal을 warping하고 forward propagation을 통해 features를 계산하였다. 이들 features를 SVN을 이용해 각 class별 score를 계산하고 이를 토대로 greedy non-maximum suppression을 적용하였다. 그 과정에서 intersection-over-union(IoU)이 학습된 임계점을 넘는 영역은 제거하였다.<br><br>
detection의 효율을 높이는 두 가지 특성이 있었다. 첫 번째는 모든 CNN parameters가 모든 categories를 공유하는 것이고 두 번째는 CNN으로 계산된 feature vectors들이 spatial paramids같은 다른 일반적인 approaches에 비해 low-demensional한 것이다. 그 결과 resion proposals와 features를 계산하는 데 걸리는 시간이 대폭 줄어들었다. 더불어 수천개의 classes, 심지어 100k classes까지도 별 다른 techniques 없이 확장이 가능했다.

# 2.3. Training

bounding-box labels data가 없었기때문에 image-level annotations만 사용하여 pre-train을 진행하였다. 그렇지만 이것만 가지고도 ILSVRC 2012에서 top-1 rate error를 기록하였는데 이는 training process의 간결함 덕분이었다.<br><br>
이후 새로운 task에 적용시키기 위해 domain-specific fine tuning을 진행하였다. warped region proposals만 사용하였고 optimizer로는 stochastic gradient descent(SGD)를 사용하였다. 1000-way classification layer를 ($N + 1$)-way classification layer로 바꾼 것을 제외하고는 모델은 변경하지 않았다. 여기서 $N$은 object classes의 수를 의미하고 1은 background를 의미한다. 또한 0.5 IoU 이상의 region proposals를 positive로, 나머지를 negative로 간주하였다. learning rate는 0.001로 설정하였고, 각 SGD iteration 마다 32개의 positive windows와 96개의 background windows를 sample하여 128개의 mini-batch를 만들었다.<br><br>
image region이 object를 완전히 감싸고 있을때는 positive로, image region에 아무것도 없을때는 negative로 판정하기 수월했지만 부분적으로 겹쳐있는 등 애매한 것들은 판정하기 어려웠다. 때문에 IoU overlap라는 개념을 도입하여, 특정 값 아래는 모두 negatives로 간주하였다. {0, 0.1, 0.2, 0.3, 0.4, 0.5} 범위 내에서 grid search해본 결과 overlab threshold가 0.3일때 가장 좋은 성능을 보였다. 앞서 fine tuning할 때는 threshold를 0.5로 설정하였었는데 fine tuning할 때와 training할 때 threshold를 같게 설정하면 오히려 성능이 떨어지는 결과를 보였다. 때문에 threshold를 각각 0.5, 0.3으로 설정하였다.<br><br>
이후 feature를 추출하여 linear SVM을 훈련시켰다. 훈련할 때 training data가 너무 커서 memory가 넘치는 현상을 보였기 때문에 hard negative mining method를 이용하여 이 문제를 해결하였다.

# 2.4. Results on PASCAL VOC 2010-12

![image](https://user-images.githubusercontent.com/110075956/226296168-0da8c0d9-ee3a-43e1-ae6e-fab77352e536.png)<br>
bounding-box regression을 사용한 것과 사용하지 않은 것을 각각 제출하였고 네 개의 다른 strong baselines인 DPM, UVA, Regionlets, SegDPM과 비교해보았다. 같은 region proposal algorithm을 사용한 UVA system과 비교해보았을때 월등한 성능을 보였고 다른 모델들과 비교해보았을 때도 가장 좋은 성능을 보였다. 

# 2.5. Results on ILSVRC2013 Detection

PASCAL VOC과 같은 system, 같은 hyperparameters를 사용하여 200개의 classes를 가진 ILSVRC2013 detection dataset에 실행시켜보았다. <br>
![image](https://user-images.githubusercontent.com/110075956/226302359-08ecce86-1137-4f22-aedf-ae11b94cf1fa.png)<br>
R-CNN은 31.4%의 mAP를 기록하며 종전 가장 좋은 모델이었던 OverFeat과 상당한 차이를 보였다. 

# 3. Visualization, Ablation, and Modes of Error

# 3.1. Visualizing Learned Features

첫번째 layer filters는 oriented edges와 opponent colors를 찾는다. 이후의 layers는 좀 복잡한데, 아이디어는 network에서 특정 unit을 선택하고 그 자체로 object detector인 것처럼 사용하는 것이다. 즉 region proposals에 대해 unit의 활성화를 계산하고, 그를 가장 높은 것부터 가장 낮은 것까지 정렬하고, non-maximum suppression을 수행한 다음 top-scoring regions를 표시한다. 이 방법은 선택된 unit이 "스스로 말할 수 있게" 한다.<br>
visualize는 pool5 layer에서 수행하였다. network의 마지막 convolutional layer의 output을 max-pool하는 layer로, $6 \times 6 \times 256$ = 9216-dimesional이다. boundary effects를 무시하고 각 pool5 unit은 $227 \times 227$ input에 대해 $195 \times 195$의 receptive field를 갖는다.<br>
![image](https://user-images.githubusercontent.com/110075956/226401564-3e257551-3387-41d1-8266-96e6c19f3607.png)<br>
위 Figure 4를 보면 network는 소수의 class-tuned features를 모양, 질감, 색상 및 재료 특성들과 함께 학습하는 것으로 보인다. 다음의 fully connected layer는 이러한 풍부한 features의 대규모 집합을 구성할 수 있는 능력을 가지고 있다.

# 3.2. Ablation Studies

어느 layers가 detection performance에 가장 중요한지 알아보기 위해 여러 실험을 진행하였다. <br>
![image](https://user-images.githubusercontent.com/110075956/226411871-2a0fba54-403e-4bb6-8c0b-4ddd04f408bd.png)<br>
먼저 fine-tuning을 하지 않고 layer-by-layer performance를 분석하였다. Table 2의 1-3행이 그것으로, fc7의 features가 fc6의 features보다 덜 일반화하는 것을 볼 수 있다. 이는 CNN의 parameters의 29%인 약 1680만개의 parameters를 mAP를 떨어뜨리지 않고 제거할 수 있음을 뜻한다. 더 놀라운 것은 fc6과 fc7을 모두 제거한 것은 제거하기 전의 단 6%의 parameters만 사용하면서도 꽤 좋은 성능을 보였다는 것이다. 위 실험을 통해 CNN의 representational power는 convolutional layers에서 온다는 것을 알 수 있었다.<br><br>
다음으로는 fine-tuning을 한 뒤 결과를 보았다. Table 2의 4-6행이 그것으로, fine-tuning은 mAP를 약 8.0%p 증가시킴을 보였다. 더불어 pool5보다 fc6와 fc7에서의 증가치가 훨씬 큰 것을 볼 수 있는데 이는 대부분의 향상이 domain-specific non-linear classifiers로부터 얻을 수 있음을 뜻한다.<br><br>
다른 learning methods를 가진 세 개의 모델(Table 2의 8-10행)과 비교해보았을 때에도 R-CNN이 훨씬 뛰어난 성능을 보였다. 특히 가장 최근 모델인 DPM v5에 대해서는 약 60% 가량의 성능 향상을 입증하였다.

# 3.3. Network Architectures

![image](https://user-images.githubusercontent.com/110075956/226541457-cf746d59-738e-4242-99a4-1e8c28e68ad4.png)<br>
지금까지는 대부분 CNN architecture로 AlexNet(T-Net)을 사용했는데 VGG16(O-Net)을 사용할 경우 mAP가 58.5%에서 66.0%로 상당히 많이 증가함을 볼 수 있다. 다만 VGG16을 사용할 시 AlexNet보다 계산 시간이 7배가량 더 걸린다는 단점이 있었다.

# 3.4. Detection Error Analysis

![image](https://user-images.githubusercontent.com/110075956/226543772-78d0b6f7-54f0-412e-a1ac-c6728f43bf23.png)<br>
![image](https://user-images.githubusercontent.com/110075956/226543816-938785e2-af70-48d0-855b-6ef075ccbd87.png)<br>
detection analysis tool을 이용해 R-CNN의 errors를 분석하였다. 

# 3.5. Bounding-box Regression

localization errors를 줄이기 위해 linear regression model이 새로운 detection window를 예측하도록 훈련시켰다. 이는 mAP를 3-4 points 정도 향상시키는 결과를 보였다.

# 3.6. Qualitative Results

![image](https://user-images.githubusercontent.com/110075956/226549854-1d4c3c65-9e97-43e8-adbf-42d2bd32aa90.png)<br>
![image](https://user-images.githubusercontent.com/110075956/226549888-7003d56f-525c-4267-bc7a-18c3e2409045.png)<br>
![image](https://user-images.githubusercontent.com/110075956/226549916-4dd4cc12-47de-482e-8152-f1b94181c951.png)<br>
![image](https://user-images.githubusercontent.com/110075956/226549972-ea4c41a8-cf34-4272-b6b2-2493cd183925.png)<br>
Figure 8과 Figure 9는 val set에서 무작위로 선별하여 나타내는 것이다. precision이 0.5를 넘는 모든 detections를 보여준다. 그리고 Figure 10과 Figure 11은 흥미롭고 놀라운 결과를 보여주는 몇 가지를 선별한 것이다. 여기도 마찬가지로 precision이 0.5를 넘는 모든 detections가 나타나있다.

# 4. The ILSVRC2013 Detection Dataset

ILSVRC 2013 detection dataset은 PASCAL VOC보다 덜 homogeneous하기 때문에 이를 어떻게 사용해야할지가 중요해서 별도의 section으로 다루었다.

# 4.1. Dataset Overview

train 395,918개, val 20,121개, test 40,152개로 이루어진 dataset이다. 이 images는 scene-like하고 PASCAL VOC images와 복잡도가 비슷하다. val과 test images는 bounding boxes와 함께 label이 모두 달려있다. 반면 train images는 복잡도가 다 다르고 label도 붙어있는 것이 있고 붙어있지 않는 것이 있다. 거기에 negative images도 있지만 본 논문에서는 사용하지 않았다. 그로 인해 train images에 hard negative mining을 적용하기 어렵다. <br>
여기서 사용한 방식은 val set을 주로 사용하고 train set을 보조적인 positive examples로 사용하는 것이다. classes가 최대한 균일하게 분리되도록 val1과 val2로 나누었다. 

# 4.2. Region Proposals

PASCAL에서 사용한 것과 같은 방법의 region proposals를 사용하였다. selective search는 train에는 사용하지 않고 val1, val2, test images에 사용하였다. ILSVRC images는 크기가 매우 작은 것부터 엄청 큰 것까지 다양하게 분포하고 있기 때문에 각 image를 500 pixels로 resize한 후 selective search를 적용하였다. 그 결과 image 당 평균 2403개의 region proposals가 도출되었다.

# 4.3. Training Data

images와 selective search, N개의 ground-truth boxes로 구성된 boxes를 한 set로 만들었다. N은 각각 $N \in \lbrace 0, 500, 1000 \rbrace$으로 진행하였다.<br>
R-CNN에서는 train data를 CNN fine-tuning, detector SVM training, bounding-box regressor training에서 사용한다. CNN fine-tuning은 PASCAL에서 했던 것과 동일한 세팅으로 50k SGD iteration만큼 진행하였다. SVM training에서는 모든 ground-truth boxes가 positive examples로 사용되었다. val1에서 무작위로 선별된 5000개의 images에 대해 hard negative mining이 수행되었다. bounding-box regressors도 마찬가지로 val1로 훈련하였다.

# 4.4. Validation and Evaluation

evaluation server에 결과를 제출하기 앞서 val2에 fine-tuning과 bounding-box regression의 효과를 검증해보았다. 모든 hyperparameters는 PASCAL에 사용한 것과 동일하게 설정하였다. 여러 조건에 따라 시험해본 후 val2에서 성능이 가장 좋았던 것을 골라 bounding-box regression을 사용한 것과 사용하지 않은 것을 각각 제출하였다.

# 4.5. Ablation Study

![image](https://user-images.githubusercontent.com/110075956/226911575-69e9114e-d7ea-4611-bd57-407aa357f25c.png)<br>
Table 4는 training data, fine-tuning, bounding-box regression에 ㅁ따른 결과를 나타낸 것이다. val2에 대한 mAP가 test에 대한 mAP와 매우 비슷한데 이는 val2가 성능을 평가하는데 매우 좋은 indicator인 것을 의미한다. 위 결과를 보면, 우선 val1만을 training data로 사용했을 때는 20.9%의 mAP를 기록하였다. training set을 val1 + train으로 확장시켰을 때는 mAP가 24.1%로 향상되었다. 이 때 N = 500일 때와 N = 1000일 때 차이를 보이지 않았다. fine-tuning과 bounding-box regression을 각각 추가하였을 때 성능이 소폭 향상되었다. 

# 4.6. Relationship to OverFeat

R-CNN과 OverFeat 간에 흥미로운 관계가 하나 있었는데, 만약 selective search region proposals를 regular square regions의 multi-scale pyramid로 대체하고 per-class bounding-box regressors를 single bounding-box regressor로 바꾼다면 system이 매우 유사하다는 것이다. OverFeat은 R-CNN보다 약 9배 가량 빠른데 이는 OverFeat에서는 sliding windows를 warping하지 않아 계산이 훨씬 간단하기 때문이다.

# 5. Semantic Segmentation

# 6. Conclusion
