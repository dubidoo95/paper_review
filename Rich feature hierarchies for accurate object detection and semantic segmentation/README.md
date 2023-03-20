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
먼저 fine-tuning을 하지 않고 layer-by-layer performance를 분석하였다. Table 2의 1~3행이 그것으로, fc7의 features가 fc6의 features보다 덜 일반화하는 것을 볼 수 있다. 이는 CNN의 parameters의 29%인 약 1680만개의 parameters를 mAP를 떨어뜨리지 않고 제거할 수 있음을 뜻한다. 더 놀라운 것은 fc6과 fc7을 모두 제거한 것은 제거하기 전의 단 6%의 parameters만 사용하면서도 꽤 좋은 성능을 보였다는 것이다. 위 실험을 통해 CNN의 representational power는 convolutional layers에서 온다는 것을 알 수 있었다.<br><br>
다음으로는 fine-tuning을 한 뒤 결과를 보았다. Table 2의 4~6행이 그것으로, fine-tuning은 mAP를 약 8.0%p 증가시킴을 보였다. 더불어 pool5보다 fc6와 fc7에서의 증가치가 훨씬 큰 것을 볼 수 있는데 이는 대부분의 향상이 domain-specific non-linear classifiers로부터 얻을 수 있음을 뜻한다.<br><br>


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
