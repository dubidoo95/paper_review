# 1. Introduction

Data는 graph 형태로 그려질 수 있다. 하나의 node와 sequence로 이루어진 가장 단순한 형태부터 trees, acyclic graphs, cyclic graphs같은 복잡한 형태까지 다양하게 존재한다. function $\tau$는 이 structured data를 학습하는 것으로, graph $G$와 그것의 nodes $n$를 vector로 연결한다. 본 논문에서는 이를 두 가지로 나누었다. 하나는 graph-focused applications이고 나머지 하나는 node-focused applications이다. <br><br>
graph-focused applications에서 function $\tau$는 node $n$에 독립적으로, graph structured data set에 대해 classifier 또는 regressor로서 작용한다. <br>
![image](https://user-images.githubusercontent.com/110075956/227203532-26dd8526-94cc-487b-8990-eb02856d5e26.png)<br>
Fig. 1(a)처럼 chemical compound를 Graph $G$로, atoms를 graph의 nodes로, chemical bonds를 graph의 edges로 그릴 수 있다. 이 때 $\tau(G)$를 mapping함으로써 chemical compound가 특정 disease를 유발하는지 추정할 수 있다.<br> 
Fig. 1(b)는 image의 intensity가 균질한 지역을 nodes로 표시하고 인접 관계를 arcs로 그린 graph이다. 이 경우에는 contents에 따른 image 분류 작업으로 $\tau(G)$를 사용 수 있다.<br><br>
node-focused applications에서 $\tau$는 node $n$에 의존적이다. 다시 말해서 각 node의 특성에 따라 classification 혹은 regression을 수행한다.<br> 대표적인 예로 object detection이 있다. region adjacency graph의 nodes를 해당 영역이 object에 속하는 지에 따라 분류한다. 또 다른 예로 web page classification이 있다. Fig. 1(c)처럼 pages를 nodes로, pages 사이의 hyperlinks를 edges로 표현한다. 이를 통해 pages를 분류할 수 있다.<br><br>
기존의 machine learning applications는 graph structured information을 더 간단한 표현으로 mapping하는 전처리를 통해 graph structured data를 다루었다. 그러나 이러한 방식은 전처리 단계에서 중요한 정보들이 소실될 위험성이 있었다. 따라서 최근에는 graph structured data를 그대로 보존하며 문제에 접근하려는 시도들이 이어져왔다. 골자는 데이터 처리 단계에서 graph structured information을 통합하기 위해 nodes간의 위상적 관계를 사용하여 graph structured data를 encoding하는 것이다.<br><br>
recurrent neural networks와 Markov chain models는 위와 같은 방법을 사용하는 models이다. recurrent neural networks부터 살펴보면, directed acyclic graphs를 input domain으로 하여 function $\varphi_\omega$의 parameters $\omega$를 추정한다. 적절한 전처리를 거치면 node-focused applications로도 사용 가능하다. Markov chain models는 사건 간의 인과 관계가 graph로 표현되는 과정을 모방한다. <br><br>
본 논문에서는 graph와 node-focused applications에 모두 적합한 supervised neural network model을 소개한다. 이 model은 위 두 models를 공통의 framework에서 통합하며 저자는 graph neural network(GNN)라는 이름을 붙였다. 즉, GNN은 recursive neural network와 Markov chain model의 특성을 혼합한 model이다.<br><br>
GNN은 information diffusion mechanism을 사용한다. graph는 graph의 연결에 따라 연결된 단위 집합에 의해 처리된다. 이 단위는 평형에 도달할 때까지 상태를 갱신하고 정보를 교환한다. 그 후 GNN의 output은 단위 상태를 기반으로 각 node에서 계산된다.

# 2. The Graph Neural Network Model

# 2.1. The Model

# 2.2. Computation of the State

# 2.3. The Learning Algorithm

# 2.4. Transition and Output Function Implementations

# 2.5. A Comparison With Random Walks and Recursive Neural Networks

# 3. Computational Complexity Issues

# 3.1. Complexity of Instructions

# 3.2. Time Complexity of the GNN Model

# 4. Experimental Results

# 4.1. The Subgraph Matching Problem

# 4.2. The Mutagenesis Problem

# 4.3. Web Page Ranking

# 5. Conclusion
