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

본 논문에서 사용한 약어들의 의미는 다음과 같다. <br>
Graph $G$는 $(N, E)$로 나타내고 $N$은 nodes의 집합을, $E$는 edges의 집합을 뜻한다. ne $[n]$은 $n$에 arc로 연결된 nodes를 뜻하며 co $[n]$은 $n$을 꼭지점으로 갖는 arcs의 집합을 뜻한다. node $n$과 edge $(n_1, n_2)$에 붙는 labels는 각각 $l_n \in R^{l_N}$, $l_{(n_1, n_2)} \in R^{l_E}$로 표현된다. $l$은 graph의 모든 labels를 합쳐 만든 vector를 의미한다. vector $y_S$는 $y$에서 $S$에 있는 nodes(혹은 edges)와 관련된 성분을 선택하여 얻은 벡터를 말한다.<br><br>
graph는 positional graph와 nonpositional graph로 나뉜다. nonpositional graph는 지금까지 설명해온 graph이고 nonpositional graph는 node $n$의 각 이웃에 그것의 위치를 가리키는 integer identifier가 있는 graph이다. 다시 말해 positional graph의 각 node $n$에 대해, injective function $v_n$: ne $[n]\rightarrow \lbrace 1, \dots, \vert N \vert \rbrace$가 존재하며, 각 function은 $n$의 각 이웃 $u$에 position $v_n(u)$를 할당한다.<br><br>
graphs의 집합을 $G$, 그들의 노드의 하위집합을 $N$이라 할 때 supervised learning framework를 다음과 같이 정의한다.
$$L = \lbrace (G_i, n_{i,j}, t_{i,j}|, G_i = (N_i, E_i) \in G; n_{i,j} \in N_i; t_{i,j} \in R^m, 1 \leq i \leq p, 1 \leq j \leq q_i \rbrace$$
여기서 $n_{i,j}$는 집합 $N_i$의 $j$번째 node를, $t_{i,j}$는 $n_{i,j}$에 관련된 target을 의미한다. 

# 2.1. The Model

nodes는 objects 혹은 concepts를, edges는 그들의 관계를 나타낸다. <br>
![image](https://user-images.githubusercontent.com/110075956/227529079-bac4de56-7156-4bc6-ae77-06967d3b22a1.png)<br>
Figure 2처럼 state $x_n \in R^s$를 각 node $n$에 붙였다. 그러면 state $x_n$은 $n$으로 표시되는 concept를 포함하고 output $o_n$를 만드는 데 사용될 수 있다. 그리고 $f_w$를 node $n$의 이웃에 대한 독립성을 나타내는 local transition function으로, $g_w$를 output이 어떻게 만들어지는지 나타내는 local output function $g_w$로 정의하였다. $x_n$와 $o_n$은 다음과 같이 표현될 수 있다.<br>
$x_n = f_w(l_n, l_{co[n]}, x_{ne[n]}, l_{ne[n]})$<br>
$o_n = g_w(x_n, l_n)$ (Equ. 1)<br>
여기서 $l_n, l_{co[n]}, x_{ne[n]}, l_{ne[n]}$은 각각 $n$의 label, edges의 labels, states, nodes의 labels이다. 위 식은 undirected graphs를 위해 만들어진 것으로, directed graphs의 경우엔 $d_l$과 같은 새로운 변수가 필요하다. <br><br>
모든 states, outputs, labels, node labels를 쌓아서 vectors로 만들고 위 식을 다시 쓰면 다음과 같아진다.<br>
$x = F_w(x, l)$<br>
$o = G_w(x, l_N)$ (Equ. 2)<br>
여기서 $F_w$는 global transition function, $G_w$는 global output function이 된다. Banach fixed point theorem에 따르면, 위 식은 $F_w$가 state에 따른 contraction map이라는 점에서 독특한 해를 가진다.<br>
Equ. 1을 더 살펴보면, 이는 positional과 nonpositional graphs에 모두 적용 가능하다. positional graphs의 경우 $f_w$는 neighbors의 positions를 추가적인 inputs으로 받는다. 실제로 $x_{ne[n]}, l_{co[n]}, l_{ne[n]}$이 neighbors의 positions에 따라 정렬되고 존재하지 않는 neighbors에 대해 특별한 null값으로 padding한다면 쉽게 얻을 수 있다. 반면 nonpositional graphs의 경우, $f_w$를 다음과 같이 대체한다.<br>
$x_n = \sum_{u \in ne[n]} h_w(l_n, l(n,u), x_u, l_u),    n \in N$ (Equ. 3)<br>
그리고 $h_w$는 parametric function이라 부르며 이는 positions와 children의 수에 영향을 받지 않는다. 

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
