# 원본 그래프 특성 기반 무작위 그래프 생성(ER, Configuration, Chung-lu, BA)
목표
원본 그래프의 특성을 보존, 활용하여 다양한 종류의 무작위 그래프 모델을 생성하는 것이 목표이다. 실제 네트워크와의 특성을 비교/분석하는 데 사용한다.

## 사용 방법
본 방법은 코랩 환경에서 실행하다고 가정하고 설명을 하겠다.
1. 패키지 설치
```
!git clone https://github.com/remote0612/random_network_maker.git
%cd ./random_network_maker/
```
2. 클래스 초기화
```
#그래프 예시 (파일 안에 있는 정치 도서 구매 그래프 사용할 경우)
import networks as nx
G = nx.read_gml('./random_network_maker/polbooks.gml')

from random_graph_analysis import *
maker= RandomGraphAnalysis(G)
```
3. 무작위 그래프 앙상블 생성
```
#그래프 이름, 시뮬레이션 수를 차례대로 입력한다.

# ex1. 100개의 ER 그래프 생성
er_ensemble_graph=maker.create_random_graph_ensemble("ER",100)

#ex 2. 20개의 chung-lu 그래프 생성
CL_ensemble_graph=maker.create_random_graph_ensemble("chunglu",20)
```

4. 차수 분포 분석
```
#기존 그래프 차수 분석
orignal=maker.degree_distribution()
print(orignal)

#무작위 그래프 차수 분석
ensemble=maker.ensemble_degree_distributions(CL_ensemble_graph)
print(CL_ensemble_graph)
```

## 클래스 설명
1. 초기화 및 정보 확인

 | 함수명 | 설명 |
| :--- | :--- |
| `__init__(self, G)` | 클래스를 초기화하고, 입력 그래프 $G$의 기본 속성을 설정한다. 모델에 사용하는 초기 매개변수도 함께 설정한다. |
| `random_graph_list(self)` | 생성 가능한 무작위 그래프 이름을 출력해준다. (ER, configuration, chunglu, BA)이 있다. |

3. ER 그래프를 만드는 함수
- ER 그래프란 ?\
 -> 원본 그래프의 $N$를 동일하게 하고, 균일 연결 확률 $p$를 이용해 노드를 무작위로 연결해놓은 그래프이다.
 -> G($N, p$) 모델을 사용한다.\
 -> $N$은 노드 수, $p$는 연결 확률을 의미한다.

 | 함수명 | 설명 |
| :--- | :--- |
| `set_init_ER(self)` | ER 그래프를 만들기 위한 엣지 연결 확률을 만드는 데, 사용하는 함수이다. 평균 차수와 노드수를 이용해 엣지 연결 확률을 계산하고 소수 2번째 자리까지 반환하도록 한다.<br> 엣지 연결 확률 계산은 노드 수 $N$과 평균 차수 $\langle k \rangle$을 사용하여 근사할 수 있다.<br> $$p \approx \frac{\langle k \rangle}{N-1}$$ 예외 : (1) $N$가 0 일 때와 (2) 엣지 수가 0 일 때는 연결 확률을 계산할 수 없도록 한다.|
| `create_ERnp_graph(self)` | 원본 그래프의 노드와 set_init_ER에서 만든 연결 확률 $p$ 이용해 ER 그래프를 만들도록 한다. <br> 예외 : 연결 확률 $p%가 0과 1 사이의 값이 아닌 경유 해당 그래프를 만들 수 없도록 한다.| 
  - 예외
    - (1) $N$가 0 일 때와 (2) 엣지 수가 0 일 때는 연결 확률을 계산할 수 없도록 한다.
  
  2) create_ERnp_graph
  - 원본 그래프의 노드와 set_init_ER에서 만든 연결 확률 $p$ 이용해 ER 그래프를 만들도록 한다.
  - 예외
    - 연결 확률 $p%가 0과 1 사이의 값이 아닌 경유 해당 그래프를 만들 수 없도록 한다.

3. Configuration 그래프를 만드는 함수
