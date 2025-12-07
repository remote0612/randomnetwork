import networkx as nx
import numpy as np
import random
import itertools

class RandomGraphAnalysis():
  def __init__(self, G: nx.Graph):
    # 원본 그래프와 기본적으로 사용되는 속성을 클래스 attribute로 생성
    self.Graph = G
    self.nodes = G.nodes(data=True)
    self.edges = G.edges(data=True)
    self.number_of_nodes = G.number_of_nodes()
    self.number_of_edges = G.number_of_edges()
    self.degree_dict = dict(G.degree())
    self.degree_seq = [d for _, d in G.degree()]

    # ER 그래프에서 사용될 기본 엣지 연결 확률 생성
    self.set_init_ER()
    # BA 그래프에서 사용될 기본 초기 노드수와 새로운 노드가 가지고 올 미연결 링크수 속성 생성
    self.set_init_BA()

    # 클래스를 이용해 생성할 수 있는 무작위 그래프 종류 출력
    self.random_graph_list()

  def random_graph_list(self):
    '''
    해당 클래스를 이용해 ensemble로 생성 가능한 무작위 그래프의 이름을 출력하는 함수
    '''

    print("random graph ensemble 생성에 사용 가능한 무작위 그래프는 다음과 같습니다: ER, configuration, chunglu, BA")

  def set_init_ER(self):
    '''
    ER graph 생성에 사용될 엣지 연결 확률을 계산하는 함수

    담당자: 장해린
    예외처리: (1) 노드수 0 초과 확인 (2) 엣지수 0 초과 확인
    '''

    if self.number_of_edges == 0 or self.number_of_nodes == 0:
      raise ValueError("노드와 엣지수가 0이 아닌 그래프를 넣어주세요")

    # 연결확률 = 평균차수 / (노드수 - 1)
    mean_k=2*self.number_of_edges/self.number_of_nodes
    max_k=(self.number_of_nodes-1)

    # 소수 둘째 자리 까지 반환하도록 함
    prob=round(mean_k/max_k,2)
    self.p = prob

  def create_ERnp_graph(self):
    '''
    G(n,p)의 Erdős–Rényi model을 이용한 그래프 생성 함수

    담당자: 장해린
    예외처리: (3) 확률 유효성 확인 (0과 1 사이)
    '''

    if self.p <= 0 or self.p >= 1:
      raise ValueError("확률은 0과 1 사이로 넣어주세요")

    new_graph=nx.Graph()
    new_graph.add_nodes_from(self.nodes) #같은 노드 추가 (특성이 있어서 노드 추가를 직접해주어야함)

    for i, j in itertools.combinations(new_graph.nodes, 2):
      r = random.random() #확률 r을 추출
      if r < self.p: #임의의 숫자 r이 p보다 작은 경우에 연결 (연결 여부)
        new_graph.add_edge(i,j) #엣지 추가

    return new_graph

  def create_config_graph(self):
    '''
    Configuration graph 생성 함수

    담당자: 이해정
    예외처리: (1) 차수 개수 짝수 확인 (2) 차수 0 이상 확인
    '''

    # error for exception
    # (1) graphicable: length of degree sequence must be even
    if sum(self.degree_seq)%2 == 1:
      raise ValueError("차수의 합은 짝수여야 합니다.")
    # (2) degrees must be over 0
    if any(d < 0 for d in self.degree_seq):
      raise ValueError("차수는 음수일 수 없습니다.")

    # create stub list from degree sequence
    stub_list = []
    for node, degree in self.degree_dict.items():
      stub_list.extend([node] * degree)

    # shuffle
    random.shuffle(stub_list)

    # link edge randomly in empty graph
    config_graph = nx.Graph()
    config_graph.add_nodes_from(self.nodes)

    # random link
    while len(stub_list) > 1:
      # (1) choose random node pair
      node1 = stub_list.pop() # shuffled
      node2 = stub_list.pop()

      # (2) except self-loop
      while node1 == node2 and len(stub_list) > 0:
        stub_list.append(node2)
        random.shuffle(stub_list) # shuffle-randomize
        node2 = stub_list.pop() # node1 != node2

      # (3) except multi-edge
      if config_graph.has_edge(node1, node2):
        continue

      # (4) link
      if node1 != node2:
        config_graph.add_edge(node1, node2)

    return config_graph

  def create_chunglu_graph(self):
    '''
    Chung-Lu graph 생성 함수

    담당자: 김시현
    예외처리: (1) 차수 0 이상 확인 (2) 차수 총합 0 아님 확인 (3) 연결 확률 1 미만 확인
    '''

    total_degree = sum(self.degree_seq)

    # 1. 예외 처리: 입력 유효성 검사

    # 1-1. 음수 Degree 체크
    if any(d < 0 for d in self.degree_seq):
      raise ValueError("Degree는 0 미만의 음수일 수 없습니다.")

    # 1-2. 총합 0 체크
    try:
      sorted_degrees = self.degree_seq[:]
      sorted_degrees.sort(reverse=True)

      W_max = sorted_degrees[0]
      if len(sorted_degrees) < 2:
        W_second_max = 0
      else:
        W_second_max = sorted_degrees[1]

      max_prob = (W_max * W_second_max) / total_degree

    # 총합 0 체크
    except ZeroDivisionError as e:
      raise ValueError("Total Degree가 0입니다. 엣지를 생성할 수 없습니다.",e)

    except IndexError as e: # Degree Sequence가 빈 리스트일 때 발생하는 오류
      raise ValueError("Degree Sequence가 비어있습니다. 그래프를 생성할 수 없습니다. 오류",e)

    # 1-3. 연결 확률 1 이상 체크: 1보다 크거나 같은 경우 최대 차수곱으로 정규화
    if max_prob >= 1.0:
      Z = (W_max * W_second_max) + 0.1
    else:
      Z = total_degree

    # 2. 그래프 생성
    G = nx.Graph()
    G.add_nodes_from(self.nodes)

    for i, j in itertools.combinations(G.nodes(), 2):
      p_ij = (self.degree_dict[i] * self.degree_dict[j]) / Z
      if random.random() < p_ij:
        G.add_edge(i, j)
    return G

  '''
  이하 담당자: 이해정
  '''
  def set_init_BA(self):
    '''
    BA graph 생성에 사용될 초기 노드수와 새로 추가하는 노드가 가지고 올 미연결 엣지수를 계산하는 함수

    예외처리: 빈 그래프 확인
    '''

    if self.number_of_edges == 0 or self.number_of_nodes == 0:
      raise ValueError("노드와 엣지수가 0이 아닌 그래프를 넣어주세요")

    # initial number of nodes
    if round(0.1 * self.number_of_nodes) < 5:
      self.m0 = round(0.1 * self.number_of_nodes)
    else:
      self.m0 = 5

    # number of edges for each new node
    if self.number_of_edges < 2:
      self.m = 1
    else:
      self.m = 2

  # 연결 노드 선택 함수 정의
  def choose_target_node(self, existing_nodes:list, G:nx.Graph):
    '''
    BA graph 생성 과정 중 새로 들어온 노드가 엣지를 연결할 노드를 고르는 함수
    그래프의 현재 노드 리스트와 그래프 자체를 매개변수로 받음
    '''

    # Preferential attachment: probability * degree
    degrees = [d for _, d in G.degree()] # 존재하는 노드들의 이웃수 리스트
    total_degree = sum(degrees)
    if total_degree == 0:
      return random.choice(existing_nodes)
    probs = [d/total_degree for d in degrees] # 각 이웃수의 전체 이웃수 대비 비율로 확률 리스트 형성

    # 기존 존재하는 노드들 중 하나를 확률에 비례하게 선택.
    # 축적 확률 사용하여 선택
    total = sum(probs) # 총 확률 리스트의 합: [0.2, 0.1, 0.3]
    cumulative_weights = [] # 하나씩 축적하여 웨이트 리스트 [0.2, 0.3, 0.6] 만들 예정
    cum_sum = 0

    for w in probs :
      cum_sum += w # 축적된 확률
      cumulative_weights.append(cum_sum) # 축적된 확률을 리스트로 추가함.

    r = random.uniform(0, total) # 균등 확률로 뽑은 지점이
    for i, cw in enumerate(cumulative_weights): # 축적 확률의 영역 이내인지 확인
      if r <= cw and r > min(0, cumulative_weights[i-1]):
        return existing_nodes[i]

  def create_BA_graph(self):
    '''
    BA graph 생성 함수 (노드 특성은 유지되지 않음)
    '''

    G = nx.complete_graph(self.m0) # fully connected Graph w/ m0 nodes

    n_total = self.number_of_nodes  # desired total number of nodes
    next_node = self.m0             # next node to add

    while next_node < n_total:
      targets = [] # 총 m 개만큼 타겟노드 선택할 초기 비어있는 데이터집합.
      while len(targets) < self.m: # target 이 m 개 채워질 때까지.
        chosen = self.choose_target_node(list(G.nodes()), G) # 함수 사용하여 타겟 1 선택
        if chosen not in targets : # 이미 뽑은 노드가 아니라면 추가
          targets.append(chosen)
      for target in targets:
        G.add_edge(next_node, target) # 다음 연결 노드인 next_node 와 선택된 타겟 노드들 순차적으로 연결
      next_node += 1

    return G

  def create_random_graph_ensemble(self, random_graph="ER", num_simulations=100):
    '''
    무작위 그래프 생성 모델을 골라 여러 그래프를 ensemble로 생성하는 함수
    '''

    # setting random graph
    if random_graph == "ER":
      create_random_graph = self.create_ERnp_graph
    elif random_graph == "configuration":
      create_random_graph = self.create_config_graph
    elif random_graph == "chunglu":
      create_random_graph = self.create_chunglu_graph
    elif random_graph == "BA":
      create_random_graph = self.create_BA_graph
    else:
      self.random_graph_list()
      raise ValueError("유효한 무작위 그래프 이름이 아닙니다.")

    # create ensemble
    random_models = []

    for _ in range(num_simulations):
      # 1) random model generation
      tmp_net = create_random_graph()
      # 2) except self-loop, multi-edge
      simple_net = nx.Graph(tmp_net)
      random_models.append(simple_net)

    return random_models

  def degree_distribution(self, graph=None):
    '''
    그래프를 입력 받아 해당 그래프의 차수 분포 array를 반환하는 함수
    '''

    if not graph:
      graph = self.Graph

    # k seq list of a random graph
    degree_sequence = [d for _, d in graph.degree()]
    # list -> freq
    degree_distribution = np.histogram(degree_sequence, bins=range(max(self.degree_seq)+2), density=True)[0]

    return degree_distribution

  def ensemble_degree_distributions(self, ensemble_graphs:list):
    '''
    ensemble 그래프 리스트를 입력 받아 차수 분포 array들의 리스트를 반환하는 함수
    '''

    degree_dists = []

    for graph in ensemble_graphs:
      degree_distribution = self.degree_distribution(graph)
      # combine random graph degree dists
      degree_dists.append(degree_distribution)

    return degree_dists
