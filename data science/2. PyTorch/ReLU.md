# ReLU

## 정의

**ReLU (Rectified Linear Unit)** 는 현대 딥러닝에서 가장 널리 사용되는 [[Activation Function]]입니다. 입력이 양수면 그대로 통과시키고, 음수면 0으로 만드는 단순한 함수입니다.

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \ 0 & \text{if } x \leq 0 \end{cases}$$

---

## 시각적 이해

```
  y
  |     /
  |    /
  |   /
  |  /
--|------ x
  |
```

- $x > 0$: 45도 직선 (기울기 1)
- $x \leq 0$: 수평선 (출력 0)

**입력-출력 예시:**

|입력 $x$|출력 $\text{ReLU}(x)$|
|---|---|
|-2|0|
|-1|0|
|0|0|
|1|1|
|2|2|

---

## 왜 ReLU가 등장했는가?

### 전통적 활성화 함수의 문제

**[[Sigmoid]]와 Tanh의 한계:**

1. **Gradient Vanishing**: 입력의 절댓값이 크면 [[Gradient]]가 거의 0
    
    - Sigmoid: $\sigma'(x) \to 0$ as $|x| \to \infty$
    - Tanh: $\tanh'(x) \to 0$ as $|x| \to \infty$
2. **계산 비용**: 지수 함수 $e^x$ 계산 필요
    
3. **Not Zero-Centered** (Sigmoid): 모든 출력이 양수
    

### ReLU의 해결책

1. **Gradient가 사라지지 않음**: $x > 0$에서 $\text{ReLU}'(x) = 1$
2. **계산이 매우 빠름**: 단순한 max 연산
3. **희소성 제공**: 음수 입력을 0으로 만들어 일부 뉴런 비활성화

---

## 수학적 성질

### 미분

$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \ 0 & \text{if } x < 0 \ \text{undefined} & \text{if } x = 0 \end{cases}$$

**실전에서는**: $x = 0$일 때 미분을 0 또는 1로 정의하여 사용합니다. [[Backpropagation]]에서 큰 문제가 되지 않습니다.

### 선형성과 비선형성

- $x > 0$ 영역: 선형 함수
- 전체: 비선형 함수

이 특성 덕분에 선형 모델의 장점(간단함, 안정성)과 비선형 모델의 장점(표현력)을 모두 가집니다.

### 희소성 (Sparsity)

ReLU는 음수 입력을 0으로 만들어 **희소 활성화(sparse activation)**를 유도합니다.

예를 들어 100개 뉴런 중 평균적으로 50개만 활성화됩니다:

- 계산량 감소
- 과적합 방지
- 더 효율적인 표현

---

## 장점

### 1. 계산 효율성

$$\text{ReLU}(x) = \max(0, x)$$

비교 연산과 선택만으로 구현 가능합니다. Sigmoid나 Tanh처럼 지수 함수가 필요 없습니다.

**속도 비교:**

- ReLU: 1배 (기준)
- Sigmoid: ~10배 느림
- Tanh: ~10배 느림

### 2. Gradient Vanishing 완화

양수 영역에서 gradient가 1로 일정합니다:

$$\frac{\partial \text{ReLU}}{\partial x} = 1 \quad (x > 0)$$

깊은 신경망에서도 gradient가 소멸하지 않습니다:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial a_n} \cdot 1 \cdot 1 \cdot \cdots \cdot 1 \cdot x$$

Sigmoid의 $0.25^{10}$과 대비됩니다.

### 3. 생물학적 타당성

실제 뉴런도 임계값 이하에서는 반응하지 않고, 이상에서는 비례적으로 반응합니다. ReLU가 이를 근사합니다.

### 4. 희소 표현

음수 입력이 0이 되어 네트워크가 희소해집니다. 이는:

- 정보를 더 효율적으로 표현
- 과적합 위험 감소
- 계산량 감소

---

## 단점

### Dying ReLU 문제

ReLU의 가장 큰 문제점입니다.

**현상:**

- 뉴런의 입력이 항상 음수가 되면 출력이 항상 0
- Gradient도 항상 0이 되어 더 이상 학습 불가
- 뉴런이 "죽는다"

**발생 원인:**

1. **큰 학습률**: 파라미터가 크게 업데이트되어 음수 영역으로
2. **잘못된 초기화**: 초기 가중치가 너무 음수 편향
3. **음수 편향 데이터**: 입력 데이터가 음수로 치우침

**예시:**

은닉층에서 100개 뉴런 중 40개가 죽으면:

- 실질적으로 60개 뉴런만 사용
- 네트워크 용량 감소
- 표현력 저하

**해결 방법:**

- 적절한 학습률 사용
- He initialization (ReLU용 초기화)
- Leaky ReLU 등 변형 사용
- Batch Normalization

### Zero-Centered 아님

ReLU의 출력은 항상 0 이상입니다. 이는 다음 층의 gradient가 모두 같은 부호를 가지게 만들어 학습을 비효율적으로 만들 수 있습니다.

하지만 [[BatchNormalization]]을 사용하면 이 문제가 크게 완화됩니다.

---

## ReLU 변형들

### Leaky ReLU

$$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \ \alpha x & \text{if } x \leq 0 \end{cases}$$

일반적으로 $\alpha = 0.01$ 사용.

**장점:**

- Dying ReLU 문제 해결
- 음수 영역에서도 작은 gradient

**단점:**

- $\alpha$를 선택해야 함
- 성능 향상이 크지 않을 수 있음

### Parametric ReLU (PReLU)

$$\text{PReLU}(x) = \begin{cases} x & \text{if } x > 0 \ \alpha x & \text{if } x \leq 0 \end{cases}$$

Leaky ReLU와 비슷하지만 $\alpha$를 **학습 가능한 파라미터**로 만듭니다.

**장점:**

- 데이터에 맞게 자동으로 조정
- 더 나은 성능 가능

**단점:**

- 파라미터 수 증가
- 과적합 위험 증가

### Randomized Leaky ReLU (RReLU)

학습 시마다 $\alpha$를 무작위로 샘플링합니다:

$$\alpha \sim \text{Uniform}[l, u]$$

**장점:**

- 정규화 효과
- 과적합 감소

### ELU (Exponential Linear Unit)

$$\text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$$

**장점:**

- Zero-centered에 가까움
- 부드러운 음수 영역
- Dying ReLU 방지

**단점:**

- 지수 연산으로 느림

---

## 사용 가이드

### 언제 ReLU를 사용하는가?

**기본 선택:**

- 대부분의 [[Deep Learning Core Concepts]]에서 ReLU가 첫 선택
- CNN, MLP 등 거의 모든 구조

**특히 효과적인 경우:**

- 깊은 네트워크 (>10층)
- 대규모 데이터셋
- 계산 속도가 중요한 경우

### 언제 변형을 사용하는가?

**Leaky ReLU/PReLU:**

- Dying ReLU 문제가 관찰될 때
- 네트워크가 학습되지 않을 때

**ELU:**

- Zero-centered 출력이 중요할 때
- 계산 속도가 덜 중요할 때

**다른 활성화 함수:**

- 출력층: [[Sigmoid]] (이진 분류), [[Softmax]] (다중 분류)
- RNN 게이트: Sigmoid, Tanh
- Transformer: GELU

### PyTorch 구현

```python
import torch.nn as nn

# ReLU
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Leaky ReLU
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.LeakyReLU(0.01),
    nn.Linear(256, 10)
)

# PReLU
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.PReLU(),
    nn.Linear(256, 10)
)
```

---

## 가중치 초기화

ReLU를 사용할 때는 **He 초기화**를 권장합니다.

### He Initialization

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$$

여기서 $n_{\text{in}}$은 입력 뉴런 수입니다.

**이유:**

- ReLU는 음수 입력을 0으로 만들어 분산이 절반으로 줄어듦
- $\sqrt{2}$ 팩터가 이를 보상

**PyTorch:**

```python
import torch.nn.init as init

layer = nn.Linear(784, 256)
init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

### Xavier vs He

- **Xavier**: Sigmoid, Tanh용
    - $W \sim \mathcal{N}(0, \sqrt{1/n_{\text{in}}})$
- **He**: ReLU용
    - $W \sim \mathcal{N}(0, \sqrt{2/n_{\text{in}}})$

---

## Gradient 흐름

### Forward Pass

입력 $x = [-2, -1, 0, 1, 2]$에 대해:

$$\text{ReLU}(x) = [0, 0, 0, 1, 2]$$

음수가 모두 0으로 변환됩니다.

### Backward Pass

[[Chain rule]]에 의해:

$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \text{ReLU}(x_i)} \cdot \frac{\partial \text{ReLU}(x_i)}{\partial x_i}$$

- $x_i > 0$: gradient 통과 (곱하기 1)
- $x_i \leq 0$: gradient 차단 (곱하기 0)

이는 [[Backpropagation]]에서 일종의 게이트 역할을 합니다.

---

## 실전 팁

### 1. Dying ReLU 모니터링

학습 중에 얼마나 많은 뉴런이 죽었는지 확인:

```python
def count_dead_neurons(model, dataloader):
    dead_count = 0
    total_count = 0
    
    with torch.no_grad():
        for data, _ in dataloader:
            for module in model.modules():
                if isinstance(module, nn.ReLU):
                    output = module(data)
                    dead_count += (output == 0).sum().item()
                    total_count += output.numel()
    
    return dead_count / total_count
```

**경험 법칙**: 50% 이상이 죽으면 문제

### 2. 학습률 조정

Dying ReLU가 발생하면:

- 학습률을 낮춤 (예: 0.01 → 0.001)
- Learning rate scheduling 사용
- [[Optimizer(Data Science)]] 변경 (SGD → Adam)

### 3. Batch Normalization과 함께 사용

```python
nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),  # BN 추가
    nn.ReLU(),
    nn.Linear(256, 10)
)
```

[[BatchNormalization]]이 입력을 정규화하여 Dying ReLU 위험을 줄입니다.

### 4. Dropout과의 순서

```python
# 올바른 순서
nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),  # ReLU 후
    nn.Linear(256, 10)
)
```

[[Dropout]]은 활성화 함수 다음에 배치합니다.

---

## 역사적 배경

- **1969**: 처음 제안 (Kunihiko Fukushima)
- **2000년대**: 관심 감소 (Sigmoid, Tanh 선호)
- **2011**: Alex Krizhevsky가 AlexNet에 사용
- **2012**: ImageNet에서 압도적 성능
- **현재**: 사실상 표준 활성화 함수

AlexNet의 성공 이후 ReLU는 딥러닝의 핵심 요소가 되었습니다.

---

## 핵심 요약

- ReLU는 $\max(0, x)$로 정의되는 단순한 [[Activation Function]]
- 계산이 빠르고 gradient vanishing 문제 완화
- Dying ReLU 문제 주의 필요
- 대부분의 경우 첫 번째 선택지
- He 초기화와 함께 사용
- Leaky ReLU, PReLU 등 변형 존재

---

## 관련 개념

**상위 개념:**

- [[Activation Function]] - 활성화 함수 전반

**대안:**

- [[Sigmoid]] - 전통적 활성화 함수
- Leaky ReLU - Dying ReLU 해결
- ELU - 부드러운 변형

**함께 사용:**

- [[BatchNormalization]] - 정규화
- [[Dropout]] - 정규화
- [[Optimizer(Data Science)]] - 최적화

**기반 개념:**

- [[Forward Pass]] - 순전파
- [[Backpropagation]] - 역전파
- [[Gradient]] - 그래디언트