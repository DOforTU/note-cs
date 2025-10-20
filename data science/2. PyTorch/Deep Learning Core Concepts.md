# Deep Learning Core Concepts

딥러닝의 핵심 개념을 이해하기 위한 가이드입니다. [[Machine Learning Basics]]를 먼저 학습한 후 이 문서를 읽는 것을 권장합니다.

---

## 개요

[[Machine Learning Basics]]에서 머신러닝이 손실을 최소화하며 학습한다는 것을 배웠습니다. 딥러닝(Deep Learning)은 이 아이디어를 **여러 층으로 쌓은 신경망**에 적용한 것입니다.

간단한 모델은 직선이나 단순한 곡선만 표현할 수 있지만, 층을 여러 개 쌓으면 훨씬 복잡한 패턴을 학습할 수 있습니다. 마치 레고 블록을 하나만 사용하면 단순한 모양만 만들 수 있지만, 수백 개를 쌓으면 복잡한 건축물을 만들 수 있는 것과 같습니다.

하지만 여기서 문제가 생깁니다. 층이 수십, 수백 개가 되면 각 층의 파라미터에 대한 [[Gradient]]를 어떻게 계산할까요? 일일이 손으로 계산하는 것은 불가능합니다. 바로 이 문제를 해결하는 것이 [[Backpropagation]]입니다.

---

## 신경망의 구조

신경망은 여러 층(layer)으로 구성됩니다. 각 층은 입력을 받아 간단한 계산을 수행하고 다음 층으로 전달합니다.

**간단한 2층 신경망:**

```
입력(x) → 층1 → 층2 → 출력(y)
```

각 층에서는 다음과 같은 계산이 일어납니다: $$h = f(W_1 x + b_1)$$ $$y = W_2 h + b_2$$

여기서:

- $W_1, W_2$: 가중치 행렬 (학습할 파라미터)
- $b_1, b_2$: 편향 (학습할 파라미터)
- $f$: 활성화 함수 (비선형 변환)
- $h$: 중간 결과 (hidden layer의 출력)

활성화 함수([[Activation Function]])가 없다면 여러 층을 쌓아도 결국 하나의 선형 변환에 불과합니다. 활성화 함수가 비선형성을 부여하여 복잡한 패턴을 학습할 수 있게 합니다.

---

## Forward Pass (순전파)

[[Forward Pass]]는 입력 데이터가 신경망의 첫 번째 층부터 마지막 층까지 순차적으로 통과하여 예측값을 계산하는 과정입니다.

### 작동 원리

데이터가 신경망을 통과하는 과정은 공장의 조립 라인과 비슷합니다. 원자재가 첫 번째 공정을 거치고, 그 결과물이 두 번째 공정으로 가고, 계속 이어져서 최종 제품이 나옵니다.

**예시: 손글씨 숫자 인식**

1. **입력층**: 28×28 픽셀 이미지 (784개 숫자)
2. **층1**: 784개 입력 → 128개 뉴런 → 활성화
3. **층2**: 128개 입력 → 64개 뉴런 → 활성화
4. **출력층**: 64개 입력 → 10개 출력 (0~9 숫자별 확률)

각 층을 통과할 때마다: $$\text{출력} = f(W \times \text{입력} + b)$$

이 과정을 순차적으로 실행하면 최종적으로 "이 이미지가 7일 확률은 95%"와 같은 예측을 얻습니다.

### 왜 중요한가?

Forward pass는 다음 두 가지 목적을 가집니다:

**1. 예측값 생성** 학습이 끝난 모델을 실제로 사용할 때, forward pass만 실행하여 빠르게 예측을 얻습니다.

**2. 학습을 위한 손실 계산** 예측값과 실제값을 비교하여 [[Loss Function]]을 계산합니다. 이 손실값이 있어야 [[Gradient]]를 계산하고 파라미터를 업데이트할 수 있습니다.

---

## Backpropagation (역전파)

[[Backpropagation]]은 신경망의 각 파라미터에 대한 [[Gradient]]를 효율적으로 계산하는 알고리즘입니다. "역전파"라는 이름은 손실에서 시작하여 출력층에서 입력층 방향으로 거꾸로 전파되기 때문에 붙었습니다.

### 왜 필요한가?

간단한 신경망을 생각해봅시다:

```
x → 층1(w1, b1) → 층2(w2, b2) → 층3(w3, b3) → y → Loss
```

[[Machine Learning Basics]]에서 배운 것처럼, 학습하려면 각 파라미터에 대한 손실의 미분, 즉 $\frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2}, \frac{\partial L}{\partial w_3}$ 등을 계산해야 합니다.

문제는 $w_1$이 손실에 미치는 영향이 층2와 층3를 거쳐서 나타난다는 것입니다. 직접적인 관계가 아니라 여러 단계를 거친 복잡한 관계입니다. 층이 10개, 100개가 되면 어떻게 계산할까요?

### 핵심 아이디어: Chain Rule

[[Chain rule]](연쇄 법칙)은 합성 함수의 미분을 계산하는 미적분 규칙입니다.

**간단한 예:** $y = f(g(x))$라면: $$\frac{dy}{dx} = \frac{dy}{dg} \times \frac{dg}{dx}$$

중간 단계의 미분들을 곱하면 전체 미분을 구할 수 있습니다.

**신경망에 적용:**

```
x → h1 → h2 → y → L
```

$w_1$이 $h_1$에 영향을 주고, $h_1$이 $h_2$에 영향을 주고, $h_2$가 $y$에 영향을 주고, $y$가 $L$에 영향을 줍니다. Chain rule을 사용하면:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial y} \times \frac{\partial y}{\partial h_2} \times \frac{\partial h_2}{\partial h_1} \times \frac{\partial h_1}{\partial w_1}$$

### 작동 과정

역전파는 다음 순서로 진행됩니다:

**1단계: Forward pass 실행** 모든 중간 결과를 저장하면서 순전파를 실행합니다. 나중에 gradient 계산에 필요하기 때문입니다.

**2단계: 출력층에서 gradient 계산** 손실 함수에서 시작합니다: $$\frac{\partial L}{\partial y}$$

**3단계: 뒤로 전파** Chain rule을 적용하여 한 층씩 거슬러 올라가며 gradient를 계산합니다: $$\frac{\partial L}{\partial w_3} \rightarrow \frac{\partial L}{\partial w_2} \rightarrow \frac{\partial L}{\partial w_1}$$

**4단계: 파라미터 업데이트** 계산된 gradient로 각 파라미터를 업데이트합니다.

### 효율성

만약 각 파라미터의 gradient를 독립적으로 계산한다면, 수백만 개의 파라미터가 있을 때 엄청나게 많은 계산이 필요합니다. 하지만 역전파는 한 번의 forward pass와 한 번의 backward pass만으로 모든 gradient를 계산합니다.

이는 마치 배달 경로를 최적화하는 것과 같습니다. 각 집을 개별적으로 방문하는 대신, 한 번의 경로로 모든 집을 효율적으로 방문하는 것입니다.

---

## Chain Rule (연쇄 법칙)

[[Chain rule]]은 역전파의 수학적 기초입니다. 복잡하게 연결된 함수의 미분을 간단한 부분들의 곱으로 계산할 수 있게 해줍니다.

### 직관적 이해

도미노를 생각해봅시다. 첫 번째 도미노를 밀면, 두 번째가 넘어지고, 세 번째가 넘어지고, 계속 이어집니다. 마지막 도미노가 얼마나 빨리 넘어질지는 각 단계의 속도를 모두 곱한 것입니다.

신경망도 마찬가지입니다. 입력의 작은 변화가 첫 번째 층에 영향을 주고, 그것이 두 번째 층에 영향을 주고, 최종적으로 손실까지 영향을 줍니다. 전체 영향은 각 단계의 영향을 곱한 것입니다.

### 수학적 표현

**1변수 합성 함수:** $y = f(u)$이고 $u = g(x)$라면: $$\frac{dy}{dx} = \frac{dy}{du} \times \frac{du}{dx}$$

**다변수 합성 함수:** $z = f(x, y)$이고 $x = g(t)$, $y = h(t)$라면: $$\frac{dz}{dt} = \frac{\partial z}{\partial x} \times \frac{dx}{dt} + \frac{\partial z}{\partial y} \times \frac{dy}{dt}$$

### 신경망에서의 적용

3층 신경망에서: $$x \xrightarrow{W_1} h_1 \xrightarrow{W_2} h_2 \xrightarrow{W_3} y \rightarrow L$$

첫 번째 층의 가중치 $W_1$에 대한 gradient는: $$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial y} \times \frac{\partial y}{\partial h_2} \times \frac{\partial h_2}{\partial h_1} \times \frac{\partial h_1}{\partial W_1}$$

각 항은 한 층에서의 미분이고, 이것들을 곱하면 전체 미분을 얻습니다. 이것이 역전파가 작동하는 원리입니다.

### 실용적 의미

Chain rule 덕분에:

- 복잡한 신경망도 단순한 부분들로 나누어 계산 가능
- 각 층을 독립적으로 설계하고 조합할 수 있음
- 자동 미분 시스템([[Autograd]])이 가능

---

## 자동 미분 (Automatic Differentiation)

현대 딥러닝 프레임워크는 **자동 미분**을 제공합니다. 개발자가 직접 gradient를 계산하는 코드를 작성할 필요가 없습니다.

### 계산 그래프

자동 미분 시스템은 forward pass 중에 **계산 그래프(computational graph)**를 구성합니다. 이는 수행된 모든 연산을 기록한 것입니다.

```
x → [×w] → [+b] → [ReLU] → h → [×w2] → y → [Loss]
```

각 연산은 노드가 되고, 데이터 흐름은 엣지가 됩니다. Backward pass에서는 이 그래프를 거꾸로 따라가며 [[Chain rule]]을 자동으로 적용하여 gradient를 계산합니다.

PyTorch의 [[Autograd]] 시스템이 바로 이 원리로 작동합니다. 사용자가 `loss.backward()`를 호출하면, 계산 그래프를 따라 자동으로 모든 파라미터의 gradient를 계산합니다.

### 장점

**정확성**: 수학적으로 정확한 gradient 계산 **효율성**: 최적화된 알고리즘으로 빠른 계산 **편의성**: 복잡한 모델도 쉽게 구현

---

## 학습 과정의 전체 흐름

딥러닝 모델의 학습은 다음 네 단계를 반복합니다:

**1. [[Forward Pass]]**

```python
# 입력 → 여러 층 → 출력
h1 = layer1(x)
h2 = layer2(h1)
y_pred = layer3(h2)
```

**2. [[Loss Function]] 계산**

```python
loss = (y_pred - y_true) ** 2
```

**3. [[Backpropagation]]**

```python
# Chain rule을 자동으로 적용하여 모든 gradient 계산
loss.backward()
```

**4. 파라미터 업데이트**

```python
# Gradient의 반대 방향으로 파라미터 조정
w = w - learning_rate * w.grad
```

이 과정을 수천, 수만 번 반복하면 모델은 점점 정확해집니다.

---

## 왜 딥러닝이 강력한가?

딥러닝이 다른 머신러닝 기법보다 강력한 이유는 **층을 쌓아 복잡한 특징을 자동으로 학습**하기 때문입니다.

**전통적인 방법:**

- 사람이 특징(feature)을 직접 설계
- 예: 이미지에서 "모서리", "질감", "색상" 같은 특징 추출
- 시간이 오래 걸리고 전문 지식 필요

**딥러닝:**

- 신경망이 데이터에서 특징을 자동으로 학습
- 초기 층: 단순한 패턴 (선, 모서리)
- 중간 층: 복잡한 패턴 (눈, 코, 귀)
- 마지막 층: 고수준 개념 (얼굴, 고양이, 개)

층을 쌓을수록 더 추상적이고 고수준의 특징을 학습할 수 있습니다. 이것이 딥러닝이 이미지 인식, 음성 인식, 자연어 처리 등에서 혁신적인 성능을 내는 이유입니다.

---

## 학습 순서

딥러닝 핵심 개념을 학습하는 권장 순서는 다음과 같습니다:

1. [[Forward Pass]] - 데이터가 신경망을 통과하는 과정
2. [[Chain rule]] - 역전파의 수학적 기초
3. [[Backpropagation]] - Gradient를 효율적으로 계산하는 알고리즘

각 개념은 서로 의존적입니다. [[Forward Pass]]로 예측값을 계산하고, [[Chain rule]]을 이용해 gradient를 구하며, [[Backpropagation]]으로 이를 효율적으로 실행합니다.

PyTorch를 사용한다면 [[Autograd]] 시스템이 이 모든 것을 자동으로 처리하지만, 내부 원리를 이해하는 것은 디버깅과 모델 설계에 큰 도움이 됩니다.

---

## 핵심 요약

- 딥러닝은 여러 층을 쌓아 복잡한 패턴을 학습
- [[Forward Pass]]로 예측값 계산
- [[Backpropagation]]으로 모든 파라미터의 gradient 계산
- [[Chain rule]]이 역전파의 수학적 기초
- 자동 미분 시스템이 이 모든 과정을 자동화
- 층을 쌓을수록 더 추상적인 특징을 학습 가능

---

## 관련 개념

**선수 지식:**

- [[Machine Learning Basics]] - 머신러닝의 기본 원리

**핵심 개념:**

- [[Forward Pass]] - 순전파
- [[Backpropagation]] - 역전파
- [[Chain rule]] - 연쇄 법칙
- [[Autograd]] - PyTorch의 자동 미분 (선택적)

**다음 단계:**

- [[Optimizer(Data Science)]] - SGD, Adam 등 파라미터 업데이트 알고리즘
- [[Neural Network Components]] - 신경망 구성 요소