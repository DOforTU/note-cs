# Chain Rule

## 정의

**연쇄 법칙(Chain Rule)**은 합성 함수의 미분을 계산하는 미적분학의 기본 규칙입니다. 여러 함수가 연결되어 있을 때, 전체의 미분은 각 부분의 미분을 곱한 것과 같습니다.

1변수 합성 함수: $$\frac{dy}{dx} = \frac{dy}{du} \times \frac{du}{dx}$$

다변수 합성 함수: $$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \times \frac{\partial y}{\partial x}$$

딥러닝에서 chain rule은 [[Backpropagation]]의 수학적 기초입니다. 신경망의 각 층을 거치며 [[Gradient]]를 효율적으로 계산할 수 있게 해줍니다.

---

## 직관적 이해

Chain rule을 이해하는 가장 좋은 방법은 도미노를 생각하는 것입니다.

첫 번째 도미노를 살짝 밀면 두 번째가 넘어지고, 두 번째가 넘어지면 세 번째가 넘어지고, 계속 이어집니다. 마지막 도미노가 얼마나 빨리 넘어질지는 각 단계의 속도를 모두 곱한 것입니다.

- 첫 번째 → 두 번째 전달 속도: 0.8배
- 두 번째 → 세 번째 전달 속도: 0.9배
- 세 번째 → 네 번째 전달 속도: 0.7배

첫 번째 도미노를 밀었을 때 네 번째 도미노가 받는 영향: $$0.8 \times 0.9 \times 0.7 = 0.504$$

신경망도 똑같습니다. 입력의 작은 변화가 첫 번째 층에 영향을 주고, 그것이 두 번째 층에 영향을 주고, 최종적으로 손실까지 영향을 줍니다. 전체 영향은 각 단계의 영향을 곱한 것입니다.

---

## 기본 규칙

### 1변수 함수

$y = f(u)$이고 $u = g(x)$라면, $y$를 $x$의 함수로 볼 수 있습니다: $y = f(g(x))$

미분은 다음과 같이 계산됩니다: $$\frac{dy}{dx} = \frac{dy}{du} \times \frac{du}{dx}$$

**예시 1: 간단한 합성**

$y = (2x + 3)^2$을 미분해봅시다.

$u = 2x + 3$로 두면 $y = u^2$입니다.

$$\frac{dy}{du} = 2u$$ $$\frac{du}{dx} = 2$$

Chain rule 적용: $$\frac{dy}{dx} = 2u \times 2 = 4u = 4(2x + 3) = 8x + 12$$

직접 전개하여 미분하면: $$y = 4x^2 + 12x + 9$$ $$\frac{dy}{dx} = 8x + 12$$

같은 결과입니다!

**예시 2: 지수 함수**

$y = e^{x^2}$을 미분해봅시다.

$u = x^2$로 두면 $y = e^u$입니다.

$$\frac{dy}{du} = e^u$$ $$\frac{du}{dx} = 2x$$

Chain rule 적용: $$\frac{dy}{dx} = e^u \times 2x = 2xe^{x^2}$$

### 다변수 함수

$z = f(x, y)$이고 $x = g(t)$, $y = h(t)$라면:

$$\frac{dz}{dt} = \frac{\partial z}{\partial x} \times \frac{dx}{dt} + \frac{\partial z}{\partial y} \times \frac{dy}{dt}$$

각 경로를 통한 영향을 모두 더합니다.

**예시: 다변수 chain rule**

$z = x^2 + y^2$이고, $x = \cos(t)$, $y = \sin(t)$일 때 $\frac{dz}{dt}$를 구해봅시다.

$$\frac{\partial z}{\partial x} = 2x, \quad \frac{\partial z}{\partial y} = 2y$$ $$\frac{dx}{dt} = -\sin(t), \quad \frac{dy}{dt} = \cos(t)$$

Chain rule: $$\frac{dz}{dt} = 2x \times (-\sin(t)) + 2y \times \cos(t)$$ $$= 2\cos(t) \times (-\sin(t)) + 2\sin(t) \times \cos(t)$$ $$= -2\cos(t)\sin(t) + 2\sin(t)\cos(t) = 0$$

실제로 $z = \cos^2(t) + \sin^2(t) = 1$ (상수)이므로 미분이 0인 것이 맞습니다!

---

## 신경망에서의 적용

신경망은 여러 함수가 연쇄적으로 연결된 구조입니다:

$$x \xrightarrow{f_1} h_1 \xrightarrow{f_2} h_2 \xrightarrow{f_3} h_3 \rightarrow L$$

각 층에서:

- $h_1 = f_1(x, W_1, b_1)$
- $h_2 = f_2(h_1, W_2, b_2)$
- $h_3 = f_3(h_2, W_3, b_3)$
- $L = \text{loss}(h_3, y)$

### Gradient 계산

첫 번째 층의 가중치 $W_1$이 손실 $L$에 미치는 영향을 알고 싶습니다:

$$\frac{\partial L}{\partial W_1}$$

하지만 $W_1$은 $h_1$을 거쳐, $h_2$를 거쳐, $h_3$를 거쳐서 $L$에 영향을 줍니다. Chain rule을 적용하면:

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_3} \times \frac{\partial h_3}{\partial h_2} \times \frac{\partial h_2}{\partial h_1} \times \frac{\partial h_1}{\partial W_1}$$

각 항은 한 층에서의 미분입니다. 이것들을 곱하면 전체 미분을 얻습니다.

### 구체적 예시

간단한 2층 신경망으로 자세히 봅시다:

$$z_1 = W_1x + b_1$$ $$h_1 = \text{ReLU}(z_1)$$ $$z_2 = W_2h_1 + b_2$$ $$h_2 = \text{ReLU}(z_2)$$ $$L = \frac{1}{2}(h_2 - y)^2$$

$W_1$의 gradient를 계산하려면:

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_2} \times \frac{\partial h_2}{\partial z_2} \times \frac{\partial z_2}{\partial h_1} \times \frac{\partial h_1}{\partial z_1} \times \frac{\partial z_1}{\partial W_1}$$

**각 항 계산:**

1. $\frac{\partial L}{\partial h_2} = h_2 - y$
    
2. $\frac{\partial h_2}{\partial z_2} = \begin{cases} 1 & \text{if } z_2 > 0 \ 0 & \text{if } z_2 \leq 0 \end{cases}$ (ReLU의 미분)
    
3. $\frac{\partial z_2}{\partial h_1} = W_2$
    
4. $\frac{\partial h_1}{\partial z_1} = \begin{cases} 1 & \text{if } z_1 > 0 \ 0 & \text{if } z_1 \leq 0 \end{cases}$ (ReLU의 미분)
    
5. $\frac{\partial z_1}{\partial W_1} = x^T$
    

모두 곱하면 최종 gradient를 얻습니다.

---

## Backpropagation에서의 역할

[[Backpropagation]]은 chain rule을 효율적으로 적용하는 알고리즘입니다. 핵심 아이디어는 **gradient를 재사용**하는 것입니다.

### 비효율적인 방법

각 파라미터의 gradient를 독립적으로 계산:

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_3} \times \frac{\partial h_3}{\partial h_2} \times \frac{\partial h_2}{\partial h_1} \times \frac{\partial h_1}{\partial W_1}$$

$$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial h_3} \times \frac{\partial h_3}{\partial h_2} \times \frac{\partial h_2}{\partial W_2}$$

$\frac{\partial L}{\partial h_3} \times \frac{\partial h_3}{\partial h_2}$를 두 번 계산합니다 (중복!).

### 효율적인 방법 (Backpropagation)

거꾸로 계산하면서 중간 결과를 저장:

1. $\delta_3 = \frac{\partial L}{\partial h_3}$ 계산
    
2. $\delta_2 = \delta_3 \times \frac{\partial h_3}{\partial h_2}$ 계산 (저장)
    
3. $\delta_1 = \delta_2 \times \frac{\partial h_2}{\partial h_1}$ 계산 (저장)
    

이제 각 파라미터의 gradient는:

- $\frac{\partial L}{\partial W_2} = \delta_2 \times \frac{\partial h_2}{\partial W_2}$
- $\frac{\partial L}{\partial W_1} = \delta_1 \times \frac{\partial h_1}{\partial W_1}$

중복 계산 없이 한 번의 pass로 모든 gradient를 구합니다.

---

## 계산 그래프

Chain rule을 시각적으로 이해하는 방법은 계산 그래프입니다.

**Forward pass 그래프:**

```
x ──[×W1]──> z1 ──[ReLU]──> h1 ──[×W2]──> z2 ──[ReLU]──> h2 ──[Loss]──> L
```

**Backward pass:**

```
     ∂L/∂W1 <──────────────────┐
                                │
x <──[×W1]<── z1 <──[ReLU]<── h1 <──[×W2]<── z2 <──[ReLU]<── h2 <──[Loss]<── L
                                                                              │
                                                                            ∂L/∂L=1
```

각 노드는 다음을 수행합니다:

1. 출력에 대한 gradient를 받음 (오른쪽에서)
2. Chain rule 적용
3. 입력에 대한 gradient를 계산하여 전달 (왼쪽으로)

### 노드별 Gradient 계산

**곱셈 노드:** $$z = x \times w$$

Forward: $z = x \times w$ Backward:

- $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \times w$
- $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \times x$

**덧셈 노드:** $$z = x + y$$

Forward: $z = x + y$ Backward:

- $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \times 1 = \frac{\partial L}{\partial z}$
- $\frac{\partial L}{\partial y} = \frac{\partial L}{\partial z} \times 1 = \frac{\partial L}{\partial z}$

덧셈은 gradient를 그대로 전달합니다.

**ReLU 노드:** $$h = \max(0, z)$$

Forward: $h = \max(0, z)$ Backward: $$\frac{\partial L}{\partial z} = \begin{cases} \frac{\partial L}{\partial h} & \text{if } z > 0 \ 0 & \text{if } z \leq 0 \end{cases}$$

양수였던 곳만 gradient를 통과시킵니다.

---

## 다중 경로에서의 Chain Rule

하나의 변수가 여러 경로로 출력에 영향을 주는 경우, 모든 경로의 gradient를 더합니다.

**예시:**

```
      ┌──[×2]──> z1 ──┐
      │                │
x ────┤                ├──[+]──> y
      │                │
      └──[×3]──> z2 ──┘
```

$y = 2x + 3x = 5x$

**Chain rule로 계산:**

경로 1을 통한 gradient: $$\frac{\partial y}{\partial x}\bigg|_{\text{path 1}} = \frac{\partial y}{\partial z_1} \times \frac{\partial z_1}{\partial x} = 1 \times 2 = 2$$

경로 2를 통한 gradient: $$\frac{\partial y}{\partial x}\bigg|_{\text{path 2}} = \frac{\partial y}{\partial z_2} \times \frac{\partial z_2}{\partial x} = 1 \times 3 = 3$$

전체 gradient: $$\frac{\partial y}{\partial x} = 2 + 3 = 5$$

직접 미분: $\frac{d(5x)}{dx} = 5$ ✓

신경망에서 skip connection이나 residual connection이 있을 때 이런 상황이 발생합니다.

---

## 고차 미분

Chain rule은 2차 미분에도 적용됩니다. 이는 일부 최적화 알고리즘(Newton's method 등)에서 필요합니다.

1차 미분: $$\frac{dy}{dx} = \frac{dy}{du} \times \frac{du}{dx}$$

2차 미분: $$\frac{d^2y}{dx^2} = \frac{d}{dx}\left(\frac{dy}{dx}\right)$$

Chain rule을 다시 적용하여 계산합니다. PyTorch는 `grad`를 다시 미분하여 2차 미분도 계산할 수 있습니다:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3

# 1차 미분
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
print(dy_dx)  # 3x² = 12

# 2차 미분
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
print(d2y_dx2)  # 6x = 12
```

---

## 벡터와 행렬에서의 Chain Rule

신경망은 벡터와 행렬 연산을 사용합니다. Chain rule도 이에 맞게 확장됩니다.

**벡터 함수:** $$\mathbf{y} = f(\mathbf{x})$$

Jacobian 행렬: $$J = \begin{bmatrix} \frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \ \vdots & \ddots & \vdots \ \frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n} \end{bmatrix}$$

**Chain rule:** $$\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{y}} \times \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$$

행렬 곱셈으로 계산됩니다.

**신경망 예시:**

$$\mathbf{z} = W\mathbf{x} + \mathbf{b}$$

여기서 $W$는 $m \times n$ 행렬, $\mathbf{x}$는 $n$ 차원 벡터입니다.

Gradient: $$\frac{\partial L}{\partial \mathbf{x}} = W^T \frac{\partial L}{\partial \mathbf{z}}$$ $$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \mathbf{z}} \mathbf{x}^T$$

---

## 실전 예제

전체 과정을 종합하는 예제입니다.

**문제:** $L = (W_2 \times \text{ReLU}(W_1 x + b_1) + b_2 - y)^2$에서 $\frac{\partial L}{\partial W_1}$ 구하기

**주어진 값:**

- $x = 2$
- $y = 5$
- $W_1 = 3$, $b_1 = 1$
- $W_2 = 2$, $b_2 = 0$

**Forward pass:**

```
z1 = W1 * x + b1 = 3 * 2 + 1 = 7
h1 = ReLU(z1) = 7
z2 = W2 * h1 + b2 = 2 * 7 + 0 = 14
L = (z2 - y)² = (14 - 5)² = 81
```

**Backward pass (Chain rule 적용):**

1. $\frac{\partial L}{\partial z_2} = 2(z_2 - y) = 2(14 - 5) = 18$
    
2. $\frac{\partial z_2}{\partial h_1} = W_2 = 2$
    
3. $\frac{\partial h_1}{\partial z_1} = 1$ (ReLU, $z_1 = 7 > 0$)
    
4. $\frac{\partial z_1}{\partial W_1} = x = 2$
    

**Chain rule:** $$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_2} \times \frac{\partial z_2}{\partial h_1} \times \frac{\partial h_1}{\partial z_1} \times \frac{\partial z_1}{\partial W_1}$$ $$= 18 \times 2 \times 1 \times 2 = 72$$

**파라미터 업데이트 (learning rate = 0.01):** $$W_1^{\text{new}} = W_1 - 0.01 \times 72 = 3 - 0.72 = 2.28$$

---

## 왜 Chain Rule이 중요한가?

Chain rule 없이는 딥러닝이 불가능합니다:

**복잡한 함수의 미분:** 신경망은 수십, 수백 개의 함수가 합성된 구조입니다. Chain rule 없이 직접 미분하는 것은 사실상 불가능합니다.

**효율적인 계산:** [[Backpropagation]]은 chain rule을 체계적으로 적용하여, 수백만 개의 파라미터 gradient를 단 한 번의 pass로 계산합니다.

**자동 미분:** PyTorch의 [[Autograd]] 같은 자동 미분 시스템은 chain rule을 자동으로 적용합니다. 개발자는 복잡한 모델을 설계하기만 하면, 시스템이 알아서 gradient를 계산합니다.

**일반성:** Chain rule은 모든 미분 가능한 함수에 적용됩니다. 새로운 층이나 연산을 추가해도, chain rule로 gradient를 계산할 수 있습니다.

---

## 핵심 요약

- Chain rule은 합성 함수의 미분을 각 부분의 미분의 곱으로 계산
- 신경망의 gradient 계산에 필수적
- [[Backpropagation]]은 chain rule을 효율적으로 적용하는 알고리즘
- 여러 경로가 있으면 각 경로의 gradient를 더함
- 벡터와 행렬 연산에도 확장 가능
- PyTorch [[Autograd]]가 자동으로 적용

---

## 관련 개념

**선수 지식:**

- [[Machine Learning Basics]] - 머신러닝 기본 원리
- [[Gradient]] - Gradient의 의미

**관련 개념:**

- [[Backpropagation]] - Chain rule을 활용한 효율적 gradient 계산
- [[Forward Pass]] - 순전파 과정
- [[Deep Learning Core Concepts]] - 딥러닝 전체 구조

**다음 단계:**

- [[Autograd]] - PyTorch의 자동 미분 시스템