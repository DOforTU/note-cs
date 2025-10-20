# Backpropagation

## 정의

**역전파(Backpropagation)** 는 신경망의 각 파라미터에 대한 [[Gradient]]를 효율적으로 계산하는 알고리즘입니다. "역전파"라는 이름은 손실 함수에서 시작하여 출력층에서 입력층 방향으로 거꾸로 전파되기 때문에 붙었습니다.

$$\text{손실}(L) \rightarrow \text{출력층} \rightarrow \cdots \rightarrow \text{층2} \rightarrow \text{층1}$$

신경망 학습의 핵심은 [[Loss Function]]을 최소화하는 파라미터를 찾는 것입니다. 역전파는 이를 위해 필요한 gradient를 단 한 번의 pass로 모두 계산합니다.

---

## 왜 필요한가?

간단한 3층 신경망을 생각해봅시다:

```
입력(x) → 층1(w1, b1) → 층2(w2, b2) → 층3(w3, b3) → 출력(y) → 손실(L)
```

[[Machine Learning Basics]]에서 배웠듯이, 학습하려면 각 파라미터에 대한 손실의 미분을 계산해야 합니다:

$$\frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2}, \frac{\partial L}{\partial w_3}, \frac{\partial L}{\partial b_1}, \frac{\partial L}{\partial b_2}, \frac{\partial L}{\partial b_3}$$

문제는 초기 층의 파라미터 $w_1$이 손실 $L$에 미치는 영향이 층2와 층3을 거쳐서 나타난다는 것입니다. 직접적인 관계가 아니라 여러 단계를 거친 복잡한 관계입니다.

만약 각 파라미터의 gradient를 독립적으로 계산한다면:

- 파라미터가 1만 개라면 1만 번의 계산 필요
- 파라미터가 1억 개라면 1억 번의 계산 필요

이는 실용적이지 않습니다. 역전파는 이 문제를 해결하여 **단 한 번의 forward pass와 한 번의 backward pass**로 모든 gradient를 계산합니다.

---

## 기본 아이디어

역전파의 핵심 아이디어는 [[Chain rule]](연쇄 법칙)을 체계적으로 적용하는 것입니다.

배달 경로를 최적화하는 상황을 생각해봅시다. 100개의 집에 각각 개별적으로 배달하러 가는 대신, 한 번의 최적 경로로 모든 집을 방문하는 것이 효율적입니다. 역전파도 마찬가지입니다. 각 파라미터의 gradient를 따로따로 계산하는 대신, 한 번의 역방향 pass로 모든 gradient를 수집합니다.

**Forward pass 중:**

- 각 층의 출력을 계산하면서 중간 결과를 저장
- 이 중간 결과들은 나중에 gradient 계산에 필요

**Backward pass 중:**

- 손실에서 시작하여 한 층씩 거슬러 올라감
- [[Chain rule]]을 적용하여 각 층의 gradient 계산
- 이전 층의 gradient 계산에 현재 층의 gradient 재사용

---

## 수학적 표현

간단한 2층 신경망으로 역전파를 이해해봅시다:

$$x \xrightarrow{W_1, b_1} z_1 \xrightarrow{f} h_1 \xrightarrow{W_2, b_2} z_2 \xrightarrow{f} h_2 \rightarrow L$$

여기서:

- $z_1 = W_1x + b_1$ (선형 변환)
- $h_1 = f(z_1)$ (활성화 함수 적용)
- $z_2 = W_2h_1 + b_2$ (선형 변환)
- $h_2 = f(z_2)$ (활성화 함수 적용)
- $L = \text{loss}(h_2, y)$ (손실 계산)

### Forward Pass

먼저 순방향으로 계산하며 모든 중간 결과를 저장합니다:

```python
z1 = W1 @ x + b1        # 저장
h1 = activation(z1)     # 저장
z2 = W2 @ h1 + b2       # 저장
h2 = activation(z2)     # 저장
loss = loss_fn(h2, y)
```

### Backward Pass

이제 역방향으로 gradient를 계산합니다.

**출력층에서 시작:** $$\frac{\partial L}{\partial h_2} = \frac{\partial L}{\partial h_2}$$

손실 함수를 직접 미분하여 계산합니다.

**두 번째 층의 gradient:**

활성화 함수의 gradient: $$\frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial h_2} \times \frac{\partial h_2}{\partial z_2}$$

가중치의 gradient: $$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial z_2} \times \frac{\partial z_2}{\partial W_2} = \frac{\partial L}{\partial z_2} \times h_1^T$$

편향의 gradient: $$\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial z_2}$$

**첫 번째 층으로 전파:**

다음 층으로 전달할 gradient: $$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial z_2} \times \frac{\partial z_2}{\partial h_1} = \frac{\partial L}{\partial z_2} \times W_2^T$$

활성화 함수의 gradient: $$\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial h_1} \times \frac{\partial h_1}{\partial z_1}$$

가중치의 gradient: $$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_1} \times x^T$$

편향의 gradient: $$\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial z_1}$$

---

## 구체적 예시

손실 함수가 평균 제곱 오차인 경우를 봅시다:

$$L = \frac{1}{2}(h_2 - y)^2$$

### Forward Pass

입력 $x = [1, 2]$, 정답 $y = 3$이고, 간단하게:

- $W_1 = [[0.5, 0.3], [0.2, 0.4]]$ (2×2)
- $b_1 = [0.1, 0.2]$
- $W_2 = [0.6, 0.8]$ (1×2)
- $b_2 = 0.1$
- 활성화 함수: [[ReLU]]

**계산 과정:**

```
z1 = W1 @ x + b1
   = [[0.5, 0.3], [0.2, 0.4]] @ [1, 2] + [0.1, 0.2]
   = [1.2, 1.2]

h1 = ReLU(z1) = [1.2, 1.2]

z2 = W2 @ h1 + b2
   = [0.6, 0.8] @ [1.2, 1.2] + 0.1
   = 1.78

h2 = ReLU(z2) = 1.78

L = 0.5 × (1.78 - 3)² = 0.74
```

### Backward Pass

**출력층:** $$\frac{\partial L}{\partial h_2} = h_2 - y = 1.78 - 3 = -1.22$$

**ReLU의 미분:** $$\frac{\partial h_2}{\partial z_2} = 1 \text{ (if } z_2 > 0\text{)}$$

**두 번째 층:** $$\frac{\partial L}{\partial z_2} = -1.22 \times 1 = -1.22$$

$$\frac{\partial L}{\partial W_2} = -1.22 \times [1.2, 1.2] = [-1.46, -1.46]$$

$$\frac{\partial L}{\partial b_2} = -1.22$$

**첫 번째 층으로 전파:** $$\frac{\partial L}{\partial h_1} = -1.22 \times [0.6, 0.8] = [-0.73, -0.98]$$

$$\frac{\partial L}{\partial z_1} = [-0.73, -0.98] \times [1, 1] = [-0.73, -0.98]$$

$$\frac{\partial L}{\partial W_1} = [[-0.73, -1.46], [-0.98, -1.96]]$$

$$\frac{\partial L}{\partial b_1} = [-0.73, -0.98]$$

이제 모든 파라미터의 gradient를 알았으므로, [[Gradient]] descent로 업데이트할 수 있습니다:

```python
learning_rate = 0.01

W2 -= learning_rate * dW2
b2 -= learning_rate * db2
W1 -= learning_rate * dW1
b1 -= learning_rate * db1
```

---

## Chain Rule의 적용

역전파가 작동하는 이유는 [[Chain rule]] 덕분입니다. 여러 함수가 합성된 경우, 전체 미분은 각 부분의 미분을 곱한 것입니다.

신경망에서: $$x \rightarrow f_1 \rightarrow f_2 \rightarrow f_3 \rightarrow L$$

첫 번째 함수의 파라미터에 대한 gradient는: $$\frac{\partial L}{\partial \theta_1} = \frac{\partial L}{\partial f_3} \times \frac{\partial f_3}{\partial f_2} \times \frac{\partial f_2}{\partial f_1} \times \frac{\partial f_1}{\partial \theta_1}$$

역전파의 핵심은 오른쪽 항부터 차례대로 계산하는 것입니다:

1. $\frac{\partial L}{\partial f_3}$ 계산 (출력층)
2. $\frac{\partial L}{\partial f_3} \times \frac{\partial f_3}{\partial f_2}$ 계산
3. 결과에 $\frac{\partial f_2}{\partial f_1}$ 곱하기
4. 최종적으로 $\frac{\partial f_1}{\partial \theta_1}$ 곱하기

이렇게 하면 중복 계산 없이 효율적으로 gradient를 구할 수 있습니다.

---

## 계산 그래프

역전파를 이해하는 또 다른 방법은 계산 그래프(computational graph)입니다. [[Forward Pass]] 중에 수행된 모든 연산을 그래프로 표현합니다.

**Forward pass 그래프:**

```
x ──×──> [W1] ──+──> [b1] ──ReLU──> h1 ──×──> [W2] ──+──> [b2] ──ReLU──> h2 ──MSE──> L
```

**Backward pass:** 이 그래프를 거꾸로 따라가며 각 노드에서 gradient를 계산합니다. 각 연산은 자신의 입력에 대한 gradient를 계산하는 방법을 알고 있습니다:

- **덧셈**: gradient를 그대로 전달
- **곱셈**: gradient에 다른 입력값을 곱함
- **ReLU**: 입력이 양수였던 곳만 gradient 전달
- **MSE**: $(h_2 - y)$ 형태로 gradient 계산

각 노드는 자신의 출력에 대한 gradient를 받아서, 자신의 입력에 대한 gradient를 계산하여 이전 노드로 전달합니다. 이 과정이 연쇄적으로 일어나는 것이 역전파입니다.

---

## 배치 처리에서의 역전파

실전에서는 여러 샘플을 동시에 처리합니다. 배치 크기가 $N$이라면:

**Forward pass:** $$Z_1 = XW_1^T + b_1 \text{ (shape: } N \times h_1\text{)}$$ $$H_1 = f(Z_1)$$

**Backward pass:**

각 샘플의 gradient를 계산한 후 평균을 냅니다: $$\frac{\partial L}{\partial W_1} = \frac{1}{N}\sum_{i=1}^{N} \frac{\partial L_i}{\partial W_1}$$

PyTorch는 이를 자동으로 처리하므로, 단일 샘플이든 배치든 같은 코드로 작동합니다.

---

## 역전파의 효율성

역전파가 효율적인 이유는 중복 계산을 제거하기 때문입니다.

**비효율적인 방법 (각 파라미터를 독립적으로 계산):**

- 파라미터 $p$개
- 각 gradient 계산에 forward pass 필요
- 총 $O(p \times \text{forward cost})$

**역전파:**

- 1번의 forward pass
- 1번의 backward pass (forward와 비슷한 비용)
- 총 $O(2 \times \text{forward cost})$

예를 들어 파라미터가 1억 개인 모델에서:

- 비효율적 방법: 1억 번 forward pass
- 역전파: 1번 forward + 1번 backward

속도 차이가 수억 배입니다. 역전파가 없었다면 현대의 딥러닝은 불가능했을 것입니다.

---

## 주의사항과 문제점

### Gradient 소실 (Vanishing Gradient)

깊은 신경망에서 [[Chain rule]]에 의해 여러 gradient가 곱해지면서 값이 점점 작아질 수 있습니다:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial h_n} \times \frac{\partial h_n}{\partial h_{n-1}} \times \cdots \times \frac{\partial h_2}{\partial h_1} \times \frac{\partial h_1}{\partial w_1}$$

각 항이 1보다 작으면 (예: 0.5), 층이 깊어질수록 gradient가 0에 가까워집니다: $$0.5^{10} \approx 0.001$$

초기 층의 파라미터는 거의 업데이트되지 않아 학습이 제대로 이루어지지 않습니다.

**해결 방법:**

- ReLU 같은 활성화 함수 사용
- Batch Normalization
- Residual Connection (ResNet)
- 적절한 가중치 초기화

### Gradient 폭발 (Exploding Gradient)

반대로 각 항이 1보다 크면 gradient가 폭발적으로 커집니다: $$2^{10} = 1024$$

파라미터가 불안정하게 변하고 손실이 `NaN`이 됩니다.

**해결 방법:**

```python
# Gradient Clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

작은 학습률과 적절한 가중치 초기화도 도움이 됩니다.

### 메모리 사용

역전파를 위해서는 forward pass의 모든 중간 결과를 저장해야 합니다. 층이 깊고 배치 크기가 크면 메모리 사용량이 급증합니다.

**메모리 사용량:** $$\text{Memory} \propto \text{layers} \times \text{batch size} \times \text{hidden size}$$

매우 깊은 네트워크에서는 gradient checkpointing 같은 기법을 사용하여 메모리를 절약할 수 있습니다.

---

## PyTorch에서의 구현

PyTorch는 역전파를 자동으로 처리합니다:

```python
import torch
import torch.nn as nn

# 모델 정의
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

# 입력과 정답
x = torch.tensor([[1.0, 2.0]])
y = torch.tensor([[3.0]])

# Forward pass
y_pred = model(x)

# 손실 계산
loss = ((y_pred - y) ** 2).mean()

# Backward pass (역전파 자동 실행)
loss.backward()

# 모든 파라미터의 gradient 확인
for name, param in model.named_parameters():
    print(f"{name}: {param.grad}")
```

`loss.backward()`를 호출하면:

1. 계산 그래프를 거꾸로 순회
2. [[Chain rule]]을 자동으로 적용
3. 모든 파라미터의 `.grad`에 gradient 저장

개발자는 역전파의 수학적 세부사항을 신경 쓸 필요 없이, 모델 구조만 정의하면 됩니다. 하지만 내부 원리를 이해하면 디버깅과 최적화에 큰 도움이 됩니다.

---

## 역전파의 역사적 의의

역전파 알고리즘은 1986년 Rumelhart, Hinton, Williams에 의해 대중화되었습니다. 이전에도 유사한 아이디어가 있었지만, 이들의 논문이 신경망 학습을 실용적으로 만들었습니다.

역전파 이전에는:

- 신경망이 2-3층으로 제한
- 각 파라미터를 독립적으로 최적화
- 계산 비용이 너무 커서 실용성 낮음

역전파 이후:

- 임의의 깊이의 신경망 학습 가능
- 효율적인 gradient 계산
- 현대 딥러닝의 기초

역전파는 딥러닝 혁명의 핵심 기술 중 하나입니다. Chain rule이라는 간단한 수학적 원리를 체계적으로 적용하여, 수억 개의 파라미터를 효율적으로 학습할 수 있게 만들었습니다.

---

## 핵심 요약

- 역전파는 신경망의 모든 파라미터에 대한 gradient를 효율적으로 계산
- [[Chain rule]]을 체계적으로 적용하여 한 번의 pass로 모든 gradient 계산
- Forward pass에서 중간 결과 저장, backward pass에서 gradient 계산
- 계산 그래프를 거꾸로 순회하며 각 노드에서 gradient 전파
- Gradient 소실/폭발 문제 주의 필요
- PyTorch의 [[Autograd]]가 자동으로 처리

---

## 관련 개념

**선수 지식:**

- [[Machine Learning Basics]] - 머신러닝 기본 원리
- [[Gradient]] - Gradient의 의미와 역할
- [[Forward Pass]] - 순전파 과정

**핵심 개념:**

- [[Chain rule]] - 역전파의 수학적 기초
- [[Deep Learning Core Concepts]] - 딥러닝 전체 구조

**다음 단계:**

- [[Autograd]] - PyTorch의 자동 미분 시스템
- [[Optimization Algorithms]] - Gradient를 사용한 최적화