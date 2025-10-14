# Sigmoid

## 정의

**Sigmoid**는 입력값을 0과 1 사이의 값으로 변환하는 S자 곡선 형태의 [[Activation Function]]입니다. 출력을 확률로 해석할 수 있어 전통적으로 많이 사용되었습니다.

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

---

## 시각적 이해

Sigmoid 함수는 부드러운 S자 곡선을 그립니다:

```
  1 |--------
    |      /
0.5 |    /
    |  /
  0 |--------
      x
```

**주요 특징:**

- $x \to -\infty$: 출력 $\to 0$
- $x = 0$: 출력 $= 0.5$
- $x \to +\infty$: 출력 $\to 1$

**입출력 예시:**

|입력 $x$|출력 $\sigma(x)$|
|---|---|
|-5|0.007|
|-2|0.119|
|0|0.500|
|2|0.881|
|5|0.993|

---

## 수학적 성질

### 출력 범위

$$0 < \sigma(x) < 1$$

모든 실수 입력에 대해 출력이 0과 1 사이입니다. 이는 확률값으로 해석할 수 있어 유용합니다.

### 대칭성

$$\sigma(-x) = 1 - \sigma(x)$$

원점에 대해 중심 대칭입니다. 예를 들어 $\sigma(-2) = 0.119$이고 $\sigma(2) = 0.881$인데, $0.119 + 0.881 = 1$입니다.

### 미분

Sigmoid의 미분은 매우 깔끔한 형태입니다:

$$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$

이미 계산한 $\sigma(x)$ 값만으로 미분을 구할 수 있어 [[Backpropagation]]에서 효율적입니다.

**중요한 특성:**

- $x = 0$에서 최댓값: $\sigma'(0) = 0.25$
- $|x|$가 크면: $\sigma'(x) \to 0$

---

## 장점

### 확률로 해석 가능

출력이 0과 1 사이이므로 확률로 해석할 수 있습니다. 이진 분류에서 "양성일 확률"로 직접 사용 가능합니다.

예를 들어 $\sigma(x) = 0.83$이면 "83% 확률로 양성"으로 해석합니다.

### 부드러운 비선형성

[[ReLU]]처럼 급격하게 꺾이지 않고 모든 점에서 부드럽게 변합니다. 이는 수학적으로 다루기 쉽고 안정적입니다.

### 명확한 확률 해석

로지스틱 회귀에서 Sigmoid의 출력은 수학적으로 정확한 확률입니다:

$$P(y=1|x) = \sigma(w^Tx + b)$$

이는 통계학적으로 탄탄한 기반을 가집니다.

---

## 단점

### Gradient Vanishing 문제

Sigmoid의 가장 큰 문제점입니다.

입력의 절댓값이 크면 미분값이 거의 0이 됩니다:

- $\sigma'(-5) \approx 0.007$
- $\sigma'(5) \approx 0.007$

깊은 신경망에서 [[Chain Rule]]로 여러 층의 gradient를 곱하면:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial a_n} \cdot \sigma'(z_n) \cdot \cdots \cdot \sigma'(z_1) \cdot x$$

각 $\sigma'(z_i) < 0.25$이므로:

$$0.25^{10} \approx 9.5 \times 10^{-7}$$

10층만 되어도 [[Gradient]]가 거의 0이 되어 초기 층이 학습되지 않습니다.

### 계산 비용

지수 함수 $e^{-x}$를 계산해야 하므로 [[ReLU]]의 단순한 max 연산보다 느립니다. 대규모 네트워크에서는 이 차이가 누적됩니다.

### Zero-Centered 아님

Sigmoid의 출력은 항상 양수입니다. 이는 다음 층의 모든 gradient가 같은 부호를 가지게 만들어 학습을 지그재그로 비효율적으로 만듭니다.

예를 들어 모든 입력이 양수면 가중치는 모두 같은 방향으로만 업데이트됩니다.

### Saturation 문제

입력이 매우 크거나 작으면 출력이 0 또는 1에 "포화(saturate)"됩니다. 이 영역에서는:

- Gradient가 거의 0
- 학습이 멈춤
- 뉴런이 "죽은" 것과 유사한 상태

---

## 사용 가이드

### 이진 분류 출력층

Sigmoid의 가장 적절한 용도입니다.

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()  # 출력층에만
)
```

출력을 임계값 0.5와 비교하여 분류합니다:

- $\sigma(x) \geq 0.5$: 양성 (1)
- $\sigma(x) < 0.5$: 음성 (0)

**손실 함수:**

```python
criterion = nn.BCELoss()  # Binary Cross-Entropy
```

또는 더 안정적인:

```python
# Sigmoid를 모델에서 제거하고
criterion = nn.BCEWithLogitsLoss()  # 내부에서 Sigmoid 처리
```

### RNN/LSTM 게이트

순환 신경망의 게이트에서 0과 1 사이의 값이 필요할 때 사용합니다.

```python
class LSTMCell(nn.Module):
    def forward(self, x, hidden):
        # 게이트는 0~1 사이 값
        forget_gate = torch.sigmoid(self.W_f(x) + self.U_f(hidden))
        input_gate = torch.sigmoid(self.W_i(x) + self.U_i(hidden))
        output_gate = torch.sigmoid(self.W_o(x) + self.U_o(hidden))
        # ...
```

각 게이트는 "얼마나 정보를 통과시킬지"를 0~1 사이의 비율로 제어합니다.

### 은닉층에서는 사용하지 말 것

현대 딥러닝에서 은닉층에 Sigmoid를 사용하는 것은 권장하지 않습니다. Gradient vanishing 문제로 깊은 네트워크 학습이 어렵습니다.

**대신 사용:**

- [[ReLU]]: 대부분의 경우
- Tanh: Zero-centered가 필요한 경우
- GELU: Transformer 모델

---

## Sigmoid vs Tanh

Tanh는 Sigmoid의 zero-centered 버전입니다.

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

**Sigmoid:**

- 출력 범위: $(0, 1)$
- Zero-centered: 아님
- 용도: 이진 분류 출력

**Tanh:**

- 출력 범위: $(-1, 1)$
- Zero-centered: 맞음
- 용도: 은닉층 (하지만 ReLU가 더 나음)

Tanh는 Sigmoid를 스케일링하고 이동시킨 것과 같으므로, 같은 문제(gradient vanishing)를 가집니다.

---

## 역사적 맥락

### 초기 신경망 시대

1980~90년대에 Sigmoid는 표준 [[Activation Function]]이었습니다. 생물학적 뉴런의 반응 곡선과 유사하고, 수학적으로 깔끔했기 때문입니다.

### Vanishing Gradient의 발견

1990년대 중반, Sepp Hochreiter 등이 깊은 네트워크에서 gradient가 소멸하는 문제를 발견했습니다. 이는 Sigmoid의 한계를 명확히 보여줬습니다.

### ReLU의 등장

2011년 Alex Krizhevsky가 AlexNet에서 [[ReLU]]를 사용하여 ImageNet을 압도하면서 패러다임이 바뀌었습니다. 이후 은닉층에서 Sigmoid 사용은 급격히 감소했습니다.

### 현재의 위치

Sigmoid는 더 이상 범용 활성화 함수가 아니지만, 특정 용도에서는 여전히 필수적입니다:

- 이진 분류 출력층
- LSTM/GRU 게이트
- Attention 메커니즘의 일부

---

## 수치 안정성

### Sigmoid 계산의 문제

$x$가 매우 음수이면 $e^{-x}$가 overflow될 수 있습니다:

```python
import numpy as np

x = -1000
result = 1 / (1 + np.exp(-x))  # 오버플로우!
```

### 안정적인 구현

```python
def stable_sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (1 + exp_x)
```

PyTorch의 `torch.sigmoid()`는 이미 안정적으로 구현되어 있으므로 직접 구현하지 말고 사용해야 합니다.

---

## 실전 예제

### 이진 분류 모델

```python
import torch
import torch.nn as nn
import torch.optim as optim

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # 로짓만 반환
        return x

model = BinaryClassifier()

# BCEWithLogitsLoss 사용 (더 안정적)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

# 학습
for x, y in dataloader:
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y.float())
    loss.backward()
    optimizer.step()

# 예측
model.eval()
with torch.no_grad():
    logits = model(test_x)
    probs = torch.sigmoid(logits)  # 확률로 변환
    predictions = (probs > 0.5).float()
```

### Sigmoid 시각화

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 200)
sigmoid = 1 / (1 + np.exp(-x))
sigmoid_derivative = sigmoid * (1 - sigmoid)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, sigmoid)
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.grid(True)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, sigmoid_derivative)
plt.title("Sigmoid Derivative")
plt.xlabel('x')
plt.ylabel("σ'(x)")
plt.grid(True)
plt.axhline(y=0.25, color='r', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 핵심 요약

Sigmoid는 S자 곡선 형태의 [[Activation Function]]으로 출력을 0과 1 사이로 변환합니다. 확률로 해석 가능하여 이진 분류의 출력층에 적합하지만, gradient vanishing 문제로 은닉층에는 부적합합니다. 현대 딥러닝에서는 [[ReLU]]가 은닉층의 표준이 되었고, Sigmoid는 특정 용도(이진 분류, LSTM 게이트)에서만 사용됩니다.

---

## 관련 개념

**상위 개념:**

- [[Activation Function]] - 활성화 함수 전반

**비교 대상:**

- [[ReLU]] - 현대의 표준
- Tanh - Zero-centered 변형
- [[Softmax]] - 다중 클래스 확장

**함께 사용:**

- [[Loss Function]] - BCELoss
- [[Backpropagation]] - 역전파
- [[Gradient]] - 그래디언트

**기반 개념:**

- [[Forward Pass]] - 순전파
- [[Deep Learning Core Concepts]] - 딥러닝 기초