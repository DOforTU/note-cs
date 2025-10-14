# LogSoftmax

## 정의

**LogSoftmax**는 [[Softmax]] 함수의 결과에 로그를 취한 함수입니다. Softmax를 먼저 계산하고 로그를 취하는 대신, 수치적으로 안정적인 방식으로 직접 계산합니다.

$$\text{LogSoftmax}(z_i) = \log(\text{softmax}(z_i)) = \log\left(\frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}\right)$$

수학적으로 간단히 하면:

$$\text{LogSoftmax}(z_i) = z_i - \log\left(\sum_{j=1}^{K} e^{z_j}\right)$$

여기서:

- $z_i$: $i$번째 클래스의 로짓 값
- $K$: 전체 클래스 개수

---

## 왜 필요한가?

[[Softmax]]와 로그를 따로 계산하면 수치적 문제가 발생할 수 있습니다.

### 문제: Softmax + Log의 불안정성

```python
import torch
import torch.nn.functional as F

z = torch.tensor([10.0, 20.0, 30.0])

# 방법 1: Softmax 후 로그
probs = F.softmax(z, dim=0)
log_probs_unstable = torch.log(probs)
print(log_probs_unstable)
# tensor([-20., -10.,  -0.])
```

일부 확률이 매우 작으면 언더플로우가 발생합니다:

```python
z = torch.tensor([1.0, 2.0, 100.0])

probs = F.softmax(z, dim=0)
print(probs)
# tensor([0., 0., 1.])  # 처음 두 개가 0!

log_probs = torch.log(probs)
print(log_probs)
# tensor([-inf, -inf, 0.])  # -inf 발생!
```

$e^{100}$은 너무 커서 다른 항들이 0으로 반올림되고, $\log(0) = -\infty$가 됩니다.

### 해결: LogSoftmax 직접 계산

```python
# 방법 2: LogSoftmax (안정적)
log_probs_stable = F.log_softmax(z, dim=0)
print(log_probs_stable)
# tensor([-99., -98.,  -0.])
```

유한한 값이 나옵니다! LogSoftmax는 내부적으로 log-sum-exp trick을 사용하여 안정적으로 계산합니다.

---

## 수학적 유도

Softmax에 로그를 취하면:

$$\log(\text{softmax}(z_i)) = \log\left(\frac{e^{z_i}}{\sum_j e^{z_j}}\right)$$

로그의 성질 $\log(a/b) = \log(a) - \log(b)$를 사용:

$$= \log(e^{z_i}) - \log\left(\sum_j e^{z_j}\right)$$

$$= z_i - \log\left(\sum_j e^{z_j}\right)$$

이것이 LogSoftmax의 수식입니다. 이제 softmax를 명시적으로 계산하지 않아도 됩니다.

### Log-Sum-Exp Trick

$\log(\sum_j e^{z_j})$를 직접 계산하면 오버플로우가 발생할 수 있습니다. 안정적인 계산 방법:

$$\log\left(\sum_j e^{z_j}\right) = c + \log\left(\sum_j e^{z_j - c}\right)$$

여기서 $c = \max(z)$로 설정하면:

$$\text{LogSoftmax}(z_i) = z_i - \max(z) - \log\left(\sum_j e^{z_j - \max(z)}\right)$$

이제 지수 부분의 최댓값이 $e^0 = 1$이 되어 오버플로우를 방지합니다.

---

## PyTorch 구현

### 기본 사용법

```python
import torch
import torch.nn.functional as F

# 로짓
z = torch.tensor([2.0, 1.0, 0.1])

# LogSoftmax
log_probs = F.log_softmax(z, dim=0)
print(log_probs)
# tensor([-0.4170, -1.4170, -2.3170])

# 검증: exp를 취하면 softmax
probs = torch.exp(log_probs)
print(probs)
# tensor([0.6590, 0.2424, 0.0986])

print(probs.sum())
# tensor(1.0000)
```

### 배치 처리

```python
# 배치 크기 4, 클래스 개수 3
z = torch.randn(4, 3)

# 각 샘플마다 LogSoftmax (dim=1)
log_probs = F.log_softmax(z, dim=1)
print(log_probs.shape)  # torch.Size([4, 3])

# 각 샘플의 log 확률들
print(log_probs[0])
# tensor([-1.4328, -0.3328, -1.9328])
```

### nn.LogSoftmax 레이어

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 3),
    nn.LogSoftmax(dim=1)
)

x = torch.randn(2, 10)
log_probs = model(x)
print(log_probs)
# tensor([[-1.2301, -0.9856, -1.3045],
#         [-1.4521, -0.8234, -1.5678]])
```

하지만 학습 시에는 보통 모델에 포함하지 않습니다. `nn.NLLLoss`와 함께 사용할 때만 필요합니다.

---

## NLLLoss와의 관계

LogSoftmax는 주로 **Negative Log Likelihood Loss**([[NegativeLogLikelihood]])와 함께 사용됩니다.

### Cross-Entropy Loss 분해

Cross-Entropy Loss는 다음과 같이 분해할 수 있습니다:

$$\text{CrossEntropy} = \text{LogSoftmax} + \text{NLLLoss}$$

**수식:**

$$\text{CrossEntropy}(z, y) = -\log\left(\frac{e^{z_y}}{\sum_j e^{z_j}}\right) = -\text{LogSoftmax}(z)_y$$

### PyTorch 구현 비교

**방법 1: CrossEntropyLoss (권장)**

```python
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

logits = torch.tensor([[2.0, 1.0, 0.1]])
target = torch.tensor([0])  # 정답 클래스

loss = criterion(logits, target)
print(loss)  # tensor(0.4170)
```

`nn.CrossEntropyLoss`는 내부에서 LogSoftmax와 NLLLoss를 모두 처리합니다.

**방법 2: LogSoftmax + NLLLoss**

```python
log_softmax = nn.LogSoftmax(dim=1)
nll_loss = nn.NLLLoss()

log_probs = log_softmax(logits)
loss = nll_loss(log_probs, target)
print(loss)  # tensor(0.4170) - 같은 값!
```

두 방법은 수학적으로 동일하지만, 방법 1이 더 효율적이고 안정적입니다.

---

## 미분 (Gradient)

[[Backpropagation]]을 위해 LogSoftmax의 미분을 계산해야 합니다.

### 수식 유도

$\ell_i = \text{LogSoftmax}(z_i)$, $s_i = \text{softmax}(z_i)$라고 하면:

$$\ell_i = z_i - \log\left(\sum_j e^{z_j}\right)$$

같은 원소에 대한 미분:

$$\frac{\partial \ell_i}{\partial z_i} = 1 - \frac{e^{z_i}}{\sum_j e^{z_j}} = 1 - s_i$$

다른 원소에 대한 미분:

$$\frac{\partial \ell_i}{\partial z_j} = -\frac{e^{z_j}}{\sum_j e^{z_j}} = -s_j \quad (i \neq j)$$

**행렬 형태:**

$$\frac{\partial \text{LogSoftmax}}{\partial z} = I - \mathbf{s}\mathbf{1}^T$$

여기서 $I$는 단위 행렬, $\mathbf{s}$는 softmax 벡터입니다.

### PyTorch 자동 미분

```python
z = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
log_probs = F.log_softmax(z, dim=0)

# 첫 번째 출력에 대한 gradient
log_probs[0].backward()
print(z.grad)
# tensor([ 0.9100, -0.2447, -0.6652])
```

첫 번째 클래스: $1 - s_0 = 1 - 0.09 = 0.91$ 두 번째 클래스: $-s_1 = -0.2447$ 세 번째 클래스: $-s_2 = -0.6652$

---

## Cross-Entropy와의 관계

### Cross-Entropy 정의

정보 이론에서 Cross-Entropy는 두 확률 분포 $p$(실제)와 $q$(예측) 사이의 차이를 측정합니다:

$$H(p, q) = -\sum_i p_i \log(q_i)$$

분류 문제에서 실제 분포 $p$는 one-hot 벡터입니다: $$p = [0, 0, 1, 0, 0]$$ (정답 클래스만 1)

따라서: $$H(p, q) = -\log(q_{\text{target}})$$

예측 분포 $q$는 softmax입니다: $$q = \text{softmax}(z)$$

결과적으로: $$\text{CrossEntropy} = -\log(\text{softmax}(z)_{\text{target}}) = -\text{LogSoftmax}(z)_{\text{target}}$$

### 구현 예시

```python
# 수동 계산
logits = torch.tensor([[2.0, 1.0, 0.1, -1.0]])
target = torch.tensor([0])

log_probs = F.log_softmax(logits, dim=1)
loss_manual = -log_probs[0, target]
print(loss_manual)  # tensor(1.4170)

# CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
loss_auto = criterion(logits, target)
print(loss_auto)  # tensor(1.4170) - 같은 값!
```

---

## 실전 예제

### MNIST 분류 (LogSoftmax + NLLLoss)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # LogSoftmax 적용
        return F.log_softmax(x, dim=1)

model = Net()
criterion = nn.NLLLoss()  # LogSoftmax와 함께 사용
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프 (가상 데이터)
for epoch in range(3):
    # 가상 배치
    data = torch.randn(64, 1, 28, 28)
    target = torch.randint(0, 10, (64,))
    
    optimizer.zero_grad()
    
    # Forward pass (log 확률)
    log_probs = model(data)
    
    # 손실 계산
    loss = criterion(log_probs, target)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# 추론
model.eval()
with torch.no_grad():
    test_data = torch.randn(1, 1, 28, 28)
    log_probs = model(test_data)
    
    # 확률로 변환
    probs = torch.exp(log_probs)
    predicted_class = torch.argmax(probs, dim=1)
    
    print(f'Predicted: {predicted_class.item()}')
    print(f'Log Probs: {log_probs[0]}')
    print(f'Probs: {probs[0]}')
```

### 권장 방법: CrossEntropyLoss

실전에서는 `CrossEntropyLoss`를 사용하는 것이 더 간단합니다:

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # LogSoftmax 적용 안 함!
        return x

model = Net()
criterion = nn.CrossEntropyLoss()  # 내부에서 LogSoftmax 처리
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프는 동일
```

이 방법이 더 효율적이고 수치적으로 안정적입니다.

---

## 언제 LogSoftmax를 사용하는가?

### 사용하는 경우

**1. NLLLoss와 함께 사용**

이미 LogSoftmax가 모델에 포함되어 있을 때:

```python
model_output = F.log_softmax(logits, dim=1)
loss = F.nll_loss(model_output, target)
```

**2. 커스텀 손실 함수**

직접 log 확률을 다루는 손실 함수를 만들 때:

```python
log_probs = F.log_softmax(logits, dim=1)
custom_loss = -log_probs.gather(1, target.unsqueeze(1)).mean()
```

**3. KL Divergence**

두 분포 간의 KL divergence를 계산할 때:

```python
log_probs1 = F.log_softmax(logits1, dim=1)
log_probs2 = F.log_softmax(logits2, dim=1)

kl_div = F.kl_div(log_probs1, log_probs2.exp(), reduction='batchmean')
```

### 사용하지 않는 경우

**대부분의 경우: CrossEntropyLoss 사용**

```python
logits = model(x)  # LogSoftmax 없이 로짓만
loss = F.cross_entropy(logits, target)
```

이것이 가장 간단하고 효율적입니다.

---

## 수치 안정성 비교

세 가지 방법을 비교해봅시다:

### 방법 1: Softmax → Log (불안정)

```python
z = torch.tensor([1.0, 2.0, 100.0])

probs = F.softmax(z, dim=0)
log_probs_unstable = torch.log(probs)
print(log_probs_unstable)
# tensor([  -inf,   -inf, -0.0000])
```

언더플로우로 `-inf`가 발생합니다.

### 방법 2: LogSoftmax 수동 구현 (불안정)

```python
def naive_log_softmax(z):
    return z - torch.log(torch.sum(torch.exp(z)))

log_probs_naive = naive_log_softmax(z)
print(log_probs_naive)
# tensor([  -inf,   -inf,   nan])  # NaN까지 발생!
```

$e^{100}$이 오버플로우하여 `inf`가 되고, $\log(\infty) = \infty$, $1 - \infty = -\infty$가 됩니다.

### 방법 3: PyTorch LogSoftmax (안정적)

```python
log_probs_stable = F.log_softmax(z, dim=0)
print(log_probs_stable)
# tensor([-99., -98.,  -0.])
```

유한한 값이 나옵니다! 내부적으로 log-sum-exp trick을 사용합니다.

---

## 성능 비교

LogSoftmax + NLLLoss vs CrossEntropyLoss:

```python
import time

logits = torch.randn(1000, 1000, requires_grad=True)
target = torch.randint(0, 1000, (1000,))

# 방법 1: LogSoftmax + NLLLoss
start = time.time()
for _ in range(100):
    log_probs = F.log_softmax(logits, dim=1)
    loss = F.nll_loss(log_probs, target)
    loss.backward()
method1_time = time.time() - start

# 방법 2: CrossEntropyLoss
logits = torch.randn(1000, 1000, requires_grad=True)
start = time.time()
for _ in range(100):
    loss = F.cross_entropy(logits, target)
    loss.backward()
method2_time = time.time() - start

print(f'LogSoftmax + NLLLoss: {method1_time:.4f}s')
print(f'CrossEntropyLoss: {method2_time:.4f}s')
# CrossEntropyLoss가 약 10-20% 더 빠름
```

`CrossEntropyLoss`가 더 효율적입니다. 내부 최적화가 잘 되어 있기 때문입니다.

---

## 핵심 요약

- LogSoftmax는 Softmax의 로그를 수치적으로 안정적으로 계산
- $\text{LogSoftmax}(z_i) = z_i - \log(\sum_j e^{z_j})$
- Softmax → Log보다 안정적 (언더플로우/오버플로우 방지)
- 주로 NLLLoss와 함께 사용
- CrossEntropyLoss = LogSoftmax + NLLLoss
- 실전에서는 CrossEntropyLoss 사용 권장
- PyTorch의 `F.log_softmax()` 사용 (직접 구현 금지)

---

## 관련 개념

**선수 지식:**

- [[Softmax]] - Softmax 함수
- [[Loss Function]] - 손실 함수의 역할

**관련 개념:**

- [[NegativeLogLikelihood]] - NLL Loss
- [[Forward Pass]] - 순전파
- [[Backpropagation]] - 역전파

**다음 단계:**

- [[NegativeLogLikelihood]] - LogSoftmax와 함께 사용하는 손실 함수
- Cross-Entropy Loss - 분류 문제의 표준 손실