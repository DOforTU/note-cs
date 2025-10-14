# Negative Log Likelihood

## 정의

**Negative Log Likelihood (NLL)** 는 음의 로그 우도(likelihood)를 계산하는 [[Loss Function]]입니다. 모델이 예측한 확률 분포가 실제 데이터를 얼마나 잘 설명하는지 측정합니다.

$$\text{NLL} = -\log P(y|\theta)$$

여기서:

- $y$: 실제 정답 레이블
- $\theta$: 모델의 파라미터
- $P(y|\theta)$: 모델이 예측한 정답 클래스의 확률

---

## 우도(Likelihood)란?

우도를 이해하려면 먼저 확률과 우도의 차이를 알아야 합니다.

### 확률 vs 우도

**확률 (Probability):** 파라미터가 고정되어 있을 때, 특정 데이터가 나올 가능성

동전을 던지는 예시입니다. 동전이 앞면이 나올 확률이 $p = 0.7$로 **고정**되어 있을 때:

- "앞면이 나올 확률은?" → $P(\text{앞면}|p=0.7) = 0.7$
- "10번 던져서 7번 앞면이 나올 확률은?" → 이항분포로 계산

**우도 (Likelihood):** 데이터가 고정되어 있을 때, 특정 파라미터일 가능성

동전을 10번 던졌더니 7번 앞면이 나왔다는 **데이터가 고정**되어 있을 때:

- "이 동전의 앞면 확률이 0.7일 가능성은?" → $L(p=0.7|\text{7번 앞면})$
- "이 동전의 앞면 확률이 0.5일 가능성은?" → $L(p=0.5|\text{7번 앞면})$

**핵심 차이:**

- 확률: 파라미터 고정, 데이터 변동
- 우도: 데이터 고정, 파라미터 변동

### 수학적 표현

동일한 수식이지만 관점이 다릅니다:

**확률로 볼 때:** $$P(\text{data}|\theta) \quad \text{(theta가 주어졌을 때 data가 나올 확률)}$$

**우도로 볼 때:** $$L(\theta|\text{data}) \quad \text{(data가 주어졌을 때 theta의 가능도)}$$

수학적으로는 같은 값이지만: $$L(\theta|\text{data}) = P(\text{data}|\theta)$$

### 머신러닝에서의 우도

머신러닝에서 우리는 데이터를 이미 가지고 있고, 최선의 파라미터를 찾고 싶습니다.

**예시: 이미지 분류**

고양이 이미지가 주어졌을 때:

- 모델 A는 "고양이일 확률 90%"라고 예측
- 모델 B는 "고양이일 확률 60%"라고 예측

어느 모델이 더 나을까요? 모델 A입니다. 정답을 더 높은 확률로 예측했기 때문입니다.

**우도로 표현:**

- 모델 A의 우도: $L(\theta_A|\text{고양이}) = 0.9$
- 모델 B의 우도: $L(\theta_B|\text{고양이}) = 0.6$

우도가 높을수록 좋은 모델입니다.

---

## 왜 로그를 취하는가?

우도를 직접 최대화하는 대신, **로그 우도**를 최대화합니다.

### 이유 1: 곱셈을 덧셈으로 변환

여러 데이터 포인트가 있을 때, 전체 우도는 각각의 곱입니다:

$$L(\theta|D) = P(y_1|\theta) \times P(y_2|\theta) \times \cdots \times P(y_N|\theta)$$

예를 들어: $$L = 0.9 \times 0.8 \times 0.7 \times \cdots = \text{매우 작은 수}$$

100개를 곱하면 $0.9^{100} \approx 2.66 \times 10^{-5}$처럼 아주 작아집니다 (언더플로우).

**로그를 취하면:** $$\log L(\theta|D) = \log P(y_1|\theta) + \log P(y_2|\theta) + \cdots + \log P(y_N|\theta)$$

곱셈이 덧셈이 되어 계산이 안정적입니다.

### 이유 2: 수치 안정성

확률은 0과 1 사이의 작은 수입니다:

- $P = 0.001$을 그대로 계산하면 언더플로우 위험
- $\log P = \log(0.001) = -6.9$로 변환하면 안정적

### 이유 3: 수학적 편의성

로그 함수의 성질:

- $\log(ab) = \log(a) + \log(b)$
- 미분이 간단: $\frac{d}{dx}\log(x) = \frac{1}{x}$

### 이유 4: 단조 증가 함수

로그는 단조 증가 함수이므로, 최대화/최소화 문제가 동일합니다:

- $L$을 최대화 = $\log L$을 최대화
- 최댓값의 위치가 같음

---

## 왜 Negative(음수)를 붙이는가?

머신러닝에서는 [[Loss Function]]을 **최소화**하는 것이 관례입니다. 하지만 우도는 **최대화**해야 하므로, 음수를 붙여 최소화 문제로 바꿉니다.

**로그 우도 최대화:** $$\max_\theta \log L(\theta|D)$$

**음의 로그 우도 최소화:** $$\min_\theta -\log L(\theta|D)$$

두 문제는 동일합니다. 음수를 붙이면 [[Gradient]] descent 같은 표준 최적화 알고리즘을 그대로 사용할 수 있습니다.

**예시:**

모델이 정답을 0.9의 확률로 예측:

- 우도: $L = 0.9$ (높을수록 좋음)
- 로그 우도: $\log L = \log(0.9) = -0.105$ (높을수록 좋음)
- NLL: $-\log L = 0.105$ (낮을수록 좋음) ✓

모델이 정답을 0.1의 확률로 예측:

- 우도: $L = 0.1$ (낮음)
- 로그 우도: $\log L = \log(0.1) = -2.303$ (낮음)
- NLL: $-\log L = 2.303$ (높음, 손실이 큼) ✓

---

## 분류 문제에서의 NLL

### 단일 샘플

모델이 클래스 $k$의 확률을 $p_k$로 예측했고, 실제 정답이 클래스 $c$일 때:

$$\text{NLL} = -\log(p_c)$$

**예시:**

3개 클래스 (고양이, 개, 새)에서:

- 예측 확률: $[0.7, 0.2, 0.1]$
- 실제 정답: 고양이 (인덱스 0)

$$\text{NLL} = -\log(0.7) = 0.357$$

만약 잘못 예측했다면:

- 예측 확률: $[0.1, 0.7, 0.2]$
- 실제 정답: 고양이 (인덱스 0)

$$\text{NLL} = -\log(0.1) = 2.303$$

손실이 훨씬 큽니다!

### 배치

$N$개의 샘플이 있을 때:

$$\text{NLL} = -\frac{1}{N}\sum_{i=1}^{N}\log(p_{y_i}^{(i)})$$

여기서 $p_{y_i}^{(i)}$는 $i$번째 샘플의 정답 클래스에 대한 예측 확률입니다.

---

## PyTorch 구현

### nn.NLLLoss

PyTorch의 `nn.NLLLoss`는 이미 log 확률을 입력으로 받습니다:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Log 확률 (LogSoftmax의 출력)
log_probs = torch.tensor([[-0.357, -1.609, -2.303],  # 샘플 1
                           [-2.303, -0.357, -1.609]])  # 샘플 2

# 정답 레이블
target = torch.tensor([0, 1])

# NLL Loss 계산
criterion = nn.NLLLoss()
loss = criterion(log_probs, target)
print(loss)  # tensor(0.3570)
```

**계산 과정:**

- 샘플 1: $-(-0.357) = 0.357$ (정답 클래스 0)
- 샘플 2: $-(-0.357) = 0.357$ (정답 클래스 1)
- 평균: $(0.357 + 0.357) / 2 = 0.357$

### 수동 계산

```python
# 수동으로 NLL 계산
loss_manual = 0
for i, target_class in enumerate(target):
    loss_manual += -log_probs[i, target_class]
loss_manual = loss_manual / len(target)
print(loss_manual)  # tensor(0.3570) - 같은 값!
```

---

## LogSoftmax와의 관계

NLLLoss는 항상 [[LogSoftmax]]와 함께 사용됩니다.

### 전체 파이프라인

```python
# 1. 신경망이 로짓 출력
logits = torch.tensor([[2.0, 1.0, 0.1],
                        [0.1, 2.0, 1.0]])

# 2. LogSoftmax 적용
log_probs = F.log_softmax(logits, dim=1)
print(log_probs)
# tensor([[-0.4170, -1.4170, -2.3170],
#         [-2.3170, -0.4170, -1.4170]])

# 3. NLLLoss 계산
target = torch.tensor([0, 1])
criterion = nn.NLLLoss()
loss = criterion(log_probs, target)
print(loss)  # tensor(0.4170)
```

### CrossEntropyLoss = LogSoftmax + NLLLoss

`nn.CrossEntropyLoss`는 이 두 단계를 한 번에 처리합니다:

```python
# 방법 1: LogSoftmax + NLLLoss
log_probs = F.log_softmax(logits, dim=1)
loss1 = F.nll_loss(log_probs, target)

# 방법 2: CrossEntropyLoss (권장)
loss2 = F.cross_entropy(logits, target)

print(loss1)  # tensor(0.4170)
print(loss2)  # tensor(0.4170) - 같은 값!
```

방법 2가 더 효율적이고 수치적으로 안정적입니다.

---

## 실전 예제

### 간단한 분류 모델

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 모델 정의
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 3)  # 3개 클래스
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # LogSoftmax 적용
        return F.log_softmax(x, dim=1)

model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 가상 데이터
X = torch.randn(100, 10)
y = torch.randint(0, 3, (100,))

# 학습
for epoch in range(50):
    optimizer.zero_grad()
    
    # Forward pass (log 확률)
    log_probs = model(X)
    
    # NLL Loss
    loss = criterion(log_probs, y)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# 예측
model.eval()
with torch.no_grad():
    test_X = torch.randn(5, 10)
    log_probs = model(test_X)
    
    # 확률로 변환
    probs = torch.exp(log_probs)
    predictions = torch.argmax(probs, dim=1)
    
    print('Predictions:', predictions)
    print('Probabilities:', probs)
```

### CrossEntropyLoss 사용 (권장)

```python
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # LogSoftmax 적용 안 함!
        return x

model = Classifier()
criterion = nn.CrossEntropyLoss()  # LogSoftmax + NLLLoss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 학습 루프는 동일
for epoch in range(50):
    optimizer.zero_grad()
    
    # Forward pass (로짓만)
    logits = model(X)
    
    # CrossEntropyLoss
    loss = criterion(logits, y)
    
    loss.backward()
    optimizer.step()
```

이 방법이 더 간단하고 효율적입니다.

---

## 가중치가 있는 NLL

클래스 불균형이 있을 때 클래스별 가중치를 설정할 수 있습니다.

### 예시: 의료 진단

정상(90%), 질병A(8%), 질병B(2%)로 데이터가 불균형할 때:

```python
# 클래스별 가중치 (희귀 클래스에 높은 가중치)
class_weights = torch.tensor([1.0, 5.0, 20.0])

criterion = nn.NLLLoss(weight=class_weights)

log_probs = torch.tensor([[-0.1, -2.3, -4.6],
                           [-3.0, -0.5, -2.0]])
target = torch.tensor([0, 2])  # 정상, 질병B

loss = criterion(log_probs, target)
print(loss)
```

**계산:**

- 샘플 1 (정상): $-(-0.1) \times 1.0 = 0.1$
- 샘플 2 (질병B): $-(-2.0) \times 20.0 = 40.0$
- 가중 평균: $(0.1 + 40.0) / 2 = 20.05$

희귀 클래스를 잘못 예측하면 손실이 크게 증가합니다.

---

## 수학적 유도

### Cross-Entropy와의 관계

Cross-Entropy는 정보 이론에서 두 확률 분포 간의 거리를 측정합니다:

$$H(p, q) = -\sum_i p_i \log(q_i)$$

분류 문제에서:

- $p$: 실제 분포 (one-hot 벡터)
- $q$: 예측 분포 (softmax 출력)

정답 클래스만 $p = 1$이고 나머지는 $p = 0$이므로:

$$H(p, q) = -1 \times \log(q_{\text{target}}) + 0 \times \log(\cdots) = -\log(q_{\text{target}})$$

이것이 바로 NLL입니다!

### Maximum Likelihood Estimation (MLE)

머신러닝의 목표는 데이터의 우도를 최대화하는 파라미터를 찾는 것입니다:

$$\theta^* = \arg\max_\theta L(\theta|D) = \arg\max_\theta \prod_{i=1}^{N} P(y_i|x_i, \theta)$$

로그를 취하면:

$$\theta^* = \arg\max_\theta \sum_{i=1}^{N} \log P(y_i|x_i, \theta)$$

음수를 붙여 최소화 문제로:

$$\theta^* = \arg\min_\theta -\sum_{i=1}^{N} \log P(y_i|x_i, \theta)$$

이것이 NLL Loss입니다. 따라서 **NLL을 최소화하는 것은 MLE와 동일**합니다.

---

## Reduction 옵션

손실을 어떻게 집계할지 제어할 수 있습니다:

```python
# mean (기본값): 평균
criterion = nn.NLLLoss(reduction='mean')
loss = criterion(log_probs, target)
print(loss)  # tensor(0.4170)

# sum: 합계
criterion = nn.NLLLoss(reduction='sum')
loss = criterion(log_probs, target)
print(loss)  # tensor(0.8340)

# none: 각 샘플의 손실
criterion = nn.NLLLoss(reduction='none')
loss = criterion(log_probs, target)
print(loss)  # tensor([0.4170, 0.4170])
```

`reduction='none'`은 샘플별 손실이 필요할 때 유용합니다:

```python
criterion = nn.NLLLoss(reduction='none')
sample_losses = criterion(log_probs, target)

# 중요한 샘플에 더 큰 가중치
sample_weights = torch.tensor([1.0, 2.0])
weighted_loss = (sample_losses * sample_weights).mean()
```

---

## NLL vs Cross-Entropy vs MSE

### 분류 문제: NLL (또는 Cross-Entropy)

```python
# 올바름
criterion = nn.NLLLoss()
# 또는
criterion = nn.CrossEntropyLoss()
```

**이유:**

- 확률 분포를 다룸
- 정답 클래스에 높은 확률을 할당하도록 학습

### 회귀 문제: MSE

```python
# 올바름
criterion = nn.MSELoss()
```

**이유:**

- 연속적인 값 예측
- 예측값과 실제값의 거리 최소화

### 잘못된 사용

```python
# 분류 문제에 MSE 사용 (잘못됨!)
logits = model(x)
probs = F.softmax(logits, dim=1)
loss = F.mse_loss(probs, target_one_hot)
```

이론적으로 작동하지만 학습이 불안정하고 성능이 나쁩니다.

---

## 핵심 요약

- NLL은 음의 로그 우도를 계산하는 손실 함수
- 우도: 주어진 데이터에 대해 파라미터가 얼마나 그럴듯한지
- 로그: 곱셈을 덧셈으로, 수치 안정성 향상
- 음수: 최대화 문제를 최소화 문제로 변환
- LogSoftmax와 함께 사용
- CrossEntropyLoss = LogSoftmax + NLLLoss
- 분류 문제의 표준 손실 함수
- MLE와 수학적으로 동일

---

## 관련 개념

**선수 지식:**

- [[Machine Learning Basics]] - 머신러닝 기본 원리
- [[Loss Function]] - 손실 함수의 역할
- [[Softmax]] - 확률 분포로 변환
- [[LogSoftmax]] - 로그 확률 계산

**관련 개념:**

- [[Forward Pass]] - 순전파
- [[Backpropagation]] - 역전파
- [[Gradient]] - Gradient descent

**다음 단계:**

- [[Neural Network Components]] - 신경망 구성 요소
- 실전 분류 모델 구현