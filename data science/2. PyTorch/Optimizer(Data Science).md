# Optimizer

## 정의

**최적화 알고리즘(Optimizer)** 은 [[Loss Function]]을 최소화하기 위해 신경망의 파라미터를 업데이트하는 알고리즘입니다. [[Gradient]]를 사용하여 파라미터를 어떻게, 얼마나 조정할지 결정합니다.

기본 형태: $$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L$$

여기서:

- $\theta$: 파라미터 (가중치와 편향)
- $\eta$: 학습률 (learning rate)
- $\nabla_\theta L$: 손실에 대한 파라미터의 gradient
- $t$: 현재 iteration

---
## 왜 필요한가?

[[Machine Learning Basics]]에서 배웠듯이, 학습의 목표는 손실을 최소화하는 파라미터를 찾는 것입니다. 하지만 단순히 gradient의 반대 방향으로 이동하는 것만으로는 충분하지 않습니다.

**단순한 경사 하강법의 문제:**

산을 내려갈 때를 생각해봅시다. 가장 가파른 방향으로만 내려간다면:

- 지그재그로 비효율적으로 이동할 수 있음
- 작은 골짜기에 갇혀 더 깊은 골짜기를 놓칠 수 있음
- 평평한 곳에서 매우 느리게 이동
- 가파른 곳에서 너무 빨리 이동하여 튀어오를 수 있음

Optimizer는 이런 문제들을 해결하여 더 빠르고 안정적으로 최적의 파라미터를 찾습니다.

---

## Gradient Descent (경사 하강법)

가장 기본적인 최적화 알고리즘입니다.

### Batch Gradient Descent

전체 데이터셋의 gradient를 계산하여 업데이트합니다:

$$\theta = \theta - \eta \cdot \frac{1}{N}\sum_{i=1}^{N}\nabla_\theta L_i$$

**장점:**

- 안정적인 수렴
- 이론적으로 명확함

**단점:**

- 데이터셋이 크면 매우 느림
- 메모리를 많이 사용
- 지역 최솟값에 갇히기 쉬움

**예시:**

```python
for epoch in range(num_epochs):
    # 전체 데이터로 gradient 계산
    output = model(X_train)
    loss = criterion(output, y_train)
    
    loss.backward()
    
    # 파라미터 업데이트
    for param in model.parameters():
        param.data -= learning_rate * param.grad
    
    model.zero_grad()
```

### Stochastic Gradient Descent (SGD)

각 샘플마다 gradient를 계산하여 업데이트합니다:

$$\theta = \theta - \eta \cdot \nabla_\theta L_i$$

**장점:**

- 빠른 업데이트
- 메모리 효율적
- 지역 최솟값에서 탈출 가능 (노이즈 덕분)

**단점:**

- 불안정한 수렴 (진동이 심함)
- 최적값 근처에서 계속 진동

실전에서는 순수한 SGD보다는 **Mini-batch SGD**를 사용합니다.

### Mini-batch Gradient Descent

일부 샘플(mini-batch)의 gradient를 평균하여 업데이트합니다:

$$\theta = \theta - \eta \cdot \frac{1}{m}\sum_{i=1}^{m}\nabla_\theta L_i$$

여기서 $m$은 배치 크기 (예: 32, 64, 128)입니다.

**장점:**

- Batch GD보다 빠름
- SGD보다 안정적
- GPU 병렬 처리에 효율적

**단점:**

- 배치 크기를 조정해야 함

**PyTorch 구현:**

```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        # Forward pass
        output = model(x_batch)
        loss = criterion(output, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # 파라미터 업데이트
        optimizer.step()
```

---

## SGD with Momentum

단순 SGD의 문제는 gradient가 작은 방향으로는 느리게 이동하고, 큰 방향으로는 진동한다는 것입니다.

### 개념

**Momentum**은 이전 gradient의 정보를 유지하여 일관된 방향으로 가속합니다. 공이 언덕을 굴러 내려가는 것처럼, 이전 속도를 고려하여 움직입니다.

$$v_t = \beta v_{t-1} + \nabla_\theta L$$ $$\theta = \theta - \eta \cdot v_t$$

여기서:

- $v_t$: 속도 (velocity)
- $\beta$: momentum 계수 (보통 0.9)

### 작동 원리

평평한 지역을 생각해봅시다:

- 단순 SGD: gradient가 작아서 매우 느리게 이동
- Momentum: 이전의 속도를 유지하여 계속 빠르게 이동

지그재그로 진동하는 경우:

- 단순 SGD: 계속 진동
- Momentum: 진동하는 방향은 상쇄되고, 일관된 방향만 가속

**PyTorch 구현:**

```python
optimizer = optim.SGD(
    model.parameters(), 
    lr=0.01, 
    momentum=0.9
)
```

**수치 예시:**

학습률 0.1, momentum 0.9, 초기 속도 0일 때:

Iteration 1:

- Gradient: 2.0
- 속도: $v_1 = 0.9 \times 0 + 2.0 = 2.0$
- 업데이트: $\theta = \theta - 0.1 \times 2.0 = \theta - 0.2$

Iteration 2:

- Gradient: 1.8
- 속도: $v_2 = 0.9 \times 2.0 + 1.8 = 3.6$
- 업데이트: $\theta = \theta - 0.1 \times 3.6 = \theta - 0.36$

Gradient가 줄어들어도 속도 덕분에 더 큰 step을 취합니다.

---

## RMSprop

Gradient의 크기가 차원마다 크게 다를 때 문제가 발생합니다. 어떤 파라미터는 gradient가 크고, 어떤 파라미터는 작습니다.

### 개념

**RMSprop**는 각 파라미터마다 다른 학습률을 적용합니다. Gradient가 큰 방향은 학습률을 줄이고, 작은 방향은 학습률을 유지합니다.

$$s_t = \beta s_{t-1} + (1-\beta)(\nabla_\theta L)^2$$ $$\theta = \theta - \frac{\eta}{\sqrt{s_t + \epsilon}} \cdot \nabla_\theta L$$

여기서:

- $s_t$: gradient 제곱의 지수 이동 평균
- $\beta$: 감쇠율 (보통 0.9 또는 0.99)
- $\epsilon$: 0으로 나누는 것을 방지 (보통 $10^{-8}$)

### 작동 원리

어떤 파라미터의 gradient가 계속 크다면:

- $s_t$가 커짐
- $\frac{1}{\sqrt{s_t}}$가 작아짐
- 해당 파라미터의 유효 학습률이 감소

어떤 파라미터의 gradient가 작다면:

- $s_t$가 작음
- 유효 학습률이 상대적으로 큼

결과적으로 모든 파라미터가 비슷한 크기로 업데이트됩니다.

**PyTorch 구현:**

```python
optimizer = optim.RMSprop(
    model.parameters(),
    lr=0.01,
    alpha=0.9  # beta
)
```

**수치 예시:**

학습률 0.1, beta 0.9, 초기 $s_0 = 0$일 때:

Iteration 1:

- Gradient: 10.0
- $s_1 = 0.9 \times 0 + 0.1 \times 10^2 = 10$
- 업데이트: $\frac{0.1}{\sqrt{10}} \times 10 = 0.316$

Iteration 2:

- Gradient: 12.0
- $s_2 = 0.9 \times 10 + 0.1 \times 12^2 = 23.4$
- 업데이트: $\frac{0.1}{\sqrt{23.4}} \times 12 = 0.248$

Gradient가 증가했지만 업데이트 크기는 오히려 감소했습니다.

---

## Adam (Adaptive Moment Estimation)

**Adam**은 현재 가장 널리 사용되는 optimizer입니다. Momentum과 RMSprop의 장점을 결합했습니다.

### 개념

Momentum처럼 gradient의 이동 평균(1차 moment)을 사용하고, RMSprop처럼 gradient 제곱의 이동 평균(2차 moment)을 사용합니다:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla_\theta L$$ $$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla_\theta L)^2$$

초기 bias를 보정: $$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$ $$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

파라미터 업데이트: $$\theta = \theta - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

여기서:

- $m_t$: gradient의 1차 moment (평균)
- $v_t$: gradient의 2차 moment (분산)
- $\beta_1$: 보통 0.9
- $\beta_2$: 보통 0.999
- $\epsilon$: 보통 $10^{-8}$

### 작동 원리

**Momentum 효과:** $m_t$가 일관된 방향으로 누적되어 가속합니다.

**RMSprop 효과:** $v_t$가 gradient 크기를 정규화하여 안정적으로 만듭니다.

**Bias 보정:** 초기에는 $m_0 = 0$, $v_0 = 0$이므로 값이 0쪽으로 치우칩니다. $(1-\beta^t)$로 나누어 이를 보정합니다.

**PyTorch 구현:**

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999)  # (beta1, beta2)
)
```

### 왜 Adam이 인기있는가?

- **적응적 학습률**: 각 파라미터마다 자동으로 학습률 조정
- **빠른 수렴**: Momentum 덕분에 빠르게 수렴
- **안정성**: RMSprop 효과로 안정적
- **하이퍼파라미터 튜닝 적음**: 기본값이 대부분 경우에 잘 작동

**주의사항:**

일부 경우(특히 작은 데이터셋)에서는 SGD with Momentum이 더 나은 일반화 성능을 보일 수 있습니다.

---

## AdamW

Adam의 변형으로, weight decay를 올바르게 구현한 버전입니다.

### Weight Decay

과적합을 방지하기 위해 파라미터의 크기에 페널티를 추가합니다:

$$L_{total} = L + \lambda |\theta|^2$$

표준 Adam에서는 이것이 제대로 작동하지 않습니다. AdamW는 weight decay를 gradient가 아닌 파라미터에 직접 적용합니다:

$$\theta = \theta - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta\right)$$

**PyTorch 구현:**

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)
```

Transformer 같은 대규모 모델에서 특히 효과적입니다.

---

## 학습률 (Learning Rate)

모든 optimizer에서 가장 중요한 하이퍼파라미터는 학습률입니다.

### 학습률의 영향

**너무 큰 학습률:**

```
Loss
  |     *
  |   *   *
  | *       *
  |*         *
  +-----------> Iterations
```

진동하거나 발산합니다.

**너무 작은 학습률:**

```
Loss
  |\
  | \___________
  |
  +-----------> Iterations
```

수렴이 매우 느립니다.

**적절한 학습률:**

```
Loss
  |\
  | \
  |  \____
  |
  +-----------> Iterations
```

빠르고 안정적으로 수렴합니다.

### 학습률 범위

일반적인 범위:

- SGD: 0.01 ~ 0.1
- Adam: 0.0001 ~ 0.001

작은 값부터 시작하여 점진적으로 증가시키며 최적값을 찾습니다.

---

## 학습률 스케줄링

학습 중에 학습률을 조정하면 더 나은 결과를 얻을 수 있습니다.

### Step Decay

일정 에포크마다 학습률을 감소시킵니다:

```python
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,  # 30 에포크마다
    gamma=0.1      # 0.1배로 감소
)

for epoch in range(num_epochs):
    train(...)
    scheduler.step()  # 학습률 업데이트
```

### Exponential Decay

지수적으로 감소:

```python
scheduler = optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.95  # 매 에포크마다 0.95배
)
```

### Cosine Annealing

코사인 함수처럼 부드럽게 감소:

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100  # 100 에포크 동안 감소
)
```

### ReduceLROnPlateau

성능 향상이 멈추면 학습률 감소:

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=10  # 10 에포크 동안 개선 없으면 감소
)

for epoch in range(num_epochs):
    val_loss = validate(...)
    scheduler.step(val_loss)  # 검증 손실 기반 조정
```

### Warm-up

초기에는 학습률을 작게 시작하여 점진적으로 증가시킵니다. 대규모 모델에서 안정성을 높입니다:

```python
def warmup_lr(epoch, warmup_epochs=5, base_lr=0.001):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr

scheduler = optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=warmup_lr
)
```

---

## Optimizer 선택 가이드

### 기본 권장사항

**대부분의 경우:**

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

Adam이 잘 작동하고 하이퍼파라미터 튜닝이 적게 필요합니다.

**컴퓨터 비전 (CNN):**

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)
```

SGD with Momentum이 더 나은 일반화 성능을 보일 수 있습니다.

**자연어 처리 (Transformer):**

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.01
)
```

AdamW가 일반적으로 선택됩니다.

### 비교

|Optimizer|속도|안정성|일반화|튜닝 난이도|
|---|---|---|---|---|
|SGD|느림|보통|좋음|어려움|
|SGD+Momentum|보통|좋음|좋음|보통|
|RMSprop|빠름|좋음|보통|쉬움|
|Adam|빠름|좋음|보통|쉬움|
|AdamW|빠름|좋음|좋음|쉬움|

---

## 실전 예제

### 전체 학습 루프

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 모델, 데이터, 손실 함수
model = MyModel()
train_loader = DataLoader(train_dataset, batch_size=32)
criterion = nn.CrossEntropyLoss()

# Optimizer와 스케줄러
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 학습 루프
for epoch in range(num_epochs):
    model.train()
    
    for x_batch, y_batch in train_loader:
        # Forward pass
        output = model(x_batch)
        loss = criterion(output, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # 파라미터 업데이트
        optimizer.step()
    
    # 학습률 조정
    scheduler.step()
    
    print(f'Epoch {epoch}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
```

### Gradient Clipping

Gradient가 너무 커지는 것을 방지:

```python
optimizer.zero_grad()
loss.backward()

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

optimizer.step()
```

RNN이나 Transformer에서 gradient 폭발을 방지하는 데 유용합니다.

---

## 고급 주제

### 혼합 정밀도 훈련

FP16과 FP32를 섞어 사용하여 메모리와 속도를 개선:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for x_batch, y_batch in train_loader:
    optimizer.zero_grad()
    
    # 자동 혼합 정밀도
    with autocast():
        output = model(x_batch)
        loss = criterion(output, y_batch)
    
    # Gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 분산 학습

여러 GPU에서 학습:

```python
model = nn.DataParallel(model)  # 단순한 방법

# 또는 DistributedDataParallel (더 효율적)
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model, device_ids=[local_rank])
```

Optimizer는 똑같이 사용하지만 gradient가 여러 GPU에서 평균됩니다.

---

## 핵심 요약

- Optimizer는 [[Gradient]]를 사용하여 파라미터를 업데이트
- SGD는 기본이지만 Momentum, Adam 등이 더 효율적
- Adam은 적응적 학습률로 대부분 경우에 잘 작동
- 학습률은 가장 중요한 하이퍼파라미터
- 학습률 스케줄링으로 성능 향상 가능
- Gradient clipping으로 안정성 확보

---

## 관련 개념

**선수 지식:**

- [[Machine Learning Basics]] - 머신러닝 기본 원리
- [[Gradient]] - Gradient의 의미와 경사하강법
- [[Loss Function]] - 손실 함수

**관련 개념:**

- [[Backpropagation]] - Gradient 계산
- [[Deep Learning Core Concepts]] - 딥러닝 전체 구조

**다음 단계:**

- [[Neural Network Components]] - 신경망 구성 요소
- 실전 PyTorch 모델 학습