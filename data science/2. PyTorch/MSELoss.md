# MSELoss

## 정의

**MSELoss(Mean Squared Error Loss)**는 평균 제곱 오차를 계산하는 [[Loss Function]]입니다. 예측값과 실제값의 차이를 제곱하여 평균을 낸 값으로, 주로 회귀 문제에서 사용됩니다.

$$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

여기서:

- $N$: 샘플의 개수
- $y_i$: 실제값 (ground truth)
- $\hat{y}_i$: 모델의 예측값
- $(y_i - \hat{y}_i)$: 오차 (error)

---

## 왜 제곱을 하는가?

단순히 차이의 평균을 구하면 안 될까요?

$$\text{Mean Error} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)$$

이 방식에는 문제가 있습니다:

**문제 1: 양수와 음수 상쇄**

예측이 실제보다 2만큼 크고, 다른 예측이 2만큼 작다면: $$\frac{1}{2}[(3-5) + (7-5)] = \frac{1}{2}[-2 + 2] = 0$$

평균 오차가 0이 나와서 모델이 완벽한 것처럼 보이지만, 실제로는 둘 다 틀렸습니다.

**문제 2: 미분 불가능한 지점**

절댓값을 사용하는 MAE (Mean Absolute Error)도 있지만: $$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|$$

절댓값 함수는 0에서 미분이 불가능합니다. [[Backpropagation]]에서 문제가 될 수 있습니다.

**제곱의 장점:**

1. **항상 양수**: 음수 오차도 양수로 변환
2. **미분 가능**: 모든 지점에서 부드럽게 미분 가능
3. **큰 오차에 페널티**: 오차가 클수록 제곱으로 더 큰 손실

---

## 수학적 성질

### 볼록 함수 (Convex Function)

MSE는 파라미터에 대해 볼록 함수입니다. 이는 지역 최솟값이 전역 최솟값이라는 의미입니다.

간단한 선형 회귀 $\hat{y} = wx + b$에서: $$L = \frac{1}{N}\sum_{i=1}^{N}(y_i - wx_i - b)^2$$

이 함수를 그래프로 그리면 그릇 모양(parabola)이 됩니다. 어디서 시작하든 [[Gradient]] descent로 최솟값에 도달할 수 있습니다.

### 미분

MSE의 미분은 간단합니다:

$$\frac{\partial L}{\partial \hat{y}_i} = \frac{2}{N}(\hat{y}_i - y_i)$$

[[Chain rule]]을 통해 모델 파라미터의 gradient를 계산할 수 있습니다.

선형 모델 $\hat{y} = wx + b$에서:

$$\frac{\partial L}{\partial w} = \frac{2}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i) \cdot x_i$$

$$\frac{\partial L}{\partial b} = \frac{2}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)$$

---

## PyTorch 구현

### 기본 사용법

```python
import torch
import torch.nn as nn

# MSELoss 생성
criterion = nn.MSELoss()

# 예측값과 실제값
y_pred = torch.tensor([2.5, 3.0, 4.5])
y_true = torch.tensor([3.0, 3.0, 5.0])

# 손실 계산
loss = criterion(y_pred, y_true)
print(loss)  # tensor(0.1667)
```

**계산 과정:** $$(2.5-3.0)^2 = 0.25$$ $$(3.0-3.0)^2 = 0.0$$ $$(4.5-5.0)^2 = 0.25$$ $$\text{MSE} = \frac{0.25 + 0.0 + 0.25}{3} = 0.1667$$

### Reduction 옵션

`reduction` 파라미터로 계산 방식을 제어할 수 있습니다:

```python
# mean (기본값): 평균
criterion = nn.MSELoss(reduction='mean')
loss = criterion(y_pred, y_true)
print(loss)  # tensor(0.1667)

# sum: 합계
criterion = nn.MSELoss(reduction='sum')
loss = criterion(y_pred, y_true)
print(loss)  # tensor(0.5000)

# none: 각 샘플의 손실 반환
criterion = nn.MSELoss(reduction='none')
loss = criterion(y_pred, y_true)
print(loss)  # tensor([0.2500, 0.0000, 0.2500])
```

`reduction='none'`은 샘플별로 가중치를 다르게 주고 싶을 때 유용합니다:

```python
criterion = nn.MSELoss(reduction='none')
loss_per_sample = criterion(y_pred, y_true)

# 중요한 샘플에 더 큰 가중치
weights = torch.tensor([1.0, 2.0, 1.0])
weighted_loss = (loss_per_sample * weights).mean()
```

---

## 실전 예제

### 간단한 회귀 모델

집 가격 예측 문제입니다:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 데이터 (면적, 방 개수 → 가격)
X = torch.tensor([[50.0, 2.0],
                  [80.0, 3.0],
                  [120.0, 4.0],
                  [100.0, 3.0]])

y = torch.tensor([[150.0],
                  [240.0],
                  [360.0],
                  [300.0]])

# 모델 정의
class HousePriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
    
    def forward(self, x):
        return self.linear(x)

model = HousePriceModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 학습
for epoch in range(100):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# 예측
with torch.no_grad():
    test_input = torch.tensor([[90.0, 3.0]])
    prediction = model(test_input)
    print(f'Predicted price: {prediction.item():.2f}')
```

### 시계열 예측

온도 예측 문제:

```python
import torch
import torch.nn as nn

# 시계열 데이터 (과거 5일 → 다음 날)
X = torch.randn(100, 5)  # 100개 샘플, 5일치 데이터
y = torch.randn(100, 1)  # 다음 날 온도

# RNN 모델
class TemperaturePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 1)
    
    def forward(self, x):
        x = x.unsqueeze(-1)  # (batch, seq, 1)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 마지막 출력만 사용
        return out

model = TemperaturePredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 학습
for epoch in range(50):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/50], Loss: {loss.item():.4f}')
```

---

## MAE vs MSE

두 손실 함수를 비교해봅시다.

### Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|$$

**장점:**

- 이상치(outlier)에 덜 민감
- 해석이 직관적 (단위가 원래 데이터와 같음)

**단점:**

- 0에서 미분 불가능
- 최적화가 MSE보다 어려움

### 비교 예시

실제값: [10, 20, 30] 예측값: [12, 18, 100] (마지막이 이상치)

**MSE:** $$(12-10)^2 + (18-20)^2 + (100-30)^2 = 4 + 4 + 4900 = 4908$$ $$\text{MSE} = 4908 / 3 = 1636$$

**MAE:** $$|12-10| + |18-20| + |100-30| = 2 + 2 + 70 = 74$$ $$\text{MAE} = 74 / 3 = 24.67$$

MSE는 이상치에 매우 큰 페널티를 주지만, MAE는 상대적으로 적은 페널티를 줍니다.

### 선택 기준

**MSE를 사용:**

- 이상치가 중요한 오류인 경우
- 큰 오차를 강하게 페널티 주고 싶을 때
- 최적화를 빠르게 하고 싶을 때

**MAE를 사용:**

- 이상치가 단순 노이즈인 경우
- 모든 오차를 동등하게 취급하고 싶을 때
- 손실값을 원래 단위로 해석하고 싶을 때

---

## Root Mean Squared Error (RMSE)

MSE의 제곱근을 취한 값입니다:

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}$$

**장점:**

- MSE의 장점 유지
- 원래 데이터와 같은 단위
- 해석이 더 직관적

**PyTorch 구현:**

```python
mse_loss = nn.MSELoss()
loss = mse_loss(y_pred, y_true)
rmse = torch.sqrt(loss)
print(f'RMSE: {rmse.item():.4f}')
```

학습 시에는 MSE를 사용하고, 평가 시에는 RMSE를 보고하는 것이 일반적입니다:

```python
# 학습
criterion = nn.MSELoss()
loss = criterion(y_pred, y_true)
loss.backward()

# 평가
with torch.no_grad():
    mse = criterion(y_pred, y_true)
    rmse = torch.sqrt(mse)
    print(f'Validation RMSE: {rmse.item():.2f}')
```

---

## 수치 안정성

직접 구현할 때보다 PyTorch의 `nn.MSELoss()`를 사용하는 것이 좋습니다.

**직접 구현:**

```python
# 수치적으로 불안정할 수 있음
loss = ((y_pred - y_true) ** 2).mean()
```

매우 큰 값이나 작은 값에서 오버플로우나 언더플로우가 발생할 수 있습니다.

**PyTorch 구현:**

```python
# 수치적으로 안정적
criterion = nn.MSELoss()
loss = criterion(y_pred, y_true)
```

내부적으로 최적화된 연산을 사용하여 안정성을 보장합니다.

---

## 배치 처리

배치 단위로 학습할 때 MSE의 동작:

```python
# 배치 크기 32, 출력 차원 10
y_pred = torch.randn(32, 10)
y_true = torch.randn(32, 10)

criterion = nn.MSELoss()
loss = criterion(y_pred, y_true)

print(loss.shape)  # torch.Size([]) - 스칼라
```

`reduction='mean'` (기본값)일 때:

1. 각 샘플의 각 출력에 대해 제곱 오차 계산
2. 모든 값을 평균 (32 × 10 = 320개 값의 평균)

**샘플별 손실이 필요하면:**

```python
criterion = nn.MSELoss(reduction='none')
loss = criterion(y_pred, y_true)
print(loss.shape)  # torch.Size([32, 10])

# 샘플별로 평균
sample_loss = loss.mean(dim=1)  # torch.Size([32])
```

---

## 가중치가 있는 MSE

특정 샘플이나 특징에 더 큰 가중치를 주고 싶을 때:

### 샘플별 가중치

```python
criterion = nn.MSELoss(reduction='none')
loss_per_sample = criterion(y_pred, y_true)

# 중요도에 따라 가중치
sample_weights = torch.tensor([1.0, 2.0, 1.5, ...])  # shape: (batch_size,)
weighted_loss = (loss_per_sample.mean(dim=1) * sample_weights).mean()
```

### 특징별 가중치

```python
criterion = nn.MSELoss(reduction='none')
loss_per_feature = criterion(y_pred, y_true)

# 특정 특징이 더 중요
feature_weights = torch.tensor([1.0, 2.0, 1.0, ...])  # shape: (num_features,)
weighted_loss = (loss_per_feature * feature_weights).mean()
```

---

## 실전 팁

### 정규화의 중요성

MSE는 스케일에 민감합니다. 데이터를 정규화하면 학습이 더 안정적입니다:

```python
# 데이터 정규화
X_mean = X.mean()
X_std = X.std()
X_normalized = (X - X_mean) / X_std

y_mean = y.mean()
y_std = y.std()
y_normalized = (y - y_mean) / y_std

# 학습
y_pred_normalized = model(X_normalized)
loss = criterion(y_pred_normalized, y_normalized)

# 예측 시 역정규화
y_pred = y_pred_normalized * y_std + y_mean
```

### 학습률 조정

MSE의 크기에 따라 학습률을 조정해야 할 수 있습니다:

```python
# 초기 손실 확인
initial_loss = criterion(model(X), y)
print(f'Initial loss: {initial_loss.item()}')

# 손실이 크면 작은 학습률
if initial_loss > 1000:
    lr = 0.0001
elif initial_loss > 100:
    lr = 0.001
else:
    lr = 0.01

optimizer = optim.Adam(model.parameters(), lr=lr)
```

### 조기 종료

검증 손실이 개선되지 않으면 학습 중단:

```python
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    # 학습
    train_loss = train(model, train_loader, criterion, optimizer)
    
    # 검증
    val_loss = validate(model, val_loader, criterion)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # 모델 저장
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch}')
        break
```

---

## 핵심 요약

- MSELoss는 회귀 문제의 표준 손실 함수
- 예측값과 실제값의 제곱 오차의 평균
- 큰 오차에 더 큰 페널티 부여
- 미분 가능하고 수학적으로 깔끔
- 이상치에 민감 (MAE는 덜 민감)
- 데이터 정규화가 중요
- PyTorch의 `nn.MSELoss()` 사용 권장

---

## 관련 개념

**선수 지식:**

- [[Machine Learning Basics]] - 머신러닝 기본 원리
- [[Loss Function]] - 손실 함수의 역할
- [[Gradient]] - Gradient의 의미

**관련 개념:**

- [[Polynomial Regression]] - MSE를 사용하는 예제
- [[Backpropagation]] - 손실로부터 gradient 계산
- [[Optimizer(Data Science)]] - 손실 최소화

**다음 단계:**

- 분류 문제 손실 함수: [[Softmax]], [[LogSoftmax]], [[NegativeLogLikelihood]]