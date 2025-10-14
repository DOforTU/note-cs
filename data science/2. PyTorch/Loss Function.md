# Loss Function

## 정의

**손실 함수(Loss Function)** 는 모델의 예측값과 실제값 사이의 차이를 수치화하는 함수입니다. 모델이 얼마나 "잘못" 예측하고 있는지를 하나의 숫자로 표현합니다.

$$\text{Loss} = f(\text{예측값}, \text{실제값})$$

손실값이 작을수록 모델의 예측이 정확하다는 의미입니다.

---
## 왜 필요한가?

머신러닝과 딥러닝에서 모델을 학습시킨다는 것은 **"손실을 최소화하는 파라미터를 찾는 것"** 입니다. 손실 함수가 없다면:

- 모델의 성능을 정량적으로 측정할 수 없음
- 어떤 방향으로 파라미터를 업데이트해야 할지 알 수 없음
- [[Gradient]]를 계산할 수 없어 [[Backpropagation]]이 불가능함

손실 함수는 학습 과정의 **나침반** 역할을 합니다.

---
## 기본 원리

### 1. 예측값 계산

모델이 입력 데이터를 받아 예측값을 출력합니다 ([[Forward Pass]]).

```python
# 예시: 선형 모델
prediction = w * x + b
```

### 2. 손실 계산

예측값과 실제값을 비교하여 오차를 계산합니다.

```python
loss = loss_function(prediction, actual)
```

### 3. 최적화

손실을 최소화하는 방향으로 파라미터를 업데이트합니다 ([[Gradient]] 기반).

---
## 주요 손실 함수 종류

### 회귀 문제 (연속적인 값 예측)

**평균 제곱 오차 (Mean Squared Error, MSE)**

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- 예측값과 실제값의 차이를 제곱하여 평균
- 큰 오차에 더 큰 페널티 부여
- 가장 널리 사용되는 회귀 손실 함수

```python
# PyTorch 예시
import torch
import torch.nn as nn

criterion = nn.MSELoss()
prediction = torch.tensor([2.5, 3.0, 4.5])
actual = torch.tensor([3.0, 3.0, 5.0])
loss = criterion(prediction, actual)
print(loss)  # tensor(0.1667)
```

**평균 절대 오차 (Mean Absolute Error, MAE)**

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

- 절댓값을 사용하여 이상치에 덜 민감
- MSE보다 해석이 직관적

---
### 분류 문제 (카테고리 예측)

**Cross-Entropy Loss**

이진 분류: $$\text{BCE} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

다중 클래스 분류: $$\text{CE} = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)$$

- 확률 분포 간의 차이를 측정
- 분류 문제의 표준 손실 함수

```python
# PyTorch 예시
criterion = nn.CrossEntropyLoss()

# 로짓 (softmax 적용 전 값)
logits = torch.tensor([[2.0, 1.0, 0.1]])
# 정답 레이블 (클래스 인덱스)
target = torch.tensor([0])

loss = criterion(logits, target)
```

---
## 손실 함수의 역할

### 1. 학습 가이드

손실 함수의 [[Gradient]]를 계산하여 파라미터를 어느 방향으로 업데이트할지 결정합니다.

```python
loss.backward()  # 그래디언트 계산
optimizer.step()  # 손실을 줄이는 방향으로 파라미터 업데이트
```

### 2. 성능 지표

모델의 학습 진행 상황을 모니터링하는 지표로 사용됩니다.

```python
print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
# Epoch 1, Loss: 0.8523
# Epoch 2, Loss: 0.6241
# Epoch 3, Loss: 0.4512
# ...
```

손실이 감소하면 모델이 학습되고 있다는 신호입니다.

### 3. 모델 비교

서로 다른 모델이나 하이퍼파라미터를 비교할 때 기준이 됩니다.

---
## 좋은 손실 함수의 조건

1. **미분 가능**: [[Gradient]]를 계산할 수 있어야 함
2. **문제에 적합**: 회귀에는 MSE, 분류에는 Cross-Entropy
3. **수치적 안정성**: 오버플로우나 언더플로우가 발생하지 않아야 함
4. **직관적 해석**: 손실값이 실제 오차를 잘 반영해야 함

---
## 실전 예제:

다항 회귀에서 MSE를 사용하는 예:

```python
import torch

# 모델 파라미터
a = torch.randn((), requires_grad=True)
b = torch.randn((), requires_grad=True)

# Forward pass
y_pred = a * x + b

# 손실 계산 (MSE)
loss = ((y_pred - y) ** 2).mean()

# Backward pass
loss.backward()

# 파라미터 업데이트
with torch.no_grad():
    a -= learning_rate * a.grad
    b -= learning_rate * b.grad
```

이 패턴은 회귀 문제([[Polynomial Regression]])와 분류 문제 모두에 동일하게 적용됩니다.

---
## 핵심 요약

- **손실 함수는 모델의 예측 오차를 수치화**합니다
- **학습의 목표는 손실을 최소화**하는 것입니다
- **회귀 문제**에는 주로 MSE, **분류 문제**에는 Cross-Entropy 사용
- 손실 함수의 [[Gradient]]를 통해 [[Backpropagation]]이 가능합니다

---

## 관련 개념

- [[Gradient]] - 손실을 줄이는 방향 계산
- [[Forward Pass]] - 예측값 계산 과정
- [[Backpropagation]] - 손실로부터 그래디언트 계산
- [[MSELoss]] - PyTorch의 MSE 구현