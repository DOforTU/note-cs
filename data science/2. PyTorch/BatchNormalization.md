# Batch Normalization

## 정의

**Batch Normalization (배치 정규화)** 은 미니배치의 평균과 분산을 이용해 각 층의 입력을 정규화하는 기법입니다. 학습을 안정화하고 가속화하여 현대 딥러닝에서 필수적인 구성 요소가 되었습니다.

**핵심 아이디어:** 각 층의 입력 분포를 일정하게 유지하여 학습을 쉽게 만듭니다.

---

## 작동 원리

Batch Normalization은 두 단계로 작동합니다.

### 정규화 단계

미니배치의 평균과 분산을 계산하여 정규화합니다:

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

여기서:

- $\mu_B$: 배치 평균 $\frac{1}{m}\sum_{i=1}^{m} x_i$
- $\sigma_B^2$: 배치 분산 $\frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2$
- $\epsilon$: 수치 안정성을 위한 작은 값 (보통 $10^{-5}$)

이 과정을 거치면 평균이 0, 분산이 1인 분포가 됩니다.

### 스케일과 시프트

정규화만 하면 모든 값이 같은 분포로 제한되므로, 학습 가능한 파라미터로 복원합니다:

$$y = \gamma \hat{x} + \beta$$

여기서:

- $\gamma$: 학습 가능한 스케일 파라미터
- $\beta$: 학습 가능한 시프트 파라미터

네트워크는 $\gamma$와 $\beta$를 학습하여 최적의 분포를 찾습니다. 원한다면 원래 분포로 되돌릴 수도 있습니다 ($\gamma = \sqrt{\sigma_B^2 + \epsilon}$, $\beta = \mu_B$).

---

## 왜 필요한가?

### Internal Covariate Shift

딥러닝에서 각 층의 입력 분포는 이전 층의 파라미터가 업데이트될 때마다 계속 변합니다. 이를 **Internal Covariate Shift**라고 합니다.

예를 들어 첫 번째 층의 가중치가 변하면, 두 번째 층은 매번 다른 분포의 입력을 받습니다. 이는 학습을 불안정하게 만들고 느리게 합니다.

Batch Normalization은 각 층의 입력 분포를 안정화시켜 이 문제를 해결합니다.

### Gradient의 안정화

정규화를 통해 [[Gradient]]의 스케일이 일정하게 유지됩니다. 이는 [[Backpropagation]]에서 gradient가 너무 크거나 작아지는 것을 방지합니다.

특히 [[Activation Function]]의 포화 영역에 빠지는 것을 막아줍니다. [[Sigmoid]]나 Tanh 같은 함수에서 입력이 너무 크거나 작으면 gradient가 0에 가까워지는데, BN이 이를 방지합니다.

---

## 학습 vs 추론

### 학습 시

학습할 때는 **현재 미니배치의 통계량**을 사용합니다:

$$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i$$ $$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2$$

동시에 전체 데이터의 이동 평균(running mean)과 이동 분산(running variance)을 업데이트합니다:

$$\mu_{\text{running}} = \alpha \mu_{\text{running}} + (1-\alpha) \mu_B$$ $$\sigma^2_{\text{running}} = \alpha \sigma^2_{\text{running}} + (1-\alpha) \sigma_B^2$$

보통 $\alpha = 0.9$ (momentum)를 사용합니다.

### 추론 시

테스트나 예측할 때는 **학습 중 계산된 이동 평균**을 사용합니다:

$$\hat{x} = \frac{x - \mu_{\text{running}}}{\sqrt{\sigma^2_{\text{running}} + \epsilon}}$$

이는 배치 크기에 관계없이 일관된 출력을 보장합니다. 단일 샘플을 예측할 때도 문제없이 작동합니다.

---

## 배치 위치

Batch Normalization은 일반적으로 선형 변환 후, [[Activation Function]] 전에 배치합니다:

```python
x = self.fc1(x)          # 선형 변환
x = self.bn1(x)          # Batch Normalization
x = torch.relu(x)        # 활성화 함수
```

**이유:** 활성화 함수 전에 정규화하면 입력이 적절한 범위에 있어 포화를 방지할 수 있습니다.

일부 연구에서는 활성화 함수 후에 배치하는 것도 제안되었지만, 일반적으로는 전에 배치하는 것이 더 흔합니다.

---

## PyTorch 구현

### 1D (Fully Connected)

```python
import torch.nn as nn

bn = nn.BatchNorm1d(num_features=128)
x = torch.randn(32, 128)  # [배치, 특징]
output = bn(x)

print(x.mean())      # 평균이 임의의 값
print(output.mean()) # 평균이 ~0
print(x.std())       # 표준편차가 임의의 값
print(output.std())  # 표준편차가 ~1
```

### 2D (Convolutional)

```python
bn = nn.BatchNorm2d(num_channels=64)
x = torch.randn(32, 64, 28, 28)  # [배치, 채널, H, W]
output = bn(x)
```

각 채널마다 독립적으로 정규화합니다. 공간 차원(H, W)에 걸쳐 평균과 분산을 계산합니다.

### 모델에 통합

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        
        x = self.fc3(x)
        return x
```

### train()과 eval() 모드

Batch Normalization은 [[Dropout]]처럼 학습과 추론에서 다르게 동작하므로 모드 설정이 필수입니다:

```python
model = Model()

# 학습
model.train()  # 배치 통계 사용
for x, y in train_loader:
    output = model(x)
    # ...

# 평가
model.eval()  # 이동 평균 사용
with torch.no_grad():
    for x, y in test_loader:
        output = model(x)
        # ...
```

---

## 장점

### 빠른 학습

BN을 사용하면 더 큰 학습률을 안전하게 사용할 수 있습니다. 입력이 정규화되어 있으므로 파라미터 업데이트가 안정적입니다.

일반적으로 학습 속도가 2~10배 빨라집니다. 같은 성능에 도달하는 데 필요한 epoch 수가 크게 줄어듭니다.

### 정규화 효과

BN 자체가 정규화 역할을 하여 [[Dropout]]의 필요성을 줄입니다. 배치마다 약간의 노이즈가 추가되는 효과가 있어 과적합을 방지합니다.

많은 경우 BN만으로 충분한 정규화 효과를 얻을 수 있습니다.

### 초기화에 덜 민감

가중치 초기화가 다소 잘못되어도 BN이 이를 보정해줍니다. 각 층의 입력이 정규화되므로 초기 가중치의 영향이 줄어듭니다.

### Gradient 개선

[[Sigmoid]]나 Tanh 같은 함수를 사용해도 gradient vanishing이 덜 발생합니다. 입력이 정규화되어 활성화 함수의 포화 영역에 덜 빠집니다.

---

## 단점

### 작은 배치 크기

배치 크기가 작으면 평균과 분산의 추정이 불안정해집니다. 통계량이 정확하지 않아 성능이 저하될 수 있습니다.

일반적으로 배치 크기가 최소 16~32는 되어야 안정적입니다. 매우 작은 배치(예: 2)에서는 사용하지 않는 것이 좋습니다.

### 추가 계산

평균, 분산 계산과 정규화 과정이 추가 계산을 요구합니다. 특히 추론 시에도 정규화를 수행해야 하므로 약간의 오버헤드가 있습니다.

### RNN에 부적합

순환 신경망(RNN)에서는 시퀀스 길이가 가변적이므로 BN을 적용하기 어렵습니다. 대신 Layer Normalization 같은 변형을 사용합니다.

### 복잡성 증가

학습과 추론에서 다르게 동작하므로 구현이 복잡해집니다. `model.train()`과 `model.eval()`을 정확히 설정해야 합니다.

---

## Batch Normalization 변형

### Layer Normalization

배치 차원이 아닌 **특징 차원**에 걸쳐 정규화합니다. 각 샘플을 독립적으로 처리하므로 배치 크기에 무관합니다.

Transformer 모델에서 주로 사용되며, RNN에도 적합합니다.

### Instance Normalization

각 샘플의 각 채널을 독립적으로 정규화합니다. 스타일 전이(Style Transfer) 같은 작업에서 효과적입니다.

### Group Normalization

채널을 그룹으로 나누어 각 그룹 내에서 정규화합니다. 작은 배치에서도 안정적으로 작동합니다.

---

## 실전 가이드

### 언제 사용하는가?

**BN이 효과적인 경우:**

- 깊은 네트워크 (>10층)
- 학습 속도가 중요할 때
- 배치 크기가 충분히 클 때 (>16)
- CNN, MLP 같은 구조

**BN이 덜 필요한 경우:**

- 얕은 네트워크 (<5층)
- 배치 크기가 매우 작을 때
- RNN, Transformer (Layer Norm 사용)

### 하이퍼파라미터

BN은 하이퍼파라미터가 거의 없어 사용이 쉽습니다:

**momentum:** 이동 평균 계산 시 사용 (기본값: 0.1)

```python
bn = nn.BatchNorm1d(128, momentum=0.1)
```

**eps:** 수치 안정성 (기본값: 1e-5)

```python
bn = nn.BatchNorm1d(128, eps=1e-5)
```

대부분의 경우 기본값으로 충분합니다.

### [[Dropout]]과 함께 사용

BN과 Dropout을 함께 사용하면 효과가 감소할 수 있습니다. 둘 다 정규화 역할을 하기 때문입니다.

**일반적인 선택:**

- BN만 사용 (더 흔함)
- Dropout만 사용
- 둘 다 사용 (신중하게, Dropout 확률을 낮춤)

```python
# BN과 Dropout 함께 사용 시
x = self.fc1(x)
x = self.bn1(x)
x = torch.relu(x)
x = self.dropout(x)  # 낮은 확률 (예: 0.2)
```

---

## CNN에서의 사용

Convolutional layer에서는 `BatchNorm2d`를 사용합니다:

```python
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        return x
```

각 채널마다 독립적으로 정규화되므로, 채널 수만큼 $\gamma$와 $\beta$ 파라미터가 있습니다.

---

## 수치 예제

배치 크기 4, 특징 2개인 경우:

```python
import torch
import torch.nn as nn

x = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0]
])

bn = nn.BatchNorm1d(2)
bn.eval()  # 평가 모드

# 첫 번째 특징의 평균과 분산
mean_0 = (1 + 3 + 5 + 7) / 4 = 4.0
var_0 = ((1-4)² + (3-4)² + (5-4)² + (7-4)²) / 4 = 5.0

# 정규화
x_norm_0 = (1 - 4) / sqrt(5 + 1e-5) = -1.34
```

실제로는 $\gamma$와 $\beta$로 스케일과 시프트가 추가됩니다.

---

## 역사적 배경

Batch Normalization은 2015년 Sergey Ioffe와 Christian Szegedy가 제안했습니다. "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"라는 논문에서 소개되었습니다.

**영향:** BN의 등장으로 매우 깊은 네트워크(수백 층)를 안정적으로 학습시킬 수 있게 되었습니다. ResNet, Inception 같은 현대 구조의 핵심 요소가 되었습니다.

초기에는 Internal Covariate Shift 감소가 주된 이유로 설명되었지만, 최근 연구들은 더 부드러운 [[Loss Function]] 표면을 만드는 것이 주된 효과라고 제안합니다.

---

## 핵심 요약

Batch Normalization은 미니배치의 평균과 분산으로 각 층의 입력을 정규화하는 기법입니다. 학습을 안정화하고 가속화하며, 더 큰 학습률 사용을 가능하게 합니다. 선형 변환 후 [[Activation Function]] 전에 배치하며, 학습 시에는 배치 통계를, 추론 시에는 이동 평균을 사용합니다. 배치 크기가 충분히 클 때 가장 효과적이며, 현대 딥러닝의 표준 구성 요소입니다.

---

## 관련 개념

**상위 개념:**

- [[Neural Network Components]] - 신경망 구성 요소

**정규화 기법:**

- [[Dropout]] - 뉴런 제거 정규화
- Layer Normalization - 특징 차원 정규화

**함께 사용:**

- [[Activation Function]] - 활성화 함수
- [[ReLU]] - 주로 사용되는 활성화

**기반 개념:**

- [[Forward Pass]] - 순전파
- [[Backpropagation]] - 역전파
- [[Gradient]] - 그래디언트
- [[Deep Learning Core Concepts]] - 딥러닝 기초