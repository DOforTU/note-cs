# Dropout

## 정의

**Dropout**은 학습 중에 무작위로 일부 뉴런을 제거하여 과적합을 방지하는 정규화 기법입니다. 마치 매번 다른 구조의 신경망을 학습시키는 효과를 냅니다.

**핵심 아이디어:** 각 학습 단계마다 확률 $p$로 뉴런을 비활성화합니다.

---

## 작동 원리

### 학습 시

학습할 때는 각 뉴런을 확률 $p$로 0으로 만듭니다. 살아남은 뉴런은 $(1-p)$로 나누어 스케일을 조정합니다.

**예시:** $p = 0.5$일 때

```
원본:     [1.0, 2.0, 3.0, 4.0, 5.0]
마스크:   [1,   0,   1,   0,   1  ]
출력:     [2.0, 0.0, 6.0, 0.0, 10.0]
```

0으로 만들어진 뉴런은 해당 iteration에서 완전히 제거된 것처럼 작동합니다. 스케일링 $(1/(1-p) = 2)$을 통해 기댓값을 유지합니다.

### 추론 시

테스트나 예측할 때는 dropout을 적용하지 않습니다. 모든 뉴런을 사용하여 안정적인 예측을 만듭니다.

```
입력:  [1.0, 2.0, 3.0, 4.0, 5.0]
출력:  [1.0, 2.0, 3.0, 4.0, 5.0]
```

스케일링을 학습 시에 했으므로 추론 시에는 추가 조정이 필요 없습니다.

---

## 왜 효과적인가?

### 앙상블 효과

매 iteration마다 다른 뉴런이 제거되므로, 사실상 여러 다른 신경망을 학습하는 것과 같습니다. 최종 모델은 이러한 많은 하위 네트워크의 평균으로 볼 수 있습니다.

예를 들어 100개 뉴런에 $p=0.5$를 적용하면 $2^{100}$가지의 다른 네트워크 조합이 가능합니다. 이는 강력한 앙상블 모델을 만드는 효과가 있습니다.

### 공동 적응 방지

Dropout 없이는 어떤 뉴런이 다른 뉴런의 실수를 보정하도록 학습될 수 있습니다. 이를 "공동 적응(co-adaptation)"이라 하며, 과적합의 원인이 됩니다.

Dropout은 각 뉴런이 다른 뉴런에 의존하지 않고 독립적으로 유용한 특징을 학습하도록 강제합니다. 무작위로 동료 뉴런이 사라지므로 특정 뉴런에 과도하게 의존할 수 없습니다.

### 희소성 유도

Dropout은 네트워크가 더 희소하고 강건한 표현을 학습하도록 유도합니다. 중복되거나 불필요한 특징 대신, 정말 중요한 특징에 집중하게 만듭니다.

---

## 수학적 표현

[[Forward Pass]]에서 dropout은 다음과 같이 표현됩니다:

$$\tilde{h} = m \odot h$$

여기서:

- $h$: 원래 활성화
- $m$: 베르누이 마스크 ($m_i \sim \text{Bernoulli}(1-p)$)
- $\odot$: 원소별 곱셈
- $\tilde{h}$: Dropout이 적용된 활성화

**스케일링 포함:**

$$\tilde{h} = \frac{1}{1-p} m \odot h$$

이 스케일링은 학습 시에 적용하여 추론 시 별도 조정이 필요 없도록 합니다.

---

## 파라미터 선택

### Dropout 확률 $p$

일반적인 값은 0.2에서 0.5 사이입니다.

**은닉층:** $p = 0.3$ ~ $0.5$

- 너무 높으면 정보 손실
- 너무 낮으면 정규화 효과 미미

**입력층:** $p = 0.1$ ~ $0.2$

- 입력 특징이 직접 제거되므로 보수적으로

**출력층:** 일반적으로 사용 안 함

- 최종 예측에 직접 영향

### 네트워크 크기에 따른 조정

큰 네트워크일수록 과적합 위험이 크므로 높은 $p$ 값을 사용할 수 있습니다. 작은 네트워크는 용량이 제한적이므로 낮은 $p$ 값을 사용해야 합니다.

---

## PyTorch 구현

### 기본 사용법

```python
import torch
import torch.nn as nn

m = nn.Dropout(p=0.5)

# 학습 모드
m.train()
input = torch.ones(10)
output = m(input)
print(output)
# tensor([2., 0., 2., 0., 2., 2., 0., 2., 0., 0.])

# 평가 모드
m.eval()
output = m(input)
print(output)
# tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
```

### 모델에 통합

```python
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)  # ReLU 후 적용
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
```

### train()과 eval() 모드

모델을 학습하거나 평가할 때 반드시 모드를 설정해야 합니다:

```python
model = Classifier()

# 학습
model.train()  # Dropout 활성화
for x, y in train_loader:
    output = model(x)
    loss = criterion(output, y)
    # ...

# 평가
model.eval()  # Dropout 비활성화
with torch.no_grad():
    for x, y in test_loader:
        output = model(x)
        # ...
```

`model.train()`과 `model.eval()`을 잊으면 학습과 평가가 올바르게 작동하지 않습니다.

---

## Dropout 배치 위치

### [[Activation Function]] 이후

일반적으로 dropout은 활성화 함수 다음에 배치합니다:

```python
x = torch.relu(self.fc1(x))
x = self.dropout(x)  # 활성화 후
```

이는 활성화된 뉴런 중 일부를 제거하는 것으로, 가장 일반적인 패턴입니다.

### [[BatchNormalization]]과 함께 사용

Dropout과 Batch Normalization을 함께 사용하면 효과가 감소할 수 있습니다. 둘 다 정규화 효과가 있기 때문입니다.

**일반적인 선택:**

- Batch Normalization만 사용 (더 흔함)
- Dropout만 사용
- 둘 다 사용 (신중하게)

```python
# BN과 함께 사용 시
x = self.fc1(x)
x = self.bn1(x)
x = torch.relu(x)
x = self.dropout(x)
```

---

## Dropout 변형

### Spatial Dropout (Dropout2d)

이미지에서 채널 전체를 제거합니다. 인접한 픽셀들이 강한 상관관계를 가지므로, 개별 픽셀 대신 전체 특징 맵을 제거하는 것이 더 효과적입니다.

```python
m = nn.Dropout2d(p=0.3)
x = torch.randn(32, 64, 28, 28)  # [배치, 채널, H, W]
output = m(x)
# 일부 채널이 완전히 0이 됨
```

### DropConnect

뉴런 대신 **가중치**를 무작위로 제거합니다. Dropout보다 더 강력한 정규화 효과가 있지만 계산 비용이 높습니다.

### Variational Dropout

RNN에서 같은 마스크를 모든 time step에 적용합니다. 시퀀스 전체에 걸쳐 일관된 정규화를 제공합니다.

---

## 장단점

### 장점

**강력한 정규화:** 과적합을 효과적으로 방지합니다. 특히 데이터가 적고 모델이 클 때 유용합니다.

**구현이 간단:** 몇 줄의 코드로 추가 가능하며, 추가 하이퍼파라미터가 거의 없습니다.

**범용성:** CNN, RNN, Transformer 등 다양한 구조에 적용 가능합니다.

### 단점

**학습 속도 감소:** 매번 다른 네트워크를 학습하므로 수렴이 느려질 수 있습니다. 더 많은 epoch이 필요할 수 있습니다.

**추가 메모리:** 마스크를 저장하고 [[Backpropagation]]을 수행해야 하므로 메모리가 더 필요합니다.

**작은 데이터셋에서 과도한 정규화:** 데이터가 매우 적으면 dropout이 오히려 학습을 방해할 수 있습니다.

---

## Dropout vs 다른 정규화

### Dropout vs L2 Regularization

**L2 정규화:**

- 가중치의 크기에 직접 페널티
- 모든 가중치를 작게 유지
- 수학적으로 명확

**Dropout:**

- 네트워크 구조를 무작위로 변경
- 특정 뉴런에 대한 의존성 감소
- 더 강력한 효과

두 방법을 함께 사용할 수도 있습니다.

### Dropout vs Batch Normalization

**Batch Normalization:**

- 활성화를 정규화
- 학습 안정화
- 더 빠른 수렴

**Dropout:**

- 뉴런을 무작위로 제거
- 과적합 방지
- 앙상블 효과

최근에는 Batch Normalization이 더 선호되는 경향이 있습니다. BN이 있으면 dropout 없이도 좋은 성능을 보이는 경우가 많습니다.

---

## 실전 가이드

### 언제 사용하는가?

**Dropout이 효과적인 경우:**

- 과적합이 명확하게 관찰될 때
- 학습 데이터가 제한적일 때
- 큰 fully connected layer가 있을 때

**Dropout이 덜 필요한 경우:**

- Batch Normalization을 사용할 때
- 데이터가 충분히 많을 때
- 모델이 이미 작을 때

### 하이퍼파라미터 튜닝

기본적으로 $p = 0.5$에서 시작합니다. 과적합이 심하면 증가시키고, 언더피팅이 발생하면 감소시킵니다.

**실험 순서:**

1. Dropout 없이 학습
2. 과적합 확인
3. $p = 0.5$로 시작
4. 성능에 따라 조정

### 모니터링

학습 중에 train loss와 validation loss를 비교합니다:

```python
for epoch in range(num_epochs):
    # 학습
    model.train()
    train_loss = 0
    for x, y in train_loader:
        # ...
        train_loss += loss.item()
    
    # 검증
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            # ...
            val_loss += loss.item()
    
    print(f'Epoch {epoch}: Train {train_loss:.4f}, Val {val_loss:.4f}')
```

Train loss가 계속 감소하는데 validation loss가 증가하면 과적합의 신호입니다. Dropout 확률을 높입니다.

---

## 역사적 배경

Dropout은 2012년 Geoffrey Hinton과 그의 학생들이 제안했습니다. ImageNet 대회에서 AlexNet이 사용하면서 유명해졌습니다.

**핵심 통찰:** 신경망이 특정 뉴런에 과도하게 의존하는 것을 막아야 한다는 아이디어는 매우 단순하지만 강력했습니다.

초기에는 주로 fully connected layer에 사용되었지만, 이후 CNN, RNN 등으로 확장되었습니다. 최근에는 [[BatchNormalization]]의 등장으로 사용 빈도가 다소 줄었지만, 여전히 중요한 정규화 기법입니다.

---

## 핵심 요약

Dropout은 학습 중에 무작위로 뉴런을 제거하여 과적합을 방지하는 정규화 기법입니다. 앙상블 효과와 공동 적응 방지를 통해 모델의 일반화 성능을 향상시킵니다. 학습 시에만 적용하고 추론 시에는 모든 뉴런을 사용합니다. 일반적으로 $p = 0.3 \sim 0.5$ 값을 사용하며, [[Activation Function]] 이후에 배치합니다. [[BatchNormalization]]과 함께 사용할 때는 신중해야 합니다.

---

## 관련 개념

**상위 개념:**

- [[Neural Network Components]] - 신경망 구성 요소

**정규화 기법:**

- [[BatchNormalization]] - 배치 정규화
- L2 Regularization - 가중치 감쇠

**함께 사용:**

- [[Activation Function]] - 활성화 함수
- [[ReLU]] - 활성화 함수

**기반 개념:**

- [[Forward Pass]] - 순전파
- [[Backpropagation]] - 역전파
- [[Deep Learning Core Concepts]] - 딥러닝 기초