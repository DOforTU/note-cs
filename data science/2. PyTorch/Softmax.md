# Softmax

## 정의

**Softmax**는 벡터의 각 원소를 0과 1 사이의 확률로 변환하는 함수입니다. 모든 출력의 합이 1이 되므로, 확률 분포로 해석할 수 있습니다. 주로 다중 클래스 분류 문제에서 사용됩니다.

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

여기서:

- $z_i$: $i$번째 클래스의 로짓(logit) 값
- $K$: 전체 클래스 개수
- $e$: 자연상수 (약 2.718)

---

## 직관적 이해

신경망의 마지막 층은 각 클래스에 대한 점수(로짓)를 출력합니다. 이 점수들을 확률로 바꾸고 싶습니다.

**예시: 이미지 분류**

고양이, 개, 새 세 클래스를 분류하는 모델이 다음 점수를 출력했습니다:

- 고양이: 3.0
- 개: 1.0
- 새: 0.2

이것을 직접 확률로 해석하기는 어렵습니다. Softmax를 적용하면:

$$P(\text{고양이}) = \frac{e^{3.0}}{e^{3.0} + e^{1.0} + e^{0.2}} = \frac{20.09}{20.09 + 2.72 + 1.22} = \frac{20.09}{24.03} = 0.836$$

$$P(\text{개}) = \frac{e^{1.0}}{24.03} = \frac{2.72}{24.03} = 0.113$$

$$P(\text{새}) = \frac{e^{0.2}}{24.03} = \frac{1.22}{24.03} = 0.051$$

합계: 0.836 + 0.113 + 0.051 = 1.0 ✓

이제 "고양이일 확률 83.6%, 개일 확률 11.3%, 새일 확률 5.1%"로 해석할 수 있습니다.

---

## 왜 지수 함수를 사용하는가?

단순히 정규화하면 안 될까요?

$$\text{normalize}(z_i) = \frac{z_i}{\sum_{j=1}^{K} z_j}$$

**문제 1: 음수 처리**

로짓이 음수일 수 있습니다:

- 로짓: [2.0, -1.0, -3.0]
- 단순 정규화: $\frac{2.0}{2.0 - 1.0 - 3.0} = \frac{2.0}{-2.0} = -1.0$ (음수 확률!)

지수 함수는 항상 양수를 출력합니다: $$e^{-1.0} = 0.368 > 0$$ $$e^{-3.0} = 0.050 > 0$$

**문제 2: 크기 차이 강조**

로짓: [3.0, 2.9, 0.1]

단순 정규화: $$\frac{3.0}{6.0} = 0.5, \quad \frac{2.9}{6.0} = 0.48, \quad \frac{0.1}{6.0} = 0.02$$

차이가 명확하지 않습니다.

Softmax: $$\frac{e^{3.0}}{e^{3.0} + e^{2.9} + e^{0.1}} \approx 0.52, \quad 0.47, \quad 0.01$$

지수 함수가 차이를 더 강조합니다. 큰 값은 더 크게, 작은 값은 더 작게 만듭니다.

---

## 수학적 성질

### 항상 확률 분포

모든 출력은 0과 1 사이이고, 합이 1입니다:

$$0 < \text{softmax}(z_i) < 1$$ $$\sum_{i=1}^{K} \text{softmax}(z_i) = 1$$

**증명:**

분자는 $e^{z_i} > 0$이므로 출력은 항상 양수입니다.

분모가 분자보다 크거나 같으므로 (다른 항들도 포함): $$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}} \leq 1$$

합: $$\sum_{i=1}^{K} \text{softmax}(z_i) = \sum_{i=1}^{K} \frac{e^{z_i}}{\sum_j e^{z_j}} = \frac{\sum_i e^{z_i}}{\sum_j e^{z_j}} = 1$$

### 최댓값 강조

로짓이 크면 softmax 출력도 훨씬 큽니다:

```python
import torch
import torch.nn.functional as F

z = torch.tensor([1.0, 2.0, 3.0])
print(F.softmax(z, dim=0))
# tensor([0.0900, 0.2447, 0.6652])
```

로짓이 [1, 2, 3]일 때 softmax는 [0.09, 0.24, 0.67]입니다. 가장 큰 값(3.0)이 전체의 67%를 차지합니다.

로짓 차이를 더 크게 하면:

```python
z = torch.tensor([1.0, 2.0, 10.0])
print(F.softmax(z, dim=0))
# tensor([0.0001, 0.0003, 0.9996])
```

가장 큰 값이 거의 100%를 차지합니다. 이것을 "winner-takes-all" 효과라고 합니다.

### 온도 스케일링 (Temperature Scaling)

Softmax에 온도 파라미터 $T$를 추가할 수 있습니다:

$$\text{softmax}(z_i, T) = \frac{e^{z_i/T}}{\sum_{j} e^{z_j/T}}$$

**$T = 1$:** 표준 softmax

**$T > 1$:** 더 부드러운 확률 분포 (평평)

```python
z = torch.tensor([1.0, 2.0, 3.0])
print(F.softmax(z / 2.0, dim=0))  # T=2
# tensor([0.1863, 0.3072, 0.5065])
```

**$T < 1$:** 더 날카로운 확률 분포 (뾰족)

```python
print(F.softmax(z / 0.5, dim=0))  # T=0.5
# tensor([0.0171, 0.1142, 0.8687])
```

온도를 조절하여 모델의 확신도를 제어할 수 있습니다.

---

## PyTorch 구현

### 기본 사용법

```python
import torch
import torch.nn.functional as F

# 로짓 (신경망의 raw 출력)
logits = torch.tensor([2.0, 1.0, 0.1])

# Softmax 적용
probs = F.softmax(logits, dim=0)
print(probs)
# tensor([0.6590, 0.2424, 0.0986])

print(probs.sum())
# tensor(1.0000)
```

`dim` 파라미터는 어느 차원에 대해 softmax를 적용할지 지정합니다.

### 배치 처리

```python
# 배치 크기 4, 클래스 개수 3
logits = torch.randn(4, 3)

# 각 샘플마다 softmax (dim=1)
probs = F.softmax(logits, dim=1)
print(probs.shape)  # torch.Size([4, 3])

# 각 샘플의 확률 합은 1
print(probs.sum(dim=1))
# tensor([1.0000, 1.0000, 1.0000, 1.0000])
```

**중요:** `dim=1`을 지정하여 클래스 차원에 대해 softmax를 적용합니다. `dim=0`으로 하면 배치 차원에 대해 적용되어 잘못된 결과가 나옵니다.

### nn.Softmax 레이어

모델에 softmax를 포함시킬 수 있습니다:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 3),
    nn.Softmax(dim=1)  # 출력층에 softmax
)

x = torch.randn(2, 10)
output = model(x)
print(output)
# tensor([[0.2891, 0.3515, 0.3594],
#         [0.3721, 0.2456, 0.3823]], grad_fn=<SoftmaxBackward>)
```

**주의:** 학습 시에는 보통 softmax를 모델에 포함시키지 않습니다. 이유는 뒤에서 설명합니다.

---

## 미분 (Gradient)

[[Backpropagation]]을 위해 softmax의 미분을 계산해야 합니다.

### 수식 유도

$s_i = \text{softmax}(z_i)$라고 하면:

같은 원소에 대한 미분: $$\frac{\partial s_i}{\partial z_i} = s_i(1 - s_i)$$

다른 원소에 대한 미분: $$\frac{\partial s_i}{\partial z_j} = -s_i s_j \quad (i \neq j)$$

**해석:**

- $s_i$가 1에 가까우면 $s_i(1-s_i) \approx 0$: gradient가 작음
- $s_i$가 0.5면 $s_i(1-s_i) = 0.25$: gradient가 최대

### PyTorch 자동 미분

```python
z = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
s = F.softmax(z, dim=0)

# 첫 번째 출력에 대한 gradient
s[0].backward()
print(z.grad)
# tensor([0.0818, -0.0222, -0.0596])
```

PyTorch의 [[Autograd]]가 자동으로 계산합니다.

---

## 수치 안정성 문제

Softmax를 직접 구현하면 수치적 문제가 발생할 수 있습니다.

### 오버플로우

```python
import numpy as np

z = np.array([1000, 1001, 1002])
exp_z = np.exp(z)
print(exp_z)
# [inf inf inf]  # 오버플로우!
```

$e^{1000}$은 너무 커서 컴퓨터로 표현할 수 없습니다.

### 해결: Log-Sum-Exp Trick

모든 로짓에서 최댓값을 빼도 softmax 결과는 같습니다:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}} = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$

**증명:**

$$\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_i} \cdot e^{-c}}{\sum_j e^{z_j} \cdot e^{-c}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**안정적 구현:**

```python
def stable_softmax(z):
    z_shifted = z - z.max()  # 최댓값을 뺌
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum()

z = np.array([1000, 1001, 1002])
print(stable_softmax(z))
# [0.09003057 0.24472847 0.66524096]
```

이제 $e^0 = 1$이 최댓값이 되어 오버플로우가 발생하지 않습니다.

**PyTorch는 자동으로 처리:**

```python
z = torch.tensor([1000., 1001., 1002.])
probs = F.softmax(z, dim=0)
print(probs)
# tensor([0.0900, 0.2447, 0.6652])
```

`F.softmax()`는 내부적으로 log-sum-exp trick을 사용하므로 직접 구현하지 말고 PyTorch 함수를 사용해야 합니다.

---

## 분류 문제에서의 사용

### 이진 분류 (Binary Classification)

클래스가 2개일 때는 softmax 대신 **Sigmoid**를 사용하는 것이 일반적입니다:

```python
# Sigmoid (이진 분류)
z = torch.tensor([2.0])
prob = torch.sigmoid(z)
print(prob)  # tensor([0.8808])

# Softmax도 가능하지만 비효율적
z = torch.tensor([2.0, 0.0])  # [positive, negative]
probs = F.softmax(z, dim=0)
print(probs)  # tensor([0.8808, 0.1192])
```

두 개의 확률이 항상 합이 1이므로, 하나만 계산하면 됩니다.

### 다중 클래스 분류 (Multi-class Classification)

클래스가 3개 이상일 때 softmax를 사용합니다:

```python
import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.classifier = nn.Linear(64 * 14 * 14, num_classes)
        # Softmax는 여기에 추가하지 않음!
    
    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits  # 로짓만 반환

model = ImageClassifier(num_classes=10)
x = torch.randn(1, 3, 32, 32)
logits = model(x)

# 예측 시에 softmax 적용
probs = F.softmax(logits, dim=1)
predicted_class = torch.argmax(probs, dim=1)
```

### 왜 모델에 Softmax를 포함하지 않는가?

**이유 1: 수치 안정성**

[[Loss Function]]으로 Cross-Entropy를 사용할 때, PyTorch의 `nn.CrossEntropyLoss()`는 내부적으로 안정적인 softmax 계산을 수행합니다:

```python
# 잘못된 방법 (불안정)
logits = model(x)
probs = F.softmax(logits, dim=1)
loss = -torch.log(probs[target])

# 올바른 방법 (안정적)
logits = model(x)
loss = F.cross_entropy(logits, target)  # 내부에서 안정적으로 처리
```

**이유 2: 효율성**

학습 시에는 확률 자체가 필요 없고 로짓만으로 gradient를 계산할 수 있습니다. Softmax를 계산하는 것은 불필요한 연산입니다.

---

## 실전 예제

### MNIST 숫자 분류

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# 데이터 로드
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)

# 모델 정의
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
        x = self.fc3(x)  # 로짓만 반환
        return x

model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # 내부에서 softmax 처리

# 학습
model.train()
for epoch in range(3):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Forward pass (로짓만)
        logits = model(data)
        
        # 손실 계산 (CrossEntropyLoss가 softmax 포함)
        loss = criterion(logits, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

# 평가
model.eval()
with torch.no_grad():
    test_data = torch.randn(1, 1, 28, 28)
    logits = model(test_data)
    
    # 추론 시에만 softmax 적용
    probs = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1)
    
    print(f'Predicted: {predicted_class.item()}')
    print(f'Probabilities: {probs[0]}')
```

### Top-K 예측

가장 가능성 높은 K개 클래스를 찾을 때:

```python
logits = model(x)
probs = F.softmax(logits, dim=1)

# Top-3 예측
top_probs, top_indices = torch.topk(probs, k=3, dim=1)

print('Top-3 predictions:')
for i in range(3):
    print(f'{i+1}. Class {top_indices[0, i].item()}: {top_probs[0, i].item():.4f}')

# 출력 예시:
# Top-3 predictions:
# 1. Class 7: 0.8523
# 2. Class 2: 0.0821
# 3. Class 9: 0.0312
```

---

## Softmax vs Sigmoid

### Sigmoid (이진 분류)

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

하나의 값을 [0, 1]로 변환합니다:

```python
z = torch.tensor([2.0])
prob = torch.sigmoid(z)
print(prob)  # tensor([0.8808])
```

### Softmax (다중 클래스)

여러 값을 확률 분포로 변환합니다:

```python
z = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(z, dim=0)
print(probs)  # tensor([0.6590, 0.2424, 0.0986])
print(probs.sum())  # tensor(1.)
```

### 관계

실제로 이진 분류에서 softmax를 사용하면 sigmoid와 동등합니다:

```python
# Sigmoid
z = torch.tensor([2.0])
sigmoid_prob = torch.sigmoid(z)
print(sigmoid_prob)  # tensor([0.8808])

# 2-class Softmax
z_softmax = torch.tensor([2.0, 0.0])
softmax_probs = F.softmax(z_softmax, dim=0)
print(softmax_probs[0])  # tensor(0.8808) - 같은 값!
```

Softmax는 sigmoid의 일반화입니다.

---

## 다중 레이블 분류 (Multi-label Classification)

하나의 샘플이 여러 클래스에 속할 수 있는 경우는 softmax를 사용하지 않습니다.

**예시:** 이미지에 "고양이"와 "개"가 모두 있음

**잘못된 방법 (Softmax):**

```python
# Softmax는 합이 1이어야 함
# 고양이와 개 둘 다 높은 확률을 가질 수 없음
logits = torch.tensor([3.0, 2.5, 0.1])  # [고양이, 개, 새]
probs = F.softmax(logits, dim=0)
print(probs)  # tensor([0.5760, 0.4145, 0.0095])
```

**올바른 방법 (Sigmoid):**

```python
# 각 클래스를 독립적으로 판단
logits = torch.tensor([3.0, 2.5, 0.1])
probs = torch.sigmoid(logits)
print(probs)  # tensor([0.9526, 0.9241, 0.5250])

# 고양이와 개 둘 다 높은 확률 가능!
# 임계값으로 결정
threshold = 0.5
predictions = (probs > threshold).float()
print(predictions)  # tensor([1., 1., 1.])
```

**손실 함수도 다름:**

```python
# 다중 클래스: CrossEntropyLoss (softmax 포함)
criterion = nn.CrossEntropyLoss()

# 다중 레이블: BCEWithLogitsLoss (sigmoid 포함)
criterion = nn.BCEWithLogitsLoss()
```

---

## 핵심 요약

- Softmax는 로짓을 확률 분포로 변환
- 모든 출력의 합이 1
- 지수 함수로 큰 값을 강조
- 다중 클래스 분류에 사용
- 수치 안정성을 위해 PyTorch 함수 사용
- 학습 시에는 모델에 포함하지 않음 (CrossEntropyLoss가 처리)
- 추론 시에만 softmax 적용하여 확률 얻음

---

## 관련 개념

**선수 지식:**

- [[Machine Learning Basics]] - 머신러닝 기본 원리
- [[Loss Function]] - 손실 함수의 역할
- [[Forward Pass]] - 순전파

**관련 개념:**

- [[LogSoftmax]] - Softmax의 로그
- [[NegativeLogLikelihood]] - 분류 손실 함수
- [[Sigmoid]] - 이진 분류 활성화 함수

**다음 단계:**

- [[Neural Network Components]] - 신경망 구성 요소
- Cross-Entropy Loss - 분류 문제의 표준 손실 함수