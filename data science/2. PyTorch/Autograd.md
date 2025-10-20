# Autograd

## 정의

**Autograd(Automatic Differentiation)** 는 PyTorch의 자동 미분 시스템입니다. 개발자가 직접 [[Gradient]]를 계산하는 코드를 작성할 필요 없이, [[Forward Pass]]에서 수행된 모든 연산을 자동으로 추적하여 [[Backpropagation]] 시 gradient를 자동으로 계산합니다.

Autograd는 **계산 그래프(computational graph)**를 동적으로 구성하고, [[Chain rule]]을 자동으로 적용하여 모든 파라미터의 gradient를 효율적으로 계산합니다.

---

## 왜 필요한가?

신경망을 직접 구현한다면 각 층의 gradient를 손으로 계산하고 코드로 작성해야 합니다.

**수동 gradient 계산의 문제:**

```python
# Forward pass
z1 = W1 @ x + b1
h1 = relu(z1)
z2 = W2 @ h1 + b2
h2 = relu(z2)
loss = ((h2 - y) ** 2).mean()

# Backward pass (직접 작성해야 함)
dL_dh2 = 2 * (h2 - y) / len(y)
dL_dz2 = dL_dh2 * (z2 > 0)  # ReLU gradient
dL_dW2 = dL_dz2.T @ h1
dL_db2 = dL_dz2.sum(axis=0)
dL_dh1 = dL_dz2 @ W2
dL_dz1 = dL_dh1 * (z1 > 0)  # ReLU gradient
dL_dW1 = dL_dz1.T @ x
dL_db1 = dL_dz1.sum(axis=0)
```

이 코드는:

- 오류가 발생하기 쉬움
- 모델 구조가 바뀌면 전체를 다시 작성해야 함
- 복잡한 신경망에서는 거의 불가능

**Autograd 사용:**

```python
# Forward pass
z1 = W1 @ x + b1
h1 = torch.relu(z1)
z2 = W2 @ h1 + b2
h2 = torch.relu(z2)
loss = ((h2 - y) ** 2).mean()

# Backward pass (자동!)
loss.backward()

# 모든 gradient가 자동으로 계산됨
print(W1.grad, W2.grad, b1.grad, b2.grad)
```

단 한 줄(`loss.backward()`)로 모든 gradient를 계산합니다. 모델 구조를 바꿔도 같은 코드가 작동합니다.

---

## 기본 사용법

### requires_grad 설정

Gradient를 추적하려면 텐서를 생성할 때 `requires_grad=True`로 설정합니다:

```python
import torch

# Gradient 추적 활성화
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
w = torch.tensor([0.5, 0.3, 0.2], requires_grad=True)

# Gradient 추적 없음
y = torch.tensor([1.0, 2.0, 3.0])  # requires_grad=False (기본값)
```

신경망의 파라미터는 항상 `requires_grad=True`여야 하지만, 입력 데이터는 보통 `False`입니다.

### Forward Pass와 계산 그래프

연산을 수행하면 Autograd가 자동으로 계산 그래프를 구성합니다:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = 2 * y + 3

print(z)  # tensor([11.], grad_fn=<AddBackward0>)
```

`grad_fn=<AddBackward0>`는 이 텐서가 어떤 연산으로 만들어졌는지 기록합니다. 이 정보로 backward pass에서 gradient를 계산합니다.

### Backward Pass

`backward()`를 호출하면 자동으로 모든 gradient가 계산됩니다:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2      # y = x²
z = 2 * y + 3   # z = 2x² + 3

z.backward()    # dz/dx 계산

print(x.grad)   # tensor([8.])
```

**수학적 검증:** $$z = 2x^2 + 3$$ $$\frac{dz}{dx} = 4x = 4 \times 2 = 8$$ 

---

## 계산 그래프

Autograd는 forward pass 중에 **동적 계산 그래프**를 구성합니다.

### 그래프 구조

각 텐서는 어떻게 생성되었는지 기록합니다:

```python
x = torch.tensor([3.0], requires_grad=True)
a = x + 2        # grad_fn=<AddBackward0>
b = a * 3        # grad_fn=<MulBackward0>
c = b ** 2       # grad_fn=<PowBackward0>
loss = c.mean()  # grad_fn=<MeanBackward0>
```

**계산 그래프:**

```
x(3) ──[+2]──> a(5) ──[×3]──> b(15) ──[²]──> c(225) ──[mean]──> loss(225)
```

각 화살표는 `grad_fn`에 저장되며, backward pass에서 역순으로 따라갑니다.

### 동적 그래프의 장점

PyTorch는 **동적 계산 그래프**를 사용합니다. 실행 시점에 그래프를 구성하므로:

**조건문 사용 가능:**

```python
x = torch.randn(1, requires_grad=True)

if x.item() > 0:
    y = x ** 2
else:
    y = x ** 3

y.backward()
```

실행할 때마다 다른 그래프가 만들어집니다. 조건에 따라 경로가 달라져도 올바른 gradient를 계산합니다.

**반복문 사용 가능:**

```python
x = torch.tensor([1.0], requires_grad=True)
y = x

for i in range(5):
    y = y * 2

y.backward()
print(x.grad)  # tensor([32.]) = 2^5
```

반복 횟수가 가변적이어도 작동합니다.

---

## Gradient 누적

PyTorch에서 `.backward()`를 호출하면 gradient가 **누적**됩니다:

```python
x = torch.tensor([1.0], requires_grad=True)

# 첫 번째 계산
y1 = x ** 2
y1.backward()
print(x.grad)  # tensor([2.])

# 두 번째 계산 (초기화 안 함)
y2 = x ** 3
y2.backward()
print(x.grad)  # tensor([5.]) = 2 + 3
```

### 왜 누적되는가?

여러 손실 함수를 동시에 최적화하는 경우가 있기 때문입니다:

```python
# Multi-task learning
loss1 = criterion1(output1, target1)
loss2 = criterion2(output2, target2)

loss1.backward(retain_graph=True)  # 첫 번째 loss의 gradient
loss2.backward()                     # 두 번째 loss의 gradient 누적

# 총 gradient = gradient1 + gradient2
optimizer.step()
```

### Gradient 초기화

일반적인 학습에서는 매 iteration마다 초기화해야 합니다:

```python
# 방법 1: 각 파라미터마다
x.grad.zero_()

# 방법 2: None으로 설정
x.grad = None

# 방법 3: Optimizer 사용 (권장)
optimizer.zero_grad()
```

**올바른 학습 루프:**

```python
for epoch in range(num_epochs):
    for x, y in dataloader:
        # Forward pass
        output = model(x)
        loss = criterion(output, y)
        
        # Gradient 초기화 (중요!)
        optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # 파라미터 업데이트
        optimizer.step()
```

---

## Gradient 계산 제어

### torch.no_grad()

Gradient를 계산하지 않으려면 `torch.no_grad()` 컨텍스트를 사용합니다:

```python
x = torch.tensor([2.0], requires_grad=True)

with torch.no_grad():
    y = x ** 2
    z = y + 3

print(y.grad_fn)  # None
print(z.grad_fn)  # None
```

**사용 시나리오:**

**1. 추론(Inference) 시:**

```python
model.eval()  # 평가 모드

with torch.no_grad():
    for x, y in test_loader:
        output = model(x)
        # Gradient 계산 안 함 → 메모리 절약, 속도 향상
```

**2. 파라미터 업데이트 시:**

```python
with torch.no_grad():
    for param in model.parameters():
        param -= learning_rate * param.grad
```

In-place 연산을 해도 gradient 계산에 영향을 주지 않습니다.

### detach()

계산 그래프에서 텐서를 분리합니다:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y.detach()  # y의 값을 복사하지만 gradient 추적 중단

print(y.requires_grad)  # True
print(z.requires_grad)  # False
```

**사용 예:**

```python
# 일부 파라미터를 고정하고 싶을 때
with torch.no_grad():
    frozen_output = frozen_model(x)

output = model(frozen_output.detach())  # frozen_model은 gradient 안 받음
loss = criterion(output, target)
loss.backward()  # model만 업데이트
```

### retain_graph

기본적으로 `.backward()`를 호출하면 계산 그래프가 삭제됩니다. 같은 그래프에서 여러 번 backward를 하려면:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2

y.backward(retain_graph=True)  # 그래프 유지
print(x.grad)  # tensor([4.])

y.backward()  # 그래프 사용 가능
print(x.grad)  # tensor([8.]) = 4 + 4 (누적)
```

메모리를 더 사용하므로 필요한 경우에만 사용합니다.

---

## 고급 기능

### 고차 미분

Gradient를 다시 미분하여 2차 미분을 계산할 수 있습니다:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3  # y = x³

# 1차 미분: dy/dx = 3x²
grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
print(grad1)  # tensor([12.]) = 3 × 2²

# 2차 미분: d²y/dx² = 6x
grad2 = torch.autograd.grad(grad1, x)[0]
print(grad2)  # tensor([12.]) = 6 × 2
```

`create_graph=True`는 gradient 계산도 추적하여 다시 미분할 수 있게 합니다.

### 특정 텐서에 대한 Gradient

`torch.autograd.grad()`를 사용하면 특정 텐서에 대한 gradient만 계산합니다:

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
w = torch.tensor([3.0, 4.0], requires_grad=True)

y = (x * w).sum()

# x에 대한 gradient만 계산
grad_x = torch.autograd.grad(y, x)[0]
print(grad_x)  # tensor([3., 4.])

# w는 gradient가 계산되지 않음
print(w.grad)  # None
```

### Jacobian 계산

벡터 출력에 대한 Jacobian 행렬을 계산할 수 있습니다:

```python
x = torch.tensor([[1.0, 2.0]], requires_grad=True)
y = torch.tensor([[3.0, 4.0, 5.0]])

z = x @ y.T  # 출력이 벡터

# Jacobian 계산
jacobian = torch.autograd.functional.jacobian(
    lambda x: x @ y.T, x
)
print(jacobian)
```

복잡한 수학적 분석이나 최적화 알고리즘에 사용됩니다.

---

## 실전 예제

### 간단한 회귀 모델

```python
import torch

# 데이터
x = torch.linspace(-3, 3, 50).reshape(-1, 1)
y = 2 * x + 1 + torch.randn(x.size()) * 0.3

# 파라미터
w = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

learning_rate = 0.01

for epoch in range(100):
    # Forward pass
    y_pred = x @ w + b
    loss = ((y_pred - y) ** 2).mean()
    
    # Backward pass (Autograd 자동 계산)
    loss.backward()
    
    # 파라미터 업데이트
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    # Gradient 초기화
    w.grad.zero_()
    b.grad.zero_()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

print(f'Final w: {w.item():.4f}, b: {b.item():.4f}')
# w ≈ 2.0, b ≈ 1.0
```

Autograd 덕분에 forward pass만 작성하면 backward pass는 자동입니다.

### 커스텀 함수

`torch.autograd.Function`을 상속하여 커스텀 연산을 만들 수 있습니다:

```python
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# 사용
x = torch.randn(5, requires_grad=True)
y = MyReLU.apply(x)
y.sum().backward()
print(x.grad)
```

`forward`에서 순전파를, `backward`에서 역전파를 직접 구현합니다. `ctx`는 forward와 backward 간에 정보를 전달합니다.

---

## 내부 작동 원리

### 동적 그래프 구성

Forward pass 중에 각 연산은:

1. 입력 텐서들을 받음
2. 출력 텐서를 계산
3. 출력 텐서에 `grad_fn` 설정 (어떤 연산으로 만들어졌는지)
4. `grad_fn`에 입력 텐서들과 필요한 정보 저장

```python
x = torch.tensor([2.0], requires_grad=True)
y = x * 3

# y.grad_fn은 MulBackward를 가리킴
# MulBackward는 x와 상수 3을 기억
```

### Backward Pass 실행

`loss.backward()`를 호출하면:

1. 손실의 `grad_fn`부터 시작
2. 재귀적으로 이전 연산들을 방문
3. 각 연산의 `backward()` 메서드 호출
4. [[Chain rule]] 적용하여 gradient 계산
5. 입력 텐서의 `.grad`에 누적

**순서:**

```
Loss ──[grad_fn]──> 이전 연산 ──[grad_fn]──> ... ──[grad_fn]──> 파라미터
```

Topological sort로 올바른 순서를 보장합니다.

---

## 주의사항

### In-place 연산

`requires_grad=True`인 텐서에 in-place 연산을 하면 에러가 발생합니다:

```python
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2

x += 1  # RuntimeError!
```

Forward pass의 값이 변경되면 backward pass에서 올바른 gradient를 계산할 수 없기 때문입니다.

**해결 방법:**

```python
# Out-of-place 연산 사용
x = x + 1  # 새로운 텐서 생성

# 또는 torch.no_grad() 사용
with torch.no_grad():
    x += 1  # 안전
```

### 메모리 관리

계산 그래프는 모든 중간 결과를 저장하므로 메모리를 많이 사용합니다:

```python
# 메모리 많이 사용
for i in range(1000):
    y = model(x)
    loss = criterion(y, target)
    loss.backward()  # 1000개의 그래프가 메모리에 남음
```

**해결:**

```python
# 매번 그래프 초기화
for i in range(1000):
    optimizer.zero_grad()
    y = model(x)
    loss = criterion(y, target)
    loss.backward()
    optimizer.step()  # 그래프 자동 해제
```

### Leaf 텐서

직접 생성한 텐서를 "leaf 텐서"라고 합니다. Leaf 텐서의 gradient만 보존됩니다:

```python
x = torch.tensor([1.0], requires_grad=True)  # leaf
y = x ** 2  # non-leaf
z = y + 3   # non-leaf

z.backward()

print(x.grad)  # tensor([2.]) - 보존됨
print(y.grad)  # None - 중간 결과는 버려짐
```

중간 텐서의 gradient가 필요하면 `retain_grad()`를 호출합니다:

```python
y.retain_grad()
z.backward()
print(y.grad)  # 이제 접근 가능
```

---

## 다른 프레임워크와의 비교

### TensorFlow 1.x (정적 그래프)

```python
# TensorFlow 1.x
import tensorflow as tf

x = tf.placeholder(tf.float32)
y = x ** 2
grad = tf.gradients(y, x)

with tf.Session() as sess:
    result = sess.run(grad, feed_dict={x: 2.0})
```

그래프를 먼저 정의하고 나중에 실행합니다. 디버깅이 어렵고 동적 구조를 만들기 힘듭니다.

### PyTorch (동적 그래프)

```python
# PyTorch
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)
```

실행 시점에 그래프를 구성합니다. 디버깅이 쉽고 Python의 모든 기능을 활용할 수 있습니다.

### JAX

JAX도 자동 미분을 제공하지만 함수형 프로그래밍 스타일입니다:

```python
import jax
import jax.numpy as jnp

def f(x):
    return x ** 2

grad_f = jax.grad(f)
print(grad_f(2.0))  # 4.0
```

PyTorch보다 더 함수형이고 컴파일 최적화가 강력하지만, 학습 곡선이 가파릅니다.

---

## 핵심 요약

- Autograd는 PyTorch의 자동 미분 시스템
- Forward pass 중에 동적으로 계산 그래프 구성
- `loss.backward()`로 모든 gradient 자동 계산
- [[Chain rule]]을 자동으로 적용
- Gradient는 누적되므로 매 iteration마다 초기화 필요
- `torch.no_grad()`로 gradient 계산 비활성화
- 동적 그래프로 조건문, 반복문 자유롭게 사용 가능

---

## 관련 개념

**선수 지식:**

- [[Machine Learning Basics]] - 머신러닝 기본 원리
- [[Gradient]] - Gradient의 의미
- [[Forward Pass]] - 순전파
- [[Backpropagation]] - 역전파
- [[Chain rule]] - 연쇄 법칙

**관련 개념:**

- [[Deep Learning Core Concepts]] - 딥러닝 전체 구조

**다음 단계:**

- [[Optimization Algorithms]] - Gradient를 사용한 최적화
- PyTorch 실전 사용법