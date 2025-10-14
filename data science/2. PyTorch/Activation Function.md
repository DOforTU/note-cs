# Activation Function

## 정의

**활성화 함수(Activation Function)**는 신경망의 각 뉴런이 입력 신호를 받아 출력을 결정하는 비선형 함수입니다. 신경망에 비선형성을 부여하여 복잡한 패턴을 학습할 수 있게 만듭니다.

$$\text{output} = f(\text{input})$$

여기서 $f$가 활성화 함수입니다.

---

## 왜 필요한가?

### 선형 함수만으로는 부족하다

신경망이 선형 함수만 사용한다면:

첫 번째 층: $h = W_1x + b_1$ 두 번째 층: $y = W_2h + b_2 = W_2(W_1x + b_1) + b_2$

전개하면: $y = W_2W_1x + W_2b_1 + b_2 = W'x + b'$

결국 **하나의 선형 함수**가 됩니다. 아무리 층을 깊게 쌓아도 선형 변환의 조합은 선형입니다.

### 선형 함수의 한계

선형 함수는 직선(또는 평면)으로만 데이터를 구분할 수 있습니다. XOR 문제처럼 선형적으로 분리할 수 없는 문제는 풀 수 없습니다.

### 비선형 함수의 힘

활성화 함수로 비선형성을 추가하면:

$$h = f(W_1x + b_1)$$ $$y = W_2h + b_2$$

$f$가 비선형이므로 전체 함수도 비선형이 되어 복잡한 곡선이나 곡면으로 데이터를 구분할 수 있습니다.

---

## 생물학적 영감

활성화 함수는 실제 뉴런의 동작에서 영감을 받았습니다.

**실제 뉴런:**

1. 여러 입력 신호를 받음
2. 신호를 합산
3. 합이 임계값을 넘으면 활성화
4. 신호를 다음 뉴런으로 전달

**인공 뉴런:** $$\text{output} = f\left(\sum_{i} w_i x_i + b\right)$$

---

## 활성화 함수의 역할

### 1. 비선형성 추가

가장 중요한 역할입니다. 비선형 함수가 없으면 신경망은 단순한 선형 모델에 불과합니다.

### 2. 특징 추출

각 층의 활성화 함수는 점점 더 추상적인 특징을 추출합니다. 이미지 분류에서 1층은 에지, 2층은 질감, 3층은 부분, 4층은 전체를 인식합니다.

### 3. 표현력 증대

비선형 활성화 덕분에 신경망은 **보편 근사 정리(Universal Approximation Theorem)**를 만족합니다. 충분히 넓은 은닉층을 가진 신경망은 어떤 연속 함수도 임의의 정확도로 근사할 수 있습니다.

---

## 이상적인 활성화 함수의 조건

### 1. 비선형성 (Nonlinearity)

필수 조건입니다. 선형 함수는 활성화 함수로 사용할 수 없습니다.

### 2. 미분 가능성 (Differentiability)

[[Backpropagation]]을 위해 거의 모든 점에서 미분 가능해야 합니다.

### 3. 단조성 (Monotonicity)

단조 증가 또는 감소하면 최적화가 더 쉽습니다. 하지만 필수는 아닙니다.

### 4. Zero-Centered

출력이 0을 중심으로 분포하면 학습이 더 효율적입니다.

- Sigmoid: [0, 1] → Not zero-centered
- Tanh: [-1, 1] → Zero-centered

### 5. 계산 효율성

빠르게 계산할 수 있어야 합니다.

- ReLU: $\max(0, x)$ → 매우 빠름
- Sigmoid: $1/(1+e^{-x})$ → 지수 연산 필요

### 6. Gradient 특성

Gradient가 너무 작거나 크면 학습이 어렵습니다.

- **Gradient Vanishing**: Sigmoid/Tanh는 입력이 크거나 작을 때 gradient가 거의 0
- **Gradient Exploding**: 일부 활성화 함수는 gradient가 폭발 가능

---

## 주요 활성화 함수

### [[ReLU]] (Rectified Linear Unit)

$$\text{ReLU}(x) = \max(0, x)$$

**장점:**

- 계산이 매우 빠름
- Gradient vanishing 문제 완화
- 희소성(sparsity) 제공

**단점:**

- Dying ReLU: 음수 입력에 대해 gradient가 0
- 음수 영역에서 뉴런이 죽을 수 있음

### Leaky ReLU

$$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \ \alpha x & \text{if } x \leq 0 \end{cases}$$

일반적으로 $\alpha = 0.01$ 사용. Dying ReLU 문제를 해결합니다.

### Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**출력 범위:** $(0, 1)$

**장점:**

- 확률로 해석 가능
- 부드러운 비선형성

**단점:**

- Gradient vanishing (큰 양수/음수에서 gradient ≈ 0)
- 출력이 zero-centered 아님

**사용처:**

- 이진 분류 출력층
- LSTM 게이트

### Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

**출력 범위:** $(-1, 1)$

Sigmoid의 zero-centered 버전입니다. Sigmoid보다 선호되지만 여전히 gradient vanishing 문제가 있습니다.

### Softmax

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

모든 출력의 합이 1이 되어 확률 분포로 해석 가능합니다. 다중 클래스 분류의 출력층에만 사용합니다.

---

## 고급 활성화 함수

### ELU (Exponential Linear Unit)

$$\text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$$

Zero-centered에 가깝고 Dying ReLU를 방지하지만 지수 연산으로 느립니다.

### GELU (Gaussian Error Linear Unit)

$$\text{GELU}(x) = x \cdot \Phi(x)$$

여기서 $\Phi(x)$는 표준 정규분포의 누적 분포 함수입니다. Transformer 모델(BERT, GPT)에서 자주 사용됩니다.

### Swish / SiLU

$$\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

Google이 자동 탐색으로 발견했으며 일부 작업에서 ReLU보다 우수합니다.

### Mish

$$\text{Mish}(x) = x \cdot \tanh(\ln(1 + e^x))$$

Swish의 변형으로 모든 점에서 부드럽게 미분 가능하며 일부 작업에서 최고 성능을 보입니다.

---

## 활성화 함수 비교

|함수|수식|범위|장점|단점|
|---|---|---|---|---|
|**ReLU**|$\max(0,x)$|$[0, \infty)$|빠름, 간단|Dying ReLU|
|**Leaky ReLU**|$\max(\alpha x, x)$|$(-\infty, \infty)$|Dying ReLU 해결|하이퍼파라미터|
|**Sigmoid**|$1/(1+e^{-x})$|$(0, 1)$|확률 해석|Vanishing gradient|
|**Tanh**|$(e^x-e^{-x})/(e^x+e^{-x})$|$(-1, 1)$|Zero-centered|Vanishing gradient|
|**Softmax**|$e^{x_i}/\sum e^{x_j}$|$(0, 1)$, 합=1|확률 분포|출력층 전용|
|**ELU**|$x$ or $\alpha(e^x-1)$|$(-\alpha, \infty)$|Zero-centered|느림|
|**GELU**|$x \cdot \Phi(x)$|$(-\infty, \infty)$|Transformer에 적합|복잡함|
|**Swish**|$x \cdot \sigma(x)$|$(-\infty, \infty)$|성능 우수|계산 비용|

---

## 활성화 함수 선택 가이드

### 은닉층

- **기본**: ReLU
- **Dying ReLU 발생 시**: Leaky ReLU 또는 ELU
- **깊은 네트워크**: ReLU + Residual Connection

### 출력층

- **이진 분류**: Sigmoid
- **다중 클래스 분류**: Softmax (또는 CrossEntropyLoss에 포함)
- **회귀**: 활성화 함수 없음 (선형)

### 특수 용도

- **RNN/LSTM 게이트**: Sigmoid, Tanh
- **Transformer**: GELU
- **GAN 생성자**: Tanh (출력을 [-1, 1]로)

---

## Gradient 분석

### ReLU의 Gradient

$$f'(x) = \begin{cases} 1 & \text{if } x > 0 \ 0 & \text{if } x \leq 0 \end{cases}$$

Gradient가 0 또는 1로 계산이 빠르지만, 한번 0이 되면 영원히 0입니다.

### Sigmoid의 Gradient

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

최댓값이 $\sigma'(0) = 0.25$이고, 큰 양수/음수에서 거의 0이 됩니다.

### Vanishing Gradient 문제

깊은 신경망에서 Sigmoid/Tanh를 사용하면:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial a_n} \cdot \sigma'(z_n) \cdot \cdots \cdot \sigma'(z_1) \cdot x$$

각 $\sigma'(z_i) < 0.25$이므로 $0.25^{10} \approx 9.5 \times 10^{-7}$. 10층만 되어도 gradient가 거의 0이 됩니다.

---

## 핵심 요약

- 활성화 함수는 신경망에 비선형성을 부여
- 선형 함수만 사용하면 깊은 네트워크의 의미가 없음
- **ReLU**가 기본 선택 (빠르고 효과적)
- **Sigmoid**는 이진 분류 출력층
- **Softmax**는 다중 클래스 분류 출력층
- Gradient vanishing/exploding 주의
- 작업과 모델 구조에 따라 적절히 선택

---

## 관련 개념

**선수 지식:**

- [[Machine Learning Basics]] - 머신러닝 기본 원리
- [[Forward Pass]] - 순전파
- [[Backpropagation]] - 역전파

**핵심 활성화 함수:**

- [[ReLU]] - ReLU 상세
- [[Sigmoid]] - Sigmoid 상세

**관련 구성 요소:**

- [[Dropout]] - 정규화
- [[BatchNormalization]] - 정규화