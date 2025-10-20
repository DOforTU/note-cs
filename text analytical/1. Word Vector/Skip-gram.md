## Contents

- 1. Skip-gram 개요
- 2. Skip-gram 구현
	- 2.1 환경 설정 및 데이터 로드
	- 2.2 Word2Vec 목적 함수 ([[Negative Sampling]])
	- 2.3 학습 데이터 전처리
	- 2.4 데이터셋 및 모델 구현
	- 2.5 학습 루프

---

## 1. Skip-gram 개요

Skip-gram은 Word2vec 프레임워크의 두 가지 주요 모델 중 하나입니다. Word2vec은 2013년 Mikolov 등이 제안한 단어 벡터 학습 방법으로, 대규모 텍스트 코퍼스에서 단어의 의미를 밀집 벡터로 표현하는 것을 목표로 합니다.

Skip-gram 모델의 핵심 아이디어는 **중심 단어(center word)가 주어졌을 때 주변 문맥 단어들(context words)을 예측**하는 것입니다. 예를 들어, "problems turning into banking crises"라는 문장에서 중심 단어가 "into"라면, Skip-gram은 이 단어로부터 주변의 "problems", "turning", "banking", "crises"를 예측하도록 학습됩니다.

이 방식은 분포 의미론(Distributional Semantics)에 기반합니다. "단어의 의미는 그 단어와 함께 자주 등장하는 주변 단어들에 의해 결정된다"는 개념으로, 언어학자 J.R. Firth의 유명한 문구 "You shall know a word by the company it keeps"로 요약됩니다. 비슷한 문맥에서 나타나는 단어들은 비슷한 의미를 가진다는 것입니다.

Skip-gram은 각 중심 단어에 대해 여러 개의 문맥 단어를 독립적으로 예측하므로, 상대적으로 더 많은 학습 샘플을 생성합니다. 이러한 특성 덕분에 희소한 단어들에 대해서도 좋은 표현을 학습하는 데 유리하며, 실무에서 더 널리 사용되는 모델입니다.

Word2vec의 또 다른 모델인 CBOW(Continuous Bag of Words)는 정반대로 작동합니다. 주변 문맥 단어들이 주어졌을 때 중심 단어를 예측하는 방식으로, 일반적으로 학습 속도가 빠르고 빈번하게 등장하는 단어들에 대해 좋은 성능을 보입니다.

### 1.1 [[Negative Sampling]]의 필요성

Skip-gram 모델의 확률 계산에는 기본적으로 Softmax 함수가 사용됩니다. 중심 단어 c가 주어졌을 때 문맥 단어 o가 나타날 확률은 다음과 같이 정의됩니다:

$$P(o|c) = \frac{\exp(u_o^T v_c)}{\sum_{w\in V} \exp(u_w^T v_c)}$$

여기서 문제는 분모의 정규화 항(normalization term)입니다. 전체 어휘 V에 대한 합을 계산해야 하는데, 어휘 크기가 수만에서 수십만에 이르면 계산 비용이 매우 높아집니다. 이는 "A big sum over words"라는 병목 지점을 만들며, 실제 대규모 코퍼스에서는 실용적이지 않습니다.

Negative Sampling은 이 문제를 해결하기 위해 고안된 방법입니다. 다중 클래스 분류 문제를 **이진 분류 문제**로 변환하는 것이 핵심입니다. 전체 어휘 중에서 정답 단어를 찾는 대신, 주어진 단어 쌍이 "실제로 함께 등장하는가?" 또는 "무작위로 선택된 것인가?"를 구별하도록 학습합니다.

실제 문맥 단어 쌍(true pair)과 무작위로 선택된 노이즈 쌍(noise pair)을 구별하는 이진 로지스틱 회귀를 학습하며, 각 실제 쌍에 대해 k개의 부정 샘플(negative samples)을 생성합니다. 이렇게 하면 각 업데이트마다 전체 어휘가 아닌 k+1개(실제 단어 1개 + 부정 샘플 k개)의 단어 벡터만 계산하면 되므로, 계산이 매우 효율적입니다. 일반적으로 k=5~20 정도면 충분히 좋은 결과를 얻을 수 있습니다.

---
## 2. Skip-gram 구현

### 2.1 설치 및 데이터 로드

Skip-gram 모델을 구현하기 위해서는 먼저 필요한 패키지를 설치해야 합니다. 이번 구현에서는 Hugging Face의 `datasets` 라이브러리를 사용하여 학습 데이터를 로드합니다.

```python
pip install datasets
```

데이터셋으로는 Stanford의 IMDB(Large Movie Review Dataset)를 사용합니다. 이 데이터셋은 영화 리뷰 텍스트로 구성되어 있으며, 다음과 같이 간단하게 로드할 수 있습니다.

```python
from datasets import load_dataset
imdb_dataset = load_dataset("stanfordnlp/imdb")
```

이렇게 로드된 데이터셋은 train, test 분할이 이미 되어 있으며, 각 샘플은 영화 리뷰 텍스트를 포함하고 있습니다. Skip-gram 학습을 위해서는 이러한 텍스트 문서들의 집합(corpus)이 필요합니다.

### 2.2 Word2Vec 목적 함수 ([[Negative Sampling]])

### 2.2.1 손실 함수

[[Negative Sampling]]을 사용한 Skip-gram의 목적은 전체 코퍼스에서 실제로 함께 등장하는 단어 쌍들의 확률을 최대화하는 것입니다. 이는 수학적으로 **손실 함수(loss function)를 최소화하는 문제**로 변환됩니다.

확률을 최대화하는 것과 음의 로그 확률을 최소화하는 것은 수학적으로 동치이며($\max P \equiv \min (-\log P)$), 실제 구현에서는 경사 하강법(Gradient Descent)을 사용한 최소화 방식으로 진행됩니다.

전체 코퍼스에 대한 손실 함수는 다음과 같이 정의됩니다:

$$J(\theta) = \frac{1}{T}\sum_{t=1}^{T} \sum_{\substack{-m\leq j\leq m \ j\neq 0}} J_{neg-sample}(\boldsymbol{u}_{t+j}, \boldsymbol{v}_t, U)$$

여기서 T는 전체 텍스트의 길이, m은 윈도우 크기를 의미합니다. 각 위치 t에서 중심 단어와 그 주변 문맥 단어들에 대해 손실을 계산하고, 이를 모두 평균내어 전체 목적 함수를 구성합니다. **학습의 목표는 이 $J(\theta)$를 최소화하는 것**이며, 손실이 작아질수록 모델은 실제 단어 쌍과 무작위 단어 쌍을 더 잘 구별하게 됩니다.

핵심이 되는 각 윈도우에 대한 손실 $J_{neg-sample}$은 다음과 같이 정의됩니다:

$$J_{neg-sample}(\boldsymbol{u}_o, \boldsymbol{v}_c, U) = -\log\sigma(\boldsymbol{u}_o^T\boldsymbol{v}_c) - \sum_{k\in{K \text{ sampled indices}}} \log\sigma(-\boldsymbol{u}_k^T\boldsymbol{v}_c)$$

이 식에서 $\sigma$는 시그모이드 함수(sigmoid function)로, 다음과 같이 정의됩니다:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

시그모이드 함수는 임의의 실수 값을 0과 1 사이의 값으로 매핑하여 이진 확률로 해석할 수 있게 합니다. 시그모이드의 출력은 항상 1 이하이므로($\sigma(x) \leq 1$), $\log\sigma(x)$는 항상 0 이하의 값을 가집니다. 따라서 손실이 항상 양수 또는 0이 되도록 앞에 음수 부호(-)를 붙입니다. **이 손실 함수의 최솟값은 0입니다.**

손실 함수는 두 개의 항으로 구성됩니다.

**첫 번째 항** $-\log\sigma(\boldsymbol{u}_o^T\boldsymbol{v}_c)$는 실제 문맥 단어 o에 대한 손실입니다. 이 항을 0에 가깝게 만들려면, $\log$ 안의 시그모이드 값이 1에 가까워야 하고, 이는 내적 $\boldsymbol{u}_o^T\boldsymbol{v}_c$가 커야 함을 의미합니다. 따라서 실제로 함께 등장하는 단어 쌍의 벡터들이 가까워지도록 학습됩니다.

**두 번째 항** $-\sum_{k} \log\sigma(-\boldsymbol{u}_k^T\boldsymbol{v}_c)$는 k개의 부정 샘플에 대한 손실입니다. 시그모이드 안에 음수 부호가 있다는 점에 주목해야 합니다. 이 항을 0에 가깝게 만들려면, $\sigma(-\boldsymbol{u}_k^T\boldsymbol{v}_c)$가 1에 가까워야 하고, 이는 $-\boldsymbol{u}_k^T\boldsymbol{v}_c$가 커야 함을, 즉 원래 내적 $\boldsymbol{u}_k^T\boldsymbol{v}_c$가 작아야(음수이거나 0에 가까워야) 함을 의미합니다. 따라서 무작위로 선택된 단어 쌍의 벡터들이 멀어지도록 학습됩니다.

결과적으로 이 손실 함수를 최소화하면, 문맥상 함께 나타나는 단어 쌍(true pair)의 벡터는 가까워지고(내적이 큼), 무작위로 선택된 단어 쌍(noise pair)의 벡터는 멀어지게(내적이 작음) 됩니다.
### 2.2.2 Skip-gram: Unigram

부정 샘플을 선택하는 방법도 중요합니다. 단순히 균등 분포를 사용하면 빈번한 단어들(the, a, is 등)이 너무 자주 선택되고 희소한 단어는 거의 선택되지 않습니다. Word2vec에서는 **유니그램 분포의 3/4 제곱**을 사용합니다.

$$P(w) = \frac{U(w)^{3/4}}{Z}$$

여기서 $U(w)$는 단어 w의 출현 빈도이고, Z는 정규화 상수입니다. 3/4승을 취하면 빈도가 낮은 단어의 샘플링 확률이 상대적으로 증가합니다. 예를 들어, "the"가 "democracy"보다 100배 더 자주 등장한다면, 원래 비율은 100:1이지만 3/4승 후에는 약 31.6:1로 줄어듭니다. 이렇게 하면 희소한 단어들도 부정 샘플로 적절히 선택되어 더 나은 학습이 가능합니다.

매개변수 $\theta$는 모델이 학습해야 할 모든 변수를 포함합니다. Word2vec에서는 각 단어마다 두 개의 벡터를 유지합니다.

- **중심 단어 벡터(v)**: 단어가 중심에 위치할 때 사용되는 벡터
- **문맥 단어 벡터(u)**: 단어가 문맥에 위치할 때 사용되는 벡터

어휘 사전의 크기가 V이고 벡터의 차원이 d일 때, 전체 매개변수의 개수는 2dV입니다. 두 개의 벡터를 사용하는 이유는 최적화를 쉽게 하기 위함이며, 학습이 끝난 후에는 일반적으로 두 벡터를 평균내어 최종 단어 벡터로 사용합니다.

### 2.3 학습 데이터 구축 단계

Skip-gram 모델을 학습시키기 위해서는 원시 텍스트 데이터를 적절한 형태로 전처리해야 합니다. 문서 집합(corpus)에서 시작하여 모델이 학습할 수 있는 (중심 단어, 문맥 단어) 쌍을 생성하기까지 여러 단계를 거칩니다.

#### 2.3.1 문서 토큰화

첫 번째 단계는 문서를 단어 단위로 쪼개는 토큰화(tokenization)입니다. IMDB 데이터셋에서 텍스트를 읽어와 각 문서를 토큰의 리스트로 변환합니다.

```python
def read_corpus() -> List[str]:
    """Large Movie Review Dataset(IMDB)에서 텍스트를 읽어 토큰화된 문장 리스트 반환"""
    imdb_dataset = load_dataset("stanfordnlp/imdb")
    files: List[str] = imdb_dataset["train"]["text"][:NUM_SAMPLES]
    print(f"files[0][:200] = {files[0][:200]}")
    return [[w.lower() for w in f.split()] for f in files]
```

이 함수는 IMDB 데이터셋의 train 분할에서 NUM_SAMPLES만큼의 문서를 가져옵니다. 각 문서는 공백을 기준으로 단어를 분리하고(`.split()`), 모든 단어를 소문자로 변환합니다(`.lower()`). 이렇게 하면 "The"와 "the"를 같은 단어로 취급할 수 있습니다.

결과적으로 `imdb_corpus`는 이중 리스트 구조를 가집니다. 외부 리스트의 각 요소는 하나의 문서를 나타내고, 내부 리스트는 해당 문서의 토큰들을 담고 있습니다. 예를 들어, `imdb_corpus[0]`은 첫 번째 문서의 모든 토큰을 포함합니다.

```python
imdb_corpus = read_corpus()
print(f"#(documents): {len(imdb_corpus)}")
print("The number of tokens in the 1st document:", len(imdb_corpus[0]))

# 첫 번째 문서의 내용 중 처음 20개의 단어(토큰)
print(f"imdb_corpus[0][:20] = {imdb_corpus[0][:20]}")
```

이렇게 토큰화된 코퍼스는 다음 단계인 어휘 사전 구축의 입력이 됩니다.

#### 2.3.2 토큰 인덱싱

토큰화된 문서들에서 고유한 단어들을 수집하여 어휘 사전(vocabulary)을 만들고, 각 단어에 고유한 인덱스를 부여합니다. 신경망은 문자열을 직접 처리할 수 없으므로, 모든 단어를 정수 인덱스로 변환해야 합니다.

```python
def build_vocab(corpus: List[List[str]]):
    tokens: List[str] = [tok for doc in corpus for tok in doc]
    vocab = set(tokens) # 중복 제거
    
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    
    indexed_docs: List[List[int]] = 
	    [[word2idx[w] for w in doc] for doc in corpus]
	    
    return word2idx, idx2word, indexed_docs
```

이 함수는 세 가지 중요한 자료구조를 생성합니다.

**word2idx**: 단어를 인덱스로 매핑하는 딕셔너리입니다. 예를 들어, `word2idx["banking"]`은 "banking"이라는 단어의 고유 인덱스를 반환합니다. 이는 단어를 벡터로 변환할 때 사용됩니다.

**idx2word**: 인덱스를 단어로 역매핑하는 딕셔너리입니다. 학습된 모델의 결과를 해석할 때 필요합니다. 예를 들어, 모델이 인덱스 42를 출력했다면 `idx2word[42]`를 통해 실제 단어를 확인할 수 있습니다.

**indexed_docs**: 원본 코퍼스의 모든 단어를 인덱스로 변환한 결과입니다. 원래 `[["the", "movie", "is"], ["great", "film"]]` 형태였던 데이터가 `[[45, 123, 78], [234, 156]]`과 같은 정수 리스트로 변환됩니다.

```python
word2idx, idx2word, indexed_docs = build_vocab(imdb_corpus)
vocab_size = len(word2idx)
print(f"어휘 수: {vocab_size}")
```

어휘 크기(vocab_size)는 전체 고유 단어의 개수로, 이후 임베딩 레이어의 크기를 결정하는 데 사용됩니다.

#### 2.3.3 Center, Context 쌍 생성

Skip-gram 모델의 핵심은 (중심 단어, 문맥 단어) 쌍을 학습하는 것입니다. 인덱싱된 문서들로부터 이러한 쌍들을 추출해야 합니다.

```python
def build_skipgram_pairs_docs(indexed_docs: List[List[int]], window_size: int) -> List[Tuple[int, int]]:
    pairs = []
    for doc in indexed_docs:
        for i, center in enumerate(doc):
	        # 음수가 될 수도 있으니 max
            start = max(0, i - window_size) 
            # len보다 커질 수 있으니 min
            end = min(len(doc), i + window_size + 1)
            # end = min(len(doc) - 1, i + window_size)
            for j in range(start, end): # range(start, end + 1)
                if i != j:
                    context = doc[j]
                    pairs.append((center, context))
    return pairs
```

이 함수는 각 문서를 순회하면서 슬라이딩 윈도우 방식으로 쌍을 생성합니다. 윈도우 크기가 2라면, 중심 단어 기준으로 좌우 2개씩 총 4개의 문맥 단어를 고려합니다.

예를 들어, "problems turning into banking crises" 문장에서 "into"가 중심 단어라면:

- (into, problems)
- (into, turning)
- (into, banking)
- (into, crises)

이렇게 4개의 쌍이 생성됩니다. 윈도우가 문서의 경계를 넘어가지 않도록 `max`와 `min` 함수로 범위를 조정합니다. 또한 `i != j` 조건으로 중심 단어 자기 자신은 문맥에서 제외합니다. 위 예시에서는 (into, into)가 되지 않도록 하는 예외처리 입니다.

```python
pairs = build_skipgram_pairs_docs(indexed_docs, WINDOW_SIZE)
print(f"학습 쌍 수(center, context): {len(pairs)}")
```

생성된 쌍의 개수는 코퍼스 크기와 윈도우 크기에 비례합니다. 대규모 코퍼스에서는 수백만에서 수억 개의 쌍이 생성될 수 있으며, 이들이 모두 학습 데이터가 됩니다.

#### 2.3.4 유니그램 분포 계산

Negative Sampling을 위해서는 부정 샘플을 선택할 확률 분포가 필요합니다. 앞서 설명한 대로, 단어 빈도의 3/4 제곱에 비례하는 유니그램 분포를 사용합니다.

```python
def make_unigram_probs(indexed_docs: List[List[int]], vocab_size: int, power: float = 0.75) -> np.ndarray:
    """단어 인덱스 리스트에서 확률 분포 계산 (f^0.75 정규화)"""
    # 어휘 크기(vocab_size)만큼의 길이를 가진 0 배열을 생성
    freqs = np.zeros(vocab_size, dtype=np.float64)
    for doc in indexed_docs:
	    # 전체 코퍼스를 구성하는 각 문서(단어 인덱스 리스트)를 순회
        for idx in doc:
	        # 현재 문서 내의 모든 단어 인덱스(idx)를 순회
            freqs[idx] += 1
            # 해당 단어 인덱스(idx)가 나타날 때마다 
            # 빈도수 배열(freqs)의 해당 위치 값을 1 증가.
			# 이 루프가 끝나면 freqs 배열에는 
			# 코퍼스 전체의 단어 빈도수가 담김.
            
    probs = freqs ** power
    probs /= probs.sum()
    return probs
```

이 함수는 먼저 각 단어의 출현 빈도를 계산합니다. `freqs` 배열은 크기가 vocab_size인 벡터로, 각 위치는 해당 인덱스를 가진 단어의 출현 횟수를 나타냅니다. 모든 문서의 모든 토큰을 순회하면서 해당 인덱스의 카운트를 증가시킵니다.

빈도 계산이 끝나면 각 빈도에 `power`(기본값 0.75)를 거듭제곱합니다. 이렇게 하면 빈도가 높은 단어의 확률은 상대적으로 낮아지고, 빈도가 낮은 단어의 확률은 상대적으로 높아집니다. 마지막으로 전체 합으로 나누어 정규화하면, 모든 확률의 합이 1이 되는 유효한 확률 분포를 얻습니다.

```python
probs = make_unigram_probs(indexed_docs, vocab_size)
```

이렇게 계산된 `probs`는 나중에 PyTorch의 Dataset 클래스에서 `np.random.choice`를 통해 부정 샘플을 샘플링할 때 사용됩니다. 이 분포를 사용함으로써 희소한 단어들도 적절히 부정 샘플로 선택되어 균형 잡힌 학습이 가능합니다.

### 2.4 데이터 로더 및 모델

학습 데이터가 준비되었다면, 이제 PyTorch의 Dataset과 Model을 구현해야 합니다. Dataset 클래스는 데이터를 효율적으로 배치 단위로 제공하고, Model 클래스는 실제 Skip-gram의 forward pass를 정의합니다.

#### 2.4.1 SkipGramDataset 클래스

PyTorch의 `Dataset` 클래스를 상속받아 Skip-gram에 특화된 데이터셋을 구현합니다. 이 클래스는 각 학습 샘플에 대해 중심 단어, 실제 문맥 단어(positive sample), 그리고 여러 개의 부정 샘플(negative samples)을 함께 반환합니다.

```python
class SkipGramDataset(Dataset):
    def __init__(self, pairs: List[Tuple[int, int]], probs: np.ndarray, vocab_size: int, k: int):
        self.pairs = pairs
        self.probs = probs
        self.vocab_size = vocab_size
        self.k = k
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        center, pos = self.pairs[idx]
        negs = np.random.choice(self.vocab_size, size=self.k, replace=True, p=self.probs)
        return (
            torch.tensor(center),
            torch.tensor(pos),
            torch.tensor(negs),
        )
```

생성자 `__init__`는 네 가지 매개변수를 받습니다. `pairs`는 앞서 생성한 (중심 단어, 문맥 단어) 쌍의 리스트이고, `probs`는 유니그램 분포입니다. `vocab_size`는 전체 어휘 크기, `k`는 각 positive sample당 생성할 negative sample의 개수입니다.

`__len__` 메서드는 전체 데이터셋의 크기를 반환합니다. 이는 학습 쌍의 개수와 동일합니다.

`__getitem__` 메서드는 특정 인덱스의 데이터를 반환하는 핵심 함수입니다. 먼저 `self.pairs[idx]`에서 중심 단어와 positive 문맥 단어를 가져옵니다. 그다음 `np.random.choice`를 사용하여 k개의 부정 샘플을 샘플링합니다. 여기서 중요한 것은 `p=self.probs` 매개변수로, 이를 통해 유니그램 분포에 따라 샘플링이 이루어집니다. `replace=True`는 중복을 허용한다는 의미로, 같은 단어가 여러 번 선택될 수 있습니다.

최종적으로 세 개의 텐서를 튜플로 묶어 반환합니다: 중심 단어 인덱스, positive 단어 인덱스, 그리고 k개의 negative 단어 인덱스들입니다.

```python
dataset = SkipGramDataset(pairs, probs, vocab_size, NEGATIVE_SAMPLES)
print(f"The first example: {dataset[0]}")
```

데이터셋의 첫 번째 샘플을 출력하면, 예를 들어 `(tensor(42), tensor(123), tensor([567, 89, 234, 456, 12]))`와 같은 형태가 나옵니다. 이는 인덱스 42번 단어가 중심 단어이고, 123번 단어가 실제 문맥 단어이며, 5개의 부정 샘플이 무작위로 선택되었음을 의미합니다.

#### 2.4.2 SkipGramNS 모델 (nn.Module)

Skip-gram 모델 자체는 `nn.Module`을 상속받아 구현합니다. 이 모델은 두 개의 임베딩 레이어를 가지며, 하나는 중심 단어용, 다른 하나는 문맥 단어용입니다.

```python
class SkipGramNS(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.uniform_(self.in_embed.weight, a=-0.5/embed_dim, b=0.5/embed_dim)
        nn.init.uniform_(self.out_embed.weight, a=-0.5/embed_dim, b=0.5/embed_dim)
```

생성자에서는 두 개의 임베딩 레이어를 초기화합니다. `self.in_embed`는 중심 단어가 입력으로 들어올 때 사용되는 임베딩으로, 최종적으로 우리가 얻고자 하는 단어 벡터입니다. `self.out_embed`는 문맥 단어(positive 또는 negative)를 처리할 때 사용되는 임베딩입니다.

임베딩 가중치의 초기화도 중요합니다. `nn.init.uniform_`을 사용하여 각 임베딩을 $[-0.5/d, 0.5/d]$ 범위의 균등 분포로 초기화합니다. 여기서 $d$는 `embed_dim`입니다. 이러한 작은 값으로 초기화하면 학습 초기에 안정적인 경사를 얻을 수 있습니다.

```python
def forward(self, center: torch.Tensor, pos: torch.Tensor, negs: torch.Tensor):
    center_embed = self.in_embed(center)  # (batch, embed_dim)
    pos_embed = self.out_embed(pos)       # (batch, embed_dim)
    negs_embed = self.out_embed(negs)     # (batch, k, embed_dim)
    
    pos_score = torch.sum(center_embed * pos_embed, dim=1)  # (batch,)
    # Batch Matrix Multiplication
    # 부정 샘플(negs)에 대한 점수(Score) 계산:
	# 중심 단어 임베딩(center_embed)과 배치 내 모든 부정 샘플 임베딩(negs_embed) 간의
	# 내적(유사도)을 배치 행렬 곱셈(BMM)을 사용하여 효율적으로 계산
    negs_score = torch.bmm(
	    negs_embed, center_embed.unsqueeze(2)).squeeze(2)  # (batch, k)
    
    pos_loss = -torch.log(torch.sigmoid(pos_score))
    negs_loss = -torch.log(torch.sigmoid(-negs_score))
    
    loss = torch.mean(pos_loss + torch.sum(negs_loss, dim=1))
    return loss
```

`forward` 메서드는 모델의 순전파를 정의합니다. 입력으로 중심 단어 인덱스 `center`, positive 단어 인덱스 `pos`, 그리고 negative 단어 인덱스들 `negs`를 받습니다.

먼저 각 인덱스를 임베딩 벡터로 변환합니다. `center_embed`는 중심 단어의 벡터 표현이고, `pos_embed`는 positive 문맥 단어의 벡터 표현입니다. `negs_embed`는 k개의 negative 샘플들의 벡터 표현으로, 3차원 텐서가 됩니다.

다음으로 스코어를 계산합니다. `pos_score`는 중심 단어와 positive 단어의 내적입니다. 요소별 곱셈(`*`)을 수행한 후 차원 1을 따라 합하면 내적이 됩니다. `negs_score`는 중심 단어와 각 negative 단어의 내적들입니다. `torch.bmm`(batch matrix multiplication)을 사용하여 배치 단위로 행렬 곱셈을 수행합니다.

손실 계산은 목적 함수의 정의를 그대로 따릅니다. `pos_loss`는 $-\log\sigma(u_o^T v_c)$에 해당하며, positive 단어의 확률을 최대화하도록 만듭니다. `negs_loss`는 $-\log\sigma(-u_k^T v_c)$에 해당하며, negative 단어들의 확률을 최소화하도록 만듭니다. 시그모이드 함수 안의 음수 부호에 주의해야 합니다.

최종 손실은 positive 손실과 모든 negative 손실의 합을 배치 전체에 대해 평균낸 값입니다. 이 손실을 최소화하는 방향으로 경사 하강법을 적용하면, 중심 단어와 실제 문맥 단어의 벡터는 가까워지고, 중심 단어와 무작위 단어의 벡터는 멀어지게 됩니다.

이렇게 구현된 모델은 표준적인 PyTorch 학습 루프에서 사용할 수 있습니다. DataLoader로 배치를 생성하고, optimizer로 매개변수를 업데이트하면서 여러 에폭 동안 학습을 진행합니다.

### 2.5 학습 루프

모든 준비가 완료되었다면 실제로 모델을 학습시키는 단계입니다. PyTorch의 표준적인 학습 루프를 사용하여 여러 에폭 동안 데이터를 반복적으로 학습합니다.

```python
def train():
    # 하이퍼파라미터 설정
    EMBED_DIM = 100
    BATCH_SIZE = 512
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5
    
    # 모델 초기화
    model = SkipGramNS(vocab_size, EMBED_DIM)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # DataLoader 생성
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2
    )
    
    # Optimizer 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 학습 루프
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch_idx, (center, pos, negs) in enumerate(dataloader):
            # 데이터를 device로 이동
            center = center.to(device)
            pos = pos.to(device)
            negs = negs.to(device)
            
            # Forward pass
            loss = model(center, pos, negs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 주기적으로 손실 출력
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                      f'Batch [{batch_idx+1}/{len(dataloader)}], '
                      f'Loss: {avg_loss:.4f}')
        
        # 에폭별 평균 손실
        avg_epoch_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] completed. '
              f'Average Loss: {avg_epoch_loss:.4f}\n')
    
    return model
```

학습 함수는 크게 세 부분으로 구성됩니다.

**초기화 단계**에서는 모델과 학습에 필요한 구성 요소들을 준비합니다. `EMBED_DIM`은 단어 벡터의 차원을 결정하며, 일반적으로 50에서 300 사이의 값을 사용합니다. `BATCH_SIZE`는 한 번에 처리할 샘플의 개수로, GPU 메모리에 따라 조정합니다. `LEARNING_RATE`는 경사 하강법의 스텝 크기를 결정하며, Adam optimizer의 경우 0.001 정도가 적절합니다.

모델을 생성한 후에는 사용 가능한 디바이스(GPU 또는 CPU)로 이동시킵니다. GPU가 있다면 `cuda`를, 없다면 `cpu`를 사용합니다. DataLoader는 데이터셋을 배치 단위로 나누고, `shuffle=True`로 설정하여 매 에폭마다 데이터 순서를 섞습니다. 이는 학습의 일반화 성능을 높이는 데 도움이 됩니다.

Optimizer로는 Adam을 사용합니다. Adam은 각 매개변수에 대해 적응적 학습률을 사용하는 최적화 알고리즘으로, Word2vec과 같은 임베딩 학습에 효과적입니다.

**학습 루프**는 이중 반복문으로 구성됩니다. 외부 루프는 에폭을 반복하고, 내부 루프는 배치를 반복합니다. 각 배치에 대해 다음 과정을 수행합니다.

먼저 배치 데이터를 GPU로 이동시킵니다. 이는 `.to(device)` 메서드를 통해 이루어지며, 데이터가 GPU 메모리에 있어야 모델도 GPU에서 계산할 수 있습니다.

Forward pass에서는 모델에 데이터를 입력하여 손실을 계산합니다. 앞서 구현한 `forward` 메서드가 호출되며, Negative Sampling 손실이 반환됩니다.

Backward pass는 세 단계로 이루어집니다. 먼저 `optimizer.zero_grad()`로 이전 배치의 경사를 초기화합니다. PyTorch는 기본적으로 경사를 누적하므로, 매 배치마다 초기화하지 않으면 잘못된 경사가 계산됩니다. 다음으로 `loss.backward()`를 호출하여 역전파를 수행하고 모든 매개변수의 경사를 계산합니다. 마지막으로 `optimizer.step()`으로 계산된 경사를 사용하여 매개변수를 업데이트합니다.

100개 배치마다 현재까지의 평균 손실을 출력하여 학습 진행 상황을 모니터링합니다. 손실이 지속적으로 감소한다면 모델이 잘 학습되고 있다는 신호입니다.

각 에폭이 끝날 때마다 전체 평균 손실을 출력합니다. 일반적으로 에폭이 진행됨에 따라 손실이 감소하며, 특정 시점 이후로는 수렴하여 더 이상 크게 변하지 않습니다.

학습이 완료되면 모델을 반환합니다. 학습된 모델에서 단어 벡터를 추출하려면 다음과 같이 할 수 있습니다.

```python
# 학습 실행
trained_model = train()

# 최종 단어 벡터 추출 (두 임베딩의 평균)
final_embeddings = (
    trained_model.in_embed.weight.data + 
    trained_model.out_embed.weight.data
) / 2

# 특정 단어의 벡터 확인
word = "banking"
word_idx = word2idx[word]
word_vector = final_embeddings[word_idx]
print(f"Vector for '{word}': {word_vector}")
```

앞서 설명한 대로, Word2vec은 각 단어에 대해 두 개의 벡터(중심 단어 벡터와 문맥 단어 벡터)를 유지하므로, 최종 단어 벡터는 이 둘의 평균으로 계산하는 것이 일반적입니다. 이렇게 얻어진 벡터는 단어의 의미를 밀집 벡터 형태로 표현하며, 유사한 의미를 가진 단어들은 벡터 공간에서 가까이 위치하게 됩니다.