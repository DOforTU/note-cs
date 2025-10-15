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

### 2.2 Word2Vec 목적 함수 (Negative Sampling)

Negative Sampling을 사용한 Skip-gram의 목적 함수는 전체 코퍼스에서 실제로 함께 등장하는 단어 쌍들의 확률을 최대화하는 것을 목표로 합니다. 수학적으로는 다음과 같이 표현됩니다.

$$J(\theta) = \frac{1}{T}\sum_{t=1}^{T} \sum_{\substack{-m\leq j\leq m \ j\neq 0}} J_{neg-sample}(\boldsymbol{u}_t, \boldsymbol{v}_t, U)$$

여기서 T는 전체 텍스트의 길이, m은 윈도우 크기를 의미합니다. 각 위치 t에서 중심 단어와 그 주변 문맥 단어들에 대해 손실을 계산하고, 이를 모두 평균내어 전체 목적 함수를 구성합니다.

핵심이 되는 $J_{neg-sample}$은 다음과 같이 정의됩니다.

$$J_{neg-sample}(\boldsymbol{u}_o, \boldsymbol{v}_c, U) = -\log\sigma(\boldsymbol{u}_o^T\boldsymbol{v}_c) - \sum_{k\in{K \text{ sampled indices}}} \log\sigma(-\boldsymbol{u}_k^T\boldsymbol{v}_c)$$

이 식은 두 개의 항으로 구성됩니다. 첫 번째 항 $-\log\sigma(\boldsymbol{u}_o^T\boldsymbol{v}_c)$는 실제 문맥 단어 o에 대한 손실입니다. 여기서 $\sigma$는 시그모이드 함수(sigmoid function)로, 다음과 같이 정의됩니다.

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

시그모이드 함수는 임의의 실수 값을 0과 1 사이의 값으로 매핑하여 이진 확률로 해석할 수 있게 합니다. $\boldsymbol{u}_o^T\boldsymbol{v}_c$가 클수록, 즉 두 벡터가 유사할수록 $\sigma(\boldsymbol{u}_o^T\boldsymbol{v}_c)$는 1에 가까워지고 $-\log$ 값은 0에 가까워집니다. 따라서 실제로 함께 등장하는 단어 쌍의 벡터들이 유사해지도록 만듭니다.

두 번째 항 $-\sum_{k} \log\sigma(-\boldsymbol{u}_k^T\boldsymbol{v}_c)$는 k개의 부정 샘플에 대한 손실입니다. 음수 부호에 주목해야 합니다. $\boldsymbol{u}_k^T\boldsymbol{v}_c$가 작을수록, 즉 무관한 단어일수록 $-\boldsymbol{u}_k^T\boldsymbol{v}_c$는 커지고 $\sigma(-\boldsymbol{u}_k^T\boldsymbol{v}_c)$는 1에 가까워져 $-\log$ 값이 0에 가깝습니다. 이는 무작위로 선택된 단어 쌍의 벡터들이 서로 다르도록 만듭니다.

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
print(f"imdb_corpus[0][:20] = {imdb_corpus[0][:20]}")
```

이렇게 토큰화된 코퍼스는 다음 단계인 어휘 사전 구축의 입력이 됩니다.

#### 2.3.2 토큰 인덱싱

토큰화된 문서들에서 고유한 단어들을 수집하여 어휘 사전(vocabulary)을 만들고, 각 단어에 고유한 인덱스를 부여합니다. 신경망은 문자열을 직접 처리할 수 없으므로, 모든 단어를 정수 인덱스로 변환해야 합니다.

```python
def build_vocab(corpus: List[List[str]]):
    tokens: List[str] = [tok for doc in corpus for tok in doc]
    vocab = set(tokens)
    
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
            start = max(0, i - window_size)
            end = min(len(doc), i + window_size + 1)
            for j in range(start, end):
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

이렇게 4개의 쌍이 생성됩니다. 윈도우가 문서의 경계를 넘어가지 않도록 `max`와 `min` 함수로 범위를 조정합니다. 또한 `i != j` 조건으로 중심 단어 자기 자신은 문맥에서 제외합니다.

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
    freqs = np.zeros(vocab_size, dtype=np.float64)
    for doc in indexed_docs:
        for idx in doc:
            freqs[idx] += 1
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

#### 2.4.1 SkipGramDataset 클래스

#### 2.4.2 SkipGramNS 모델 (nn.Module)