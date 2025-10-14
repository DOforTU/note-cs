## 1. 벡터 및 행렬 조작 패키지로서의 NumPy

NumPy는 근본적으로 **벡터(Vector)와 행렬(Matrix) 조작**을 위한 패키지입니다. 데이터를 효율적으로 저장하고 처리할 수 있는 기본 배열 구조를 제공하며, 이를 통해 복잡한 수학적 계산을 간결하게 수행할 수 있게 합니다.

---
## 2. NumPy의 주요 특징

NumPy를 사용하면 일반 파이썬 리스트를 사용하는 것보다 훨씬 빠르고 간결하게 작업을 수행할 수 있습니다. 특히 대규모 과학 계산 분야에서 시간을 절약하고 코드의 양을 줄이는 데 크게 기여합니다.

| 특징           | 설명                                                        |
| :----------- | :-------------------------------------------------------- |
| **효율성**      | Broadcasting과 Vectorization 기능을 통해 시간과 코드의 양을 절약할 수 있습니다. |
| **자동 타입 유추** | 배열을 선언할 때 입력된 데이터의 타입을 자동으로 유추하여 배열을 생성합니다.               |
| **구조화**      | 기본 `array` 선언부터 시작하여 다양한 형태의 행렬(Matrix) 생성을 지원합니다.        |
| **다양한 연산**   | **다양한 벡터 연산**이 가능하도록 설계되어 있습니다.                           |

```python
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
b = np.zeros((2, 2)) # 요소가 전부 0인 2 x 2 매트릭스
c = np.random.random((2, 2)) # 0~1 사이 랜덤 값을 가진 2 x 2 매트릭스
```

---
## 3. 핵심 인덱싱 기법

데이터 배열에서 원하는 요소들을 정확히 가져오는 것은 분석의 기본입니다. NumPy는 데이터를 참조(reference)할 수 있도록 여러 가지 강력한 인덱싱 방법을 제공합니다.

### 3.1. Integer Indexing 및 Slice Indexing

일반적인 리스트에서 사용하듯이 정수 인덱싱이나 슬라이스 기법을 사용하여 요소에 접근할 수 있습니다. 다차원 배열에서는 각 차원에 대해 인덱싱을 적용합니다.

**코드 예시:**
```Python
import numpy as np
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# 2차원 배열에서 (1행, 2열) 요소 접근 (Integer Indexing)
element = a[1, 2]
print(f"a[1, 2]: {element}") # 출력: a[1, 2]: 7

# 0행부터 1행까지, 모든 열 선택 (Slice Indexing)
rows_slice = a[0:2, :]
print(f"a[0:2, :]:\n{rows_slice}") 
# 출력: a[0:2, :]:
# [[1 2 3 4]
#  [5 6 7 8]]

# 첫 번째 행을 슬라이스 없이 선택 (결과는 1차원)
row_int = a[1, :] 
print(f"a[1, :]: {row_int}, shape: {row_int.shape}") 
# 출력: a[1, :]: [5 6 7 8], shape: (4,)

# 첫 번째 행을 슬라이스로 선택 (차원 유지, 결과는 2차원)
row_slice = a[1:2, :] 
print(f"a[1:2, :]:\n{row_slice}, shape: {row_slice.shape}") 
# 출력: a[1:2, :]:
# [[5 6 7 8]], shape: (1, 4)
```

### 3.2. Integer Array를 사용한 인덱싱 (Fancy Indexing)

**정수 배열**을 사용하여 여러 위치에 있는 요소를 한 번에 선택적으로 가져올 수 있습니다. 행과 열 인덱스를 각각 정수 배열로 지정하여 원하는 좌표의 원소를 가져오거나 수정할 수 있습니다.

**코드 예시:**
```Python
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(f"Original a:\n{a}")

# 행 인덱스 배열과 열 인덱스 배열을 사용해 특정 요소 선택
row_indices = np.array([0, 2, 3])
col_indices = np.array([0, 1, 2])
# (0, 0), (2, 1), (3, 2) 위치의 원소 선택
selected_elements = a[row_indices, col_indices]
print(f"Selected elements: {selected_elements}") 
# 출력: Selected elements: [ 1  8 12]

# 선택된 요소 값 업데이트 (우측 이미지 예시 참고)
a[np.arange(4), [0, 2, 0, 1]] += 10
print(f"Updated a:\n{a}")
# (0,0), (1,2), (2,0), (3,1) 위치에 10이 더해짐
# 출력: 
# [[11  2  3]
#  [ 4  5 16]
#  [17  8  9]
#  [10 21 12]]
```

### 3.3. Boolean Array Indexing

**특정 조건**을 만족하는 원소(element)만 효율적으로 가져오기 위해 **불리언(Boolean) 배열**을 인덱스로 활용합니다. 조건문(`a > N`)의 결과를 인덱스로 사용하며, 결과는 **True**에 해당하는 모든 원소만 포함하는 **1차원 배열**입니다.

**코드 예시:**
```Python
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 조건: 배열 a의 원소 중 5보다 큰 것만 True로 하는 불리언 배열 생성
boolean_mask = (a > 5)
print(f"Boolean Mask:\n{boolean_mask}")
# 출력:
# [[False False False]
#  [False False  True]
#  [ True  True  True]]

# 불리언 마스크를 사용하여 조건 만족 원소만 선택
selected_elements = a[boolean_mask]
print(f"Elements > 5: {selected_elements}") 
# 출력: Elements > 5: [6 7 8 9]
```

---
## 4. 필수 기본 연산

NumPy는 행렬 기반의 수학적 연산을 매우 쉽게 수행할 수 있도록 합니다.
NumPy가 지원하는 주요 기본 연산은 다음과 같습니다:

### 4.1. Matrix 기본 연산: 사칙연산을 포함한 기본적인 행렬 연산

NumPy 배열 간의 사칙연산은 **요소별(Element-wise)**로 수행됩니다. 즉, 같은 위치에 있는 원소끼리 연산됩니다. 이 연산을 수행하려면 두 배열의 **모양(shape)이 같아야** 합니다.

**코드 예시:**
```Python
import numpy as np

x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

# 덧셈
print("덧셈 (x + y):\n", x + y) 
# 출력: [[ 6.  8.]
#       [10. 12.]]

# 뺄셈
print("뺄셈 (x - y):\n", x - y)
# 출력: [[-4. -4.]
#       [-4. -4.]]

# 곱셈 (요소별 곱셈)
print("곱셈 (x * y):\n", x * y)
# 출력: [[ 5. 12.]
#       [21. 32.]]

# 나눗셈
print("나눗셈 (x / y):\n", x / y)
# 출력: [[0.2        0.33333333]
#       [0.42857143 0.5       ]]
```

### 4.2. Dot product (내적): 두 행렬 또는 벡터 간의 내적 계산

**내적**은 벡터 간의 스칼라 곱 또는 행렬 간의 **행렬 곱셈(Matrix Multiplication)** 을 의미하며, `numpy.dot()` 함수나 `@` 연산자를 사용합니다. 행렬 곱셈을 위해서는 **첫 번째 행렬의 열 수와 두 번째 행렬의 행 수가 일치**해야 합니다.

**코드 예시:**
```Python
import numpy as np

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

# 내적 (행렬 곱셈) - np.dot() 사용
dot_product_1 = np.dot(x, y)
print("내적 (np.dot(x, y)):\n", dot_product_1)
# 출력: [[1*5+2*7, 1*6+2*8],   => [19, 22]
#       [3*5+4*7, 3*6+4*8]]  => [43, 50]

# 내적 (행렬 곱셈) - @ 연산자 사용
dot_product_2 = x @ y
print("내적 (x @ y):\n", dot_product_2)
# 출력: [[19 22]
#       [43 50]]
```

### 4.3. 합계 및 전치 (Sum and Transpose)

#### 4.3.1 합계 (Sum)

`np.sum()` 함수를 사용하여 배열의 모든 원소의 합을 구하거나, `axis` 인자를 지정하여 특정 축(행 또는 열)을 따라 합계를 계산할 수 있습니다.

**코드 예시:**
```Python
import numpy as np

x = np.array([[1, 2], [3, 4]])

# 전체 원소의 합계
total_sum = np.sum(x)
print(f"전체 합계: {total_sum}") # 출력: 전체 합계: 10

# 행 방향(axis=0) 합계: 각 열의 합계
sum_by_column = np.sum(x, axis=0)
print(f"열(axis=0) 합계: {sum_by_column}") # 출력: 열(axis=0) 합계: [4 6]

# 열 방향(axis=1) 합계: 각 행의 합계
sum_by_row = np.sum(x, axis=1)
print(f"행(axis=1) 합계: {sum_by_row}") # 출력: 행(axis=1) 합계: [3 7]
```

#### 4.3.2 전치 (Transpose)

`T` 속성이나 `np.transpose()` 함수를 사용하여 행렬의 **행과 열을 바꾼** 전치 행렬을 얻을 수 있습니다.

**코드 예시:**
```Python
import numpy as np

x = np.array([[1, 2], [3, 4]])
print("원래 행렬 x:\n", x)
# 출력: [[1 2]
#       [3 4]]

# 전치 (x.T 사용)
x_transposed = x.T
print("전치 행렬 (x.T):\n", x_transposed)
# 출력: [[1 3]
#       [2 4]]
```

---
## 5. Broadcasting 및 Vectorization

NumPy의 핵심 강점은 바로 **'효율성'** 을 극대화하는 브로드캐스팅과 벡터화 기법입니다.

### 5-1. 브로드캐스팅 (Broadcasting)

브로드캐스팅은 NumPy의 강력한 기능 중 하나로, **모양이 다른 배열(array) 간의 연산**을 수행할 때 사용됩니다.

- **작동 원리:** 특정 조건을 만족하는 경우에 브로드캐스팅이 작동하며, 이로 인해 연산이 **빠르고 효율적으로** 진행됩니다.
- **내부 구조:** 브로드캐스팅은 내부적으로 C 언어로 작동하여 속도 측면에서 큰 이점을 가집니다.
    - **개념 예시:** 모양이 다른 배열끼리 연산할 때, 작은 배열이 큰 배열의 모양에 맞게 **확장(Stretch)** 되어 연산이 가능해집니다. 이 확장은 메모리 복사 없이 가상으로 이루어지기 때문에 매우 빠릅니다.
        

**코드 예시:**
```Python
import numpy as np

# 스칼라 값과 배열 간의 연산
a = np.array([1.0, 2.0, 3.0])
b = 2.0
result_scalar = a * b # 스칼라 b가 a의 모양 (3,)에 맞게 확장되어 연산
print(f"스칼라 연산 결과: {result_scalar}") 
# 출력: 스칼라 연산 결과: [2. 4. 6.]

# 2차원 배열과 1차원 배열 간의 연산
A = np.array([[1, 2, 3], 
              [4, 5, 6]]) # 모양 (2, 3)
B = np.array([10, 20, 30]) # 모양 (3,)

# B가 A의 각 행에 맞춰 (1, 3) -> (2, 3)으로 확장되어 연산
result_array = A + B 
print(f"브로드캐스팅 연산 결과 (A + B):\n{result_array}")
# 출력:
# [[11 22 33]
#  [14 25 36]]
```

### 5-2. 벡터화 (Vectorization)

벡터화는 효율적인 코드 작성을 위한 핵심 방법론입니다.

- **목표:** 동일한 작업을 수행해야 할 경우, **반복문(for loop) 사용을 최소화**하고 되도록이면 **벡터 단위 연산**을 사용하는 것을 권장합니다.
    
- **중요성:** 벡터 단위를 활용하면 파이썬의 느린 반복 구조를 피하고 내부적으로 최적화된 NumPy 코어를 사용하게 되어 실행 속도가 **비약적으로 빨라집니다.** 실제로 큰 데이터 행렬에 대해 반복문(`for loop`)을 사용하는 것과 벡터화 방법을 사용하는 것의 실행 시간을 비교하는 연습이 중요합니다.
    

**코드 예시 (반복문 vs 벡터화):**
```Python
import numpy as np
import time

x = np.random.rand(1000000) # 100만 개의 요소를 가진 배열
y = np.random.rand(1000000)

# 1. 반복문을 사용한 덧셈 (비효율적)
start_time = time.time()
z_loop = np.zeros_like(x)
for i in range(len(x)):
    z_loop[i] = x[i] + y[i]
end_time = time.time()
print(f"반복문(for loop) 실행 시간: {end_time - start_time:.6f} 초")

# 2. 벡터화된 연산을 사용한 덧셈 (권장)
start_time = time.time()
z_vector = x + y # 벡터 단위 연산
end_time = time.time()
print(f"벡터화 연산 실행 시간: {end_time - start_time:.6f} 초") 
# 일반적으로 벡터화 연산이 반복문보다 훨씬 빠름
```