## 1. 테이블 데이터 처리 및 활용 목적

Pandas는 스프레드시트나 데이터베이스처럼 행(Row)과 열(Column)로 구성된 테이블 형태의 데이터를 다루는 데 특화되어 있습니다.

**용도 (Statistical analysis, ML preprocessing 등)**

Pandas의 활용 범위는 매우 넓습니다. 데이터를 불러와 정제하는 **전처리(Preprocessing)** 과정에서 핵심적인 역할을 수행하며, 구체적으로 다음과 같은 작업에 사용됩니다:

1. **통계 분석 (Statistical analysis)**: 데이터의 기초 통계를 계산합니다.
2. **시각화 (Visualization)**: 데이터를 시각화하기 위한 준비 작업 및 간단한 시각화 기능을 제공합니다.
3. **머신러닝 전처리 (Machine learning preprocessing)**: 머신러닝 모델에 데이터를 입력하기 전에 형태를 맞추고 결측치를 처리하는 등의 준비 작업에 필수적입니다.
    
> **설치 참고:** Pandas를 사용하려면 일반적으로 `pip install pandas` 명령어를 통해 설치합니다.

---

## 2. Pandas의 두 가지 주요 구성 요소

Pandas를 이해하는 데 가장 중요한 핵심 구조는 **Series**와 **DataFrame**입니다. 이 두 가지 구성 요소가 테이블 형태의 데이터를 구성하는 기본 단위입니다.

|구성 요소|설명 (역할)|
|---|---|
|**Series**|데이터프레임의 **단일 열(a column)**을 의미합니다.|
|**DataFrame**|**Series들의 모음(a collection of Series)**으로, Pandas에서 테이블 전체를 나타내는 주요 객체입니다.|

**코드 예시:**
```Python
import pandas as pd

# Series 생성 예시 (1차원 배열 형태)
s = pd.Series([10, 20, 30, 40], name='Scores')
print("Series 객체:\n", s)
# 출력:
# 0    10
# 1    20
# 2    30
# 3    40
# Name: Scores, dtype: int64

# DataFrame 생성 예시 (2차원 테이블 형태)
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35]}
df = pd.DataFrame(data)
print("\nDataFrame 객체:\n", df)
# 출력:
#       Name  Age
# 0    Alice   25
# 1      Bob   30
# 2  Charlie   35

# DataFrame의 단일 열은 Series 객체임
print(f"\n'Name' 컬럼은 Series인가요? {isinstance(df['Name'], pd.Series)}") 
# 출력: 'Name' 컬럼은 Series인가요? True
```

---
## 3. DataFrame 생성 방법

데이터 분석을 시작하기 위해서는 먼저 DataFrame을 만들어야 합니다. Pandas는 여러 가지 방법으로 DataFrame을 처음부터 생성(Creating DataFrames from scratch)하는 것을 지원합니다.

주요 데이터 프레임 생성 방법은 다음과 같습니다:

### 3.1. Dictionary 이용: 파이썬의 딕셔너리(dictionary)를 사용하여 DataFrame을 생성할 수 있습니다.

**코드 예시:**
```Python
import pandas as pd

# 딕셔너리의 키(key)가 컬럼 이름, 값(value)은 해당 컬럼의 데이터(리스트)가 됨
data_dict = {'City': ['Seoul', 'Busan', 'Incheon'],
             'Population': [9700000, 3400000, 2900000],
             'Area': [605, 770, 1063]}
df_dict = pd.DataFrame(data_dict)
print("Dictionary로 생성된 DataFrame:\n", df_dict)
# 출력:
#       City  Population  Area
# 0    Seoul     9700000   605
# 1    Busan     3400000   770
# 2  Incheon     2900000  1063
```

### 3.2. List 이용: 리스트(list)를 기반으로 DataFrame을 생성할 수 있습니다.

**코드 예시:**
```Python
import pandas as pd

# 리스트의 리스트 형태로, 내부 리스트가 각 행(Row)을 의미함
data_list = [['A', 10], ['B', 20], ['C', 30]]
df_list = pd.DataFrame(data_list, columns=['ID', 'Value']) # 컬럼 이름을 지정
print("List로 생성된 DataFrame:\n", df_list)
# 출력:
#   ID  Value
# 0  A     10
# 1  B     20
# 2  C     30
```

### 3.3. Numpy Array 이용: 수치 계산에 강력한 Numpy 라이브러리의 배열(numpy array)을 통해 DataFrame을 만들 수 있습니다.

**코드 예시:**
```Python
import pandas as pd
import numpy as np

# 2차원 Numpy 배열 생성
data_numpy = np.array([[100, 150], [200, 250], [300, 350]])
df_numpy = pd.DataFrame(data_numpy, 
                        index=['Day1', 'Day2', 'Day3'], # 행 인덱스 지정
                        columns=['Sales', 'Cost'])    # 컬럼 이름 지정
print("Numpy Array로 생성된 DataFrame:\n", df_numpy)
# 출력:
#       Sales  Cost
# Day1    100   150
# Day2    200   250
# Day3    300   350
```

---
## 4. 주요 기능

Pandas는 데이터를 효율적으로 조작하기 위한 다양한 핵심 기능을 제공합니다.

### 4.1. 데이터 입출력 (File I/O)

- 파일을 읽거나 저장하는 기능 (`File read / File save`).
    
**코드 예시 (CSV 파일 예시):**
```Python
import pandas as pd
import os

# 예시 DataFrame 생성
df_io = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})

# DataFrame을 CSV 파일로 저장
df_io.to_csv('test_data.csv', index=False)
print("DataFrame을 'test_data.csv'로 저장했습니다.")

# CSV 파일 읽기
if os.path.exists('test_data.csv'):
    df_read = pd.read_csv('test_data.csv')
    print("\nCSV 파일에서 읽은 DataFrame:\n", df_read)
    # 출력:
    #    A  B
    # 0  1  3
    # 1  2  4
```

### 4.2. 데이터 선택 및 필터링 (Selecting and Filtering)

- 특정 값이나 이름을 기준으로 열을 선택하고 필터링합니다 (`Column selecting and filtering based on value / name`).
- **행 인덱싱(Row indexing)**: 행을 선택하기 위해 `loc` (label-based)와 `iloc` (integer-based) 두 가지 방법을 사용합니다.
    

**코드 예시:**
```Python
import pandas as pd
df_select = pd.DataFrame({
    'Name': ['A', 'B', 'C', 'D'],
    'Score': [90, 85, 95, 80]
}, index=['r1', 'r2', 'r3', 'r4'])

# 컬럼 이름으로 선택
names = df_select['Name']
print(f"컬럼 'Name' 선택:\n{names}")
# 출력: r1    A ...

# 조건 필터링 (Score가 90 이상인 행 선택)
filtered_df = df_select[df_select['Score'] >= 90]
print(f"\n점수 >= 90 필터링:\n{filtered_df}")
# 출력:
#     Name  Score
# r1    A     90
# r3    C     95

# loc (레이블 기반 인덱싱) - 'r3' 행 선택
row_loc = df_select.loc['r3']
print(f"\nloc['r3'] 선택:\n{row_loc}")
# 출력: Name      C ...

# iloc (정수 기반 인덱싱) - 두 번째 행(인덱스 1) 선택
row_iloc = df_select.iloc[1]
print(f"\niloc[1] 선택:\n{row_iloc}")
# 출력: Name      B ...
```

### 데이터 조작 및 분석 (Manipulation and Analysis)

- 데이터를 반복하거나(`Iteration`) 새로운 데이터를 추가하는 기능 (`Adding data`).
- 데이터의 평균, 중앙값 등 **기본적인 통계(Basic statistics)**를 계산할 수 있습니다.
- 데이터를 기반으로 간단한 그래프를 그리는 기능 (`Simple plotting`)도 제공합니다.
    
**코드 예시:**
```Python
import pandas as pd
df_anal = pd.DataFrame({
    'Value': [10, 20, 30, 40, 50],
    'Category': ['A', 'B', 'A', 'B', 'A']
})

# 새로운 컬럼 추가
df_anal['Double_Value'] = df_anal['Value'] * 2
print("새로운 컬럼 추가:\n", df_anal)
# 출력:
#    Value Category  Double_Value
# 0     10        A            20
# ...

# 기초 통계 계산 (describe)
stats = df_anal['Value'].describe()
print("\n'Value' 컬럼 통계:\n", stats)
# 출력: count    5.000000 ...

# 그룹별 통계 (Category별 평균)
group_mean = df_anal.groupby('Category')['Value'].mean()
print(f"\n카테고리별 Value 평균:\n{group_mean}")
# 출력:
# Category
# A    30.0
# B    30.0
# Name: Value, dtype: float64
```