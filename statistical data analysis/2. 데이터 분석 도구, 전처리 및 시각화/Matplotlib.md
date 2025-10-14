## Matplotlib: 파이썬 시각화의 기본을 다지는 라이브러리

Matplotlib는 파이썬에서 **정적(static), 애니메이션(animated), 상호작용(interactive)이 가능한 시각화**를 생성하기 위한 포괄적인(comprehensive) 라이브러리입니다. Matplotlib는 기본적인 그래프를 쉽게 그릴 수 있게 해주면서도, 복잡하고 어려운 시각화도 구현할 수 있게 해주는 강력한 도구입니다.

---

## 1. Matplotlib의 주요 기능

Matplotlib는 데이터 분석 결과를 효과적으로 전달하기 위한 다양한 핵심 기능을 제공합니다.

- **고품질 플롯 제작:** 출판 품질(publication quality) 수준의 그래프를 생성할 수 있습니다.
- **상호작용 피겨:** 확대(zoom), 이동(pan), 업데이트가 가능한 상호작용 피겨를 만들 수 있습니다.
- **커스터마이징:** 시각적인 스타일과 레이아웃을 자유롭게 사용자 정의할 수 있습니다.
- **다양한 환경 지원:** JupyterLab 환경이나 그래픽 사용자 인터페이스(GUI)에 그래프를 삽입(Embed)하여 사용할 수 있습니다.
- **파일 형식:** 다양한 파일 형식으로 결과를 내보낼 수 있습니다.
    
**코드 예시 (기본 플롯 생성 및 저장):**
```Python
import matplotlib.pyplot as plt
import numpy as np

# 1. 간단한 데이터 생성
x = np.arange(0, 10, 0.1)
y = np.sin(x)

# 2. pyplot을 사용하여 플롯 생성
plt.plot(x, y) 
plt.title("Simple Sine Wave") 

# 3. 그림 표시 및 파일로 저장
plt.savefig('sine_wave.png') # 파일 저장 기능
plt.show() # 화면에 표시
```

---

## 2. Matplotlib의 핵심 구성 요소

Matplotlib가 그림을 그릴 때 사용하는 구조는 크게 세 가지 핵심 요소로 나뉘는데, 이 구성 요소를 이해하는 것이 Matplotlib를 다루는 데 가장 중요합니다.

|구성 요소|역할|상세 설명|
|---|---|---|
|**Figure**|**전체 그림 영역**|Axes, 제목(titles), 범례(legends), 색상 막대(colorbars) 등 모든 하위 요소를 관리하는 가장 큰 컨테이너입니다.|
|**Axes**|**실제 플롯 영역**|데이터가 시각화되는 공간입니다. Figure 안에 포함되며, 대부분의 플로팅 기능은 Axes 객체에서 호출됩니다.|
|**Artists**|**시각적 요소**|Figure와 Axes, 그리고 그래프 위에 그려지는 모든 요소(예: 텍스트, 선, 점)를 통칭합니다.|

> **입력 데이터 유형:** Matplotlib 플로팅 함수에 데이터를 전달할 때, **NumPy 배열**을 입력으로 받는 것이 기본입니다. 다만, Pandas DataFrame이나 Dictionary도 `data` 키워드를 사용하여 입력으로 사용할 수 있습니다.

**코드 예시 (Figure 및 Axes 객체 확인):**
```Python
import matplotlib.pyplot as plt

# Figure와 Axes 객체 생성 (OO 스타일의 기본)
fig = plt.figure() # Figure 객체 생성
ax = fig.add_subplot(1, 1, 1) # Figure에 Axes(서브플롯) 추가
ax.plot([1, 2, 3], [10, 20, 30]) # Axes 객체에 플롯 그리기

print(f"Figure 객체 타입: {type(fig)}")
print(f"Axes 객체 타입: {type(ax)}")

plt.show()
```

---

## 3. Matplotlib 코딩 스타일

Matplotlib에는 그래프를 작성하는 두 가지 주요 코딩 스타일이 있으며, 복잡한 그래프를 다룰 때는 특정 스타일이 권장됩니다.

### 1. 객체 지향 (Object-Oriented, OO) 스타일

- **방식:** Figures와 Axes를 명시적으로 생성하고, 이들 객체에 직접 메서드를 호출하여 플롯을 그립니다.
- **권장 이유:** Matplotlib에서 권장하는 방식입니다. 특히 여러 개의 그래프가 포함된 **복잡한 피겨**를 그릴 때 효율적이고 관리가 쉽습니다.
    

**코드 예시 (OO 스타일):**
```Python
import matplotlib.pyplot as plt
import numpy as np

# Figure와 Axes를 동시에 생성하는 함수
fig, ax = plt.subplots(figsize=(6, 4)) 

# Axes 객체에 직접 메서드 호출
ax.plot(np.arange(5), label='Data A')
ax.set_title("OO Style Plot") # 제목 설정
ax.set_xlabel("X Axis")      # X축 레이블 설정
ax.legend()                  # 범례 추가

plt.show()
```

### 2. pyplot 스타일

- **방식:** `pyplot` 함수를 사용하여 Figures와 Axes의 생성 및 관리를 자동으로 처리하도록 의존합니다.
- **특징:** 편하고 빠르게 그림을 그릴 수 있다는 장점이 있습니다.
    
**코드 예시 (pyplot 스타일):**
```Python
import matplotlib.pyplot as plt

# plt 모듈의 함수만 사용
plt.plot([1, 2, 3], [5, 10, 15]) 
plt.title("pyplot Style Plot") # plt 함수를 사용하여 제목 설정
plt.xlabel("X Axis")           # plt 함수를 사용하여 X축 레이블 설정

plt.show()
```

> **헬퍼 함수(Helper Functions) 활용:** 같은 플롯을 다른 데이터에 대해 반복해서 그려야 하는 경우, 코드를 재사용하기 위해 래퍼(wrapper) 형태의 도우미 함수를 만들어 사용하는 것이 좋습니다.

![[matplotlib_coding_styles.png]]

---

## 4. 스타일링 및 커스터마이징

Matplotlib는 그래프의 시각적 요소를 세밀하게 조정할 수 있는 기능을 제공합니다.

**스타일 지정 방법**
- **메서드 호출 시 지정:** `Plot` 메서드를 호출할 때 색상, 선 스타일 등을 미리 지정할 수 있습니다.
- **Setter 사용:** 각 Artist 요소에 대해 Setter 메서드를 사용하여 나중에 스타일을 지정하거나 수정할 수 있습니다.
    

**주요 스타일링 요소**

|스타일링 요소|설명|관련 기능|
|---|---|---|
|**색상 (Color)**|Matplotlib는 다양한 형태의 색상 배열을 지원하여 시각적 구분을 돕습니다.||
|**마커 및 선 스타일**|다양한 기본 마커(marker)와 선 스타일(line style)을 제공합니다.||
|**축 레이블 및 제목**|플롯의 제목, X축 레이블, Y축 레이블 등을 지정하여 그래프의 내용을 설명합니다.|`set_title`, `set_xlabel`, `set_ylabel` 메서드 사용.|
|**범례 (Legend)**|여러 데이터 계열을 구분하기 위한 범례를 추가할 수 있습니다.||
|**축 수정**|`set_major_locator`와 `set_major_formatter` 등의 기능을 사용하여 X축의 눈금(ticks) 표시 방식이나 위치를 수정할 수 있습니다.||

**코드 예시 (스타일링 적용):**
```Python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

x = np.linspace(0, 2, 100)

# 선 스타일과 색상 지정, 범례 레이블 지정
ax.plot(x, x**2, 'r--', label='Quadratic') # 'r--': 빨간색, 점선
ax.plot(x, x**3, color='blue', linestyle='-', marker='o', label='Cubic') # 명시적 지정

# 축 레이블 및 제목 설정 (Setter 사용)
ax.set_title("Customized Plot")
ax.set_xlabel('Time (s)')
ax.set_ylabel('Power (mW)')

# 범례 표시
ax.legend() 

plt.show()
```

---

## 5. Matplotlib: Lineplot, Barplot, Boxplot

### 5.1. Lineplot (선 그래프)

**시계열 데이터**나 연속적인 데이터의 변화 추이를 보여줄 때 주로 사용됩니다. Seaborn의 `sns.lineplot` 함수는 Matplotlib의 Axes 객체에 선 그래프를 편리하게 그릴 수 있게 해줍니다1.

**코드 예시 (시계열 데이터 시각화):**
```Python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib import pyplot
import numpy as np

# 폰트 설정 (한글 깨짐 방지 예시)
mpl.rcParams['font.family'] = 'NanumGothic'

# 예시 데이터 (시계열 데이터 재구성)
data_line = pd.DataFrame({
    '기준일': pd.to_datetime(pd.date_range('2022-06-01', periods=90, freq='D')),
    '추가 확진자': np.random.randint(20, 50, 90) + np.arange(90) * 4 
})

fig, ax = pyplot.subplots(figsize=(12, 6))

sns.lineplot(
    ax=ax,
    data=data_line,
    x="기준일",
    y="추가 확진자"
)

# X축을 월 단위로 표시하도록 설정
months = mdates.MonthLocator()
months_fmt = mdates.DateFormatter('%m')
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(months_fmt)
ax.set_title("일별 확진자 변화")
ax.set_xlabel("날짜")
ax.set_ylabel("일별 확진자 (명)")

plt.show()
```

### 5.2. Barplot (막대 그래프)

**범주형 변수**에 따른 **수치형 변수**의 통계량(예: 합계 또는 평균)을 막대로 표시하며, `order` 파라미터를 사용하여 막대의 순서를 지정할 수 있습니다2.

**코드 예시 (지역구별 확진자 비교 - `order` 파라미터 사용):**
```Python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# 폰트 설정
mpl.rcParams['font.family'] = 'NanumGothic'

# 데이터 생성 (누적 확진자 수 예시)
data_barplot = pd.DataFrame({
    '지역구': ["종로구 전체", "용산구 전체", "관악구 전체", "강남구 전체"],
    '확진자': [12669, 21406, 48189, 50807]
})

plt.figure(figsize=(8, 5))
sns.barplot(
    x="지역구", 
    y="확진자", 
    data=data_barplot,
    # 확진자 수 기준 내림차순 정렬
    order=["강남구 전체", "관악구 전체", "용산구 전체", "종로구 전체"] 
)

plt.title("지역구별 누적 확진자 수")
plt.show()
```

### 5.3. Boxplot (상자 그림)

**범주형 변수**에 따른 **수치형 변수**의 **분포(사분위수, 중앙값, 이상치)**를 시각화하여 분포의 특성을 비교합니다. `hue` 파라미터를 사용하여 그룹별로 세분화된 비교가 가능합니다.

**코드 예시 (월별/지역구별 확진자 분포 비교 - `hue` 파라미터 사용):**
```Python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# 폰트 설정
mpl.rcParams['font.family'] = 'NanumGothic'

# 데이터를 Long-form으로 변환 (Boxplot 그룹화를 위해 필수)
dates = pd.to_datetime(pd.date_range('2022-06-01', periods=90, freq='D'))
months = dates.month.to_numpy()

# 데이터 시뮬레이션
data_for_box = {
    '월': np.tile(months, 4),
    '지역구': np.repeat(['종로구', '강남구', '관악구', '용산구'], 90),
    '확진자': np.concatenate([
        np.random.randint(50, 150, 90) * (months / 6),
        np.random.randint(100, 350, 90) * (months / 4),
        np.random.randint(80, 280, 90) * (months / 4.5),
        np.random.randint(70, 200, 90) * (months / 5)
    ]).astype(int)
}
boxdata_long = pd.DataFrame(data_for_box)

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=boxdata_long, 
    x="월", 
    y="확진자", 
    hue="지역구" # 지역구별로 그룹화하여 표시
)

plt.title("월별/지역구별 일별 확진자 수 분포")
plt.show()
```