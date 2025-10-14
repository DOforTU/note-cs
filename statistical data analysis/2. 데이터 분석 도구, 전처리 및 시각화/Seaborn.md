## Seaborn: 통계적 시각화를 위한 고급 라이브러리

Seaborn은 Matplotlib를 기반으로 개발된 파이썬 데이터 시각화 라이브러리입니다1. Matplotlib가 시각화의 기본 틀을 제공한다면, Seaborn은 통계 분석 결과를 시각화하는 데 특화된 고급 기능들을 제공하여 더욱 매력적이고 유익한 통계 그래픽을 쉽게 그릴 수 있도록 돕습니다.

---

## 1. 통계 그래픽에 최적화된 고급 인터페이스

Seaborn의 핵심은 사용자에게 **고급 인터페이스(high-level interface)** 를 제공하는 것입니다.

이는 복잡한 통계 계산 및 시각적 설정을 내부적으로 처리하여, 사용자가 적은 코드로도 **매력적이고 정보력이 풍부한(attractive and informative) 통계 그래픽**을 그릴 수 있게 해줍니다4

---

## 2. 주요 기능 및 특징

#### 2.1. Pandas DataFrame 사용의 편리성

Seaborn은 **Pandas에서 더 쓰기 편한 모듈**로 개발되었습니다. 데이터 분석 시 Pandas DataFrame을 직접 활용하여 시각화를 수행하는 데 최적화되어 있어 데이터 처리와 시각화 간의 연계가 매우 자연스럽습니다.

#### 2.2. 통계적 추정 (Statistical Estimation) 기능 제공

Seaborn은 데이터의 관계나 분포를 단순히 보여주는 것을 넘어, **통계적 추정(statistical estimation)** 을 통한 분석 정보를 시각화에 통합합니다.

- **신뢰 구간 (Confidence Interval) 및 오차 막대 (Error Bar):** Seaborn은 통계량에 대한 **신뢰 구간**을 **부트스트랩핑(bootstrapping)** 방식으로 추정하여 **오차 막대** 형태로 그래프에 표시해 줍니다.
    
- **회귀 모델:** 데이터셋의 조건부 하위 집합에 걸쳐 회귀 모델을 적합시키는 편리한 인터페이스로 활용되기도 합니다.
    
- **세부 관계 표현:** `hue`, `size`, `style` 등의 파라미터를 사용하여 데이터의 다양한 하위 집합 간의 x와 y의 관계를 보여줄 수 있습니다.
    

#### 2.3. 자동화된 기본 설정 (Opinionated Defaults)

Seaborn은 그래프를 그릴 때 **Opinionated defaults**를 제공하여 사용 편의성을 높입니다.

- **자동 레이블 및 범례:** Seaborn은 주어진 데이터를 바탕으로 축 레이블(axis label)이나 범례 레이블(legend label) 등을 **자동으로 채워주는** 기능을 제공합니다.
    
- **유연한 커스터마이징:** 이러한 자동 설정 기능 외에도 유연한 커스터마이징이 가능하여 사용자가 원하는 방식으로 시각적 요소를 조정할 수 있습니다.
    

---

## 3. 주요 플롯 함수별 코드 예시

Pandas, NumPy는 앞서 정의된 것으로 가정합니다.

### 3.1. 관계형 플롯 (relplot)

`sns.relplot`은 산점도(scatter plot) 또는 선 그래프(line plot)를 포함하며, `col`, `row` 등을 사용하여 데이터를 하위 집합으로 나누어 그릴 수 있습니다.

**코드 예시 (Scatter Plot):**
```Python
import seaborn as sns
sns.set_theme()

# 내장 데이터셋 'tips' 로드
tips = sns.load_dataset("tips")

# relplot을 이용해 scatter plot 생성 (28페이지 참고)
# col="time"으로 점심(Lunch)과 저녁(Dinner)으로 분할
# hue, style, size로 'smoker'와 'size' 변수를 추가하여 세부 관계 표현
sns.relplot(
    data=tips,
    x="total_bill", y="tip", col="time",
    hue="smoker", style="smoker", size="size"
)
plt.show() # Matplotlib의 plt를 사용하여 표시
```

### 3.2. 선형 모델 플롯 (lmplot)

`sns.lmplot`은 두 변수 간의 **선형 회귀 관계**를 시각화합니다. 자동으로 회귀선과 함께 신뢰 구간(confidence interval, CI)을 표시해 줍니다.

**코드 예시 (Regression Plot):**
```Python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

# lmplot을 이용해 total_bill과 tip의 선형 관계 시각화 (33페이지 참고)
# col="time", hue="smoker"를 사용하여 조건부 하위 집합별 회귀 모델 적합 및 시각화
sns.lmplot(
    data=tips, 
    x="total_bill", y="tip", 
    col="time", 
    hue="smoker"
)
plt.show()
```

### 3.3. 분포 플롯 (displot)

`sns.displot`은 단일 변수의 **분포**를 시각화하는 Figure-level 인터페이스입니다. 히스토그램 (`kind="hist"`), KDE 플롯 (`kind="kde"`), ECDF 플롯 (`kind="ecdf"`) 등을 그릴 수 있습니다.

**코드 예시 (Histogram & KDE):**
```Python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

# displot을 이용해 total_bill의 분포 시각화 (35페이지 참고)
# kind="hist" (기본값)으로 히스토그램, kde=True로 커널 밀도 추정 곡선 추가
# col="time"으로 점심/저녁별로 분할
sns.displot(
    data=tips, 
    x="total_bill", 
    col="time", 
    kde=True
)
plt.show()
```