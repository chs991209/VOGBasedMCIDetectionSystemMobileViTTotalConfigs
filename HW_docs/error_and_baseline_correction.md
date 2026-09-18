## 1. 개요 및 파이프라인 상의 위치

본 연산은 원본 VOG 시계열 신호(안구 좌표 및 자극 좌표)로부터 자극 전환 시점($t_{\text{event}}$)을 기준으로 추출한 1.0초 길이의 에포크(Epoch, $[-0.2, +0.8]\,\text{s}$)에 대해, 연속 웨이블릿 변환(CWT)을 적용하기 직전에 수행되는 정규화 단계입니다.

---

## 2. 구체적인 처리 과정 및 수식

### 2.1 시선 추적 오차 신호 산출 (Gaze Tracking Error Calculation)

1. **신호 정의**:
   샘플링 레이트 $f_s \approx 120\,\text{Hz}$ 환경에서, 자극 전환 시점 $t_{\text{event}}$ 기준 $t \in [t_{\text{event}} - 0.2, \, t_{\text{event}} + 0.8]\,\text{s}$ 구간 동안 기록된 안구 위치와 목표물 위치:
   - 좌안 수평/수직 위치: $x_{LH}(t), y_{LV}(t)$
   - 우안 수평/수직 위치: $x_{RH}(t), y_{RV}(t)$
   - 시각 자극 목표물 위치: $x_{TH}(t), y_{TV}(t)$

2. **안티새케이드(Anti-saccade) 조건의 목표 좌표 반전**:
   안티새케이드 과제(Task ID 2, 6)에서는 피험자에게 자극의 반대 방향을 주시하도록 지시하므로, 주 과제 축(Task axis)의 목표물 좌표 부호를 반전합니다:
   $$x_{T,\text{task}}(t) \leftarrow - x_{T,\text{task}}(t)$$

3. **4채널 오차 신호 계산**:
   좌우안 각각의 수평(H) 및 수직(V) 4개 채널에 대해 안구 위치와 목표 위치 간의 차이(Tracking Error)를 산출합니다:
   $$e_{LH}(t) = x_{LH}(t) - x_{TH}(t)$$
   $$e_{RH}(t) = x_{RH}(t) - x_{TH}(t)$$
   $$e_{LV}(t) = x_{LV}(t) - x_{TV}(t)$$
   $$e_{RV}(t) = x_{RV}(t) - x_{TV}(t)$$

### 2.2 사전 자극 구간 베이스라인 보정 (Pre-stimulus Baseline Correction)

1. **사전 자극 구간(Pre-stimulus window) 정의**:
   자극이 전환되기 직전의 0.2초 구간($t \in [t_{\text{event}} - 0.2, \, t_{\text{event}}]$)을 베이스라인 구간으로 설정합니다. 해당 구간의 이산 표본 수는 다음과 같습니다:
   $$N_{\text{pre}} = \lfloor 0.2 \cdot f_s \rfloor$$

2. **베이스라인 직류(DC) 오프셋 추정**:
   각 채널 $c \in \{LH, RH, LV, RV\}$에 대해 사전 자극 구간의 시간 평균값을 산출합니다:
   $$\mu_{\text{pre}, c} = \frac{1}{N_{\text{pre}}} \sum_{k=1}^{N_{\text{pre}}} e_c(t_k)$$

3. **베이스라인 감산 (Zero-centering)**:
   전체 1.0초 윈도우 신호 $e_c(t)$에서 추정된 베이스라인 평균 $\mu_{\text{pre}, c}$를 감산하여 보정된 오차 신호 $\bar{e}_c(t)$를 획득합니다:
   $$\bar{e}_c(t) = e_c(t) - \mu_{\text{pre}, c}, \quad \forall t \in [t_{\text{event}} - 0.2, \, t_{\text{event}} + 0.8]$$

보정된 신호 $\bar{e}_c(t)$는 사전 자극 구간의 평균이 $0$이 되도록 정렬됩니다:
$$\frac{1}{N_{\text{pre}}} \sum_{k=1}^{N_{\text{pre}}} \bar{e}_c(t_k) = 0$$

---

## 3. 수식 전개의 과학적·공학적 근거

### 3.1 신경생리학적 근거: 망막 오차(Retinal Motor Error) 제어

- **안구운동 신경망의 제어 변수**:
  인간의 새케이드 생성 신경계(전두엽 안구운동 영역 FEF, 상구 Superior Colliculus, 뇌간 망상체 Brainstem reticular formation)는 안구의 절대적인 안와 내 각도(Absolute orbital position)를 직접 제어하지 않습니다. 대신 목표물의 상이 망막 중심와(Fovea)로부터 벗어난 거리인 **망막 오차(Retinal Error)** 또는 상구 신경세포가 연산하는 **운동 오차(Motor Error)**를 $0$으로 수렴시키기 위한 피드백/피드포워드 제어를 수행합니다.
- **오차 신호 변환의 당위성**:
  절대 시선 좌표 $x_{\text{eye}}(t)$를 그대로 신경망에 입력하면 목표물의 위치가 바뀔 때마다 신호의 기저 레벨이 변동합니다. 반면 $e(t) = x_{\text{eye}}(t) - x_{\text{target}}(t)$로 변환하면 신경계가 제어하고자 하는 목표 오차 변수 자체를 직접 모델링할 수 있습니다.
- **인용 문헌**:
  - Sparks, D. L. (2002). "The brainstem control of saccadic eye movements." *Nature Reviews Neuroscience*, 3(12), 952-964.
  - Munoz, D. P., & Everling, S. (2004). "Look away: the antisaccade task and the voluntary control of eye movement." *Nature Reviews Neuroscience*, 5(3), 218-228.

### 3.2 계측공학적 근거: VOG 하드웨어 드리프트 및 영점 오프셋 제거

- **비디오 안구운동 기록기(VOG)의 오프셋 요인**:
  VOG 장비는 안경/헤드셋 형태의 카메라로 동공 중심(Pupil center)과 각막 반사(Corneal reflection)를 측정합니다. 측정 과정에서 다음과 같은 비생리학적 오프셋이 발생합니다:
  1. 헤드셋의 미세 밀림(Headband slippage)
  2. 표정 변화나 눈 깜빡임에 의한 카메라-안구 간 상대 위치 변위
  3. 피험자의 기저 사시(Strabismus) 또는 안구 편위
- **사전 0.2초 구간 활용 이유**:
  자극이 전환되기 전 $0.2\,\text{s}$ 구간은 피험자가 이전 목표점을 주시하고 있는 고정 상태(Fixation period)입니다. 이 구간에 나타나는 0이 아닌 오차는 생리학적 반응 지연이 아니라 **기저 드리프트(Drift) 및 캘리브레이션 영점 편향(Zero-point calibration offset)**입니다. 따라서 이 값을 감산함으로써 피험자 간, 시도 간 하드웨어 측정 편향을 제거하고 순수한 자극 유발 동적 반응(Stimulus-evoked dynamic response)만 격리합니다.
- **인용 문헌**:
  - Holmqvist, K., Nyström, M., Andersson, R., Dewhurst, R., Jarodzka, H., & van de Weijer, J. (2011). *Eye tracking: A comprehensive guide to methods and measures*. Oxford University Press.
  - Hessels, R. S., Niehorster, D. C., Nyström, M., Andersson, R., & Hooge, I. T. (2018). "Is the pupil center-corneal reflection method always better than the pupil center method in the presence of head movement?" *Behavior Research Methods*, 50(5), 2030-2042.

### 3.3 신호처리적 근거: CWT 스펙트럼 누설(Spectral Leakage) 및 경계 왜곡 차단

- **직류(DC) 성분 잔존 시 CWT 왜곡 현상**:
  연속 웨이블릿 변환은 모체 웨이블릿 $\psi(t)$와의 합성곱(Convolution)으로 계산됩니다:
  $$W(a, \tau) = \frac{1}{\sqrt{a}} \int \bar{e}(t) \psi^*\left(\frac{t - \tau}{a}\right) dt$$
  모체 웨이블릿은 복소 모를레(Complex Morlet, `cmor4.0-1.0`)이며, 허용성 조건(Admissibility condition, $\int \psi(t)dt = 0$)을 만족합니다. 그러나 이산 신호의 유한 구간 연산 시 신호에 0이 아닌 상수값(DC offset, $\mu \neq 0$)이 잔존하면 다음과 같은 문제가 발생합니다:
  1. 파이프라인의 분석 대역인 **$15\text{–}60\,\text{Hz}$의 최저 주파수 영역($15\,\text{Hz}$ 인근)으로 강한 에너지 누설(Spectral leakage)**이 발생합니다.
  2. 윈도우 양단에서 불연속면이 형성되어 웨이블릿 영향 원뿔(Cone of Influence, COI) 내의 경계 왜곡(Boundary artifact)이 증폭됩니다.
- **평균 감산의 효과**:
  사전 자극 구간의 평균을 제거하여 기저 전압/변위를 $0$ 근방으로 정렬함으로써, 직류 성분에 의한 스펙트럼 왜곡을 억제하고 새케이드 전환 시점에 발생하는 안구 미세 진동(Micro-tremor) 및 급격한 속도 전이 신호의 순수 진폭을 CWT 스칼로그램 상에 정확히 반영합니다.
- **인용 문헌**:
  - Torrence, C., & Compo, G. P. (1998). "A practical guide to wavelet analysis." *Bulletin of the American Meteorological Society*, 79(1), 61-78.
  - Mallat, S. (2008). *A wavelet tour of signal processing: the sparse way*. Academic Press.

### 3.4 사전 자극 0.2초(200ms) 설정의 구체적 근거

수식에서 사전 자극 윈도우 길이를 $0.2\,\text{s}$ ($200\,\text{ms}$)로 지정한 배경은 신경생리학, 통계 신호처리, 실험 패러다임 설계의 4가지 근거에 기반합니다.

#### (1) 안구운동 생리학적 근거: 안정 주시(Stable Fixation)의 최소 지속 시간
- 인간의 정상 주시(Fixation) 지속 시간은 평균 $200\text{–}300\,\text{ms}$입니다. 
- 자극 전환 직전의 $200\,\text{ms}$는 피험자가 이전 목표물에 대한 주시 상태를 안정적으로 유지하고 있는 단일 주시(Single fixation) 단위와 시간적으로 정확히 일치합니다.
- *인용*: Rayner, K. (1998). "Eye movements in reading and information processing: 20 years of research." *Psychological Bulletin*, 124(3), 372-422.

#### (2) 시계열적 분리(Temporal Segregation): 이전 반응 완료 및 신규 반응 미도달
- **이전 반응의 완전 종결**: 본 실험의 자극 간 간격(Inter-stimulus interval, ISI)은 통상 $1.0\text{–}2.0\,\text{s}$입니다. 이전 자극에 유발된 주 새케이드(잠복기 약 $200\,\text{ms}$ + 이동 시간 약 $50\,\text{ms}$)와 이후의 교정 새케이드(약 $150\text{–}200\,\text{ms}$)는 자극 전환 후 최대 $450\text{–}500\,\text{ms}$ 이내에 종결됩니다. 따라서 자극 전환 직전 $200\,\text{ms}$는 이전 운동의 동적 잔류가 완전히 소멸된 구간입니다.
  - *인용*: Leigh, R. J., & Zee, D. S. (2015). *The Neurology of Eye Movements*. Oxford University Press.
- **신규 자극 반응의 미도달**: 시각 정보가 망막에서 시각 피질 및 전두엽을 거쳐 안구운동 신경핵에 도달하는 최소 반응 잠복기(Minimum saccade latency)는 $150\text{–}200\,\text{ms}$ (초고속 반응인 Express saccade의 경우도 $80\text{–}120\,\text{ms}$)입니다. $80\,\text{ms}$ 미만의 반응은 예측 반응(Anticipatory saccade)으로 간주되어 기각됩니다. 따라서 $t \in [-200, 0]\,\text{ms}$ 구간은 새로운 시각 자극에 대한 어떠한 자극 유발 안구 운동도 시작될 수 없는 순수한 기저 구간임이 보장됩니다.
  - *인용*: Fischer, B., & Weber, H. (1993). "Express saccades and visual attention." *Behavioral and Brain Sciences*, 16(3), 553-567.

#### (3) 통계적 신호처리 관점: 표본 크기($N=24$)와 국소 정상성(Local Stationarity) 간 절충
- 본 시스템의 샘플링 레이트 $f_s = 120\,\text{Hz}$에서 $0.2\,\text{s}$는 정확히 $N_{\text{pre}} = 0.2 \times 120 = 24$개의 이산 표본을 산출합니다.
- **고주파 노이즈 감쇄율**: 중심극한정리(CLT)에 의해, $N=24$ 표본 평균의 표준 오차(Standard Error)는 단일 표본 노이즈 표준편차 $\sigma$ 대비 다음과 같이 감소합니다:
  $$\text{SE}(\mu_{\text{pre}}) = \frac{\sigma}{\sqrt{24}} \approx 0.204 \cdot \sigma$$
  즉, 기저 안구 미세 진전(Ocular micro-tremor, $50\text{–}100\,\text{Hz}$) 및 센서 양자화 잡음을 약 $80\%$ 상쇄하여 직류 오프셋을 안정적으로 추정할 수 있습니다.
- **시간 창 길이의 한계점**:
  - $0.2\,\text{s}$보다 짧은 경우 (예: $0.05\,\text{s}$, $N=6$): 표본 부족으로 고주파 센서 노이즈가 베이스라인 값에 과도하게 반영되어 오프셋 추정이 불안정해집니다.
  - $0.2\,\text{s}$보다 긴 경우 (예: $0.5\,\text{s}$): 이전 시행의 후속 교정 새케이드가 침범하거나 안구의 서수 표류(Slow drift)가 누적되어 신호의 국소 정상성(Stationarity)이 붕괴됩니다.
- *인용*: Luck, S. J. (2014). *An introduction to the event-related potential technique*. MIT press. (생체 신호 이벤트 고정 분석에서 $100\text{–}200\,\text{ms}$ 사전 베이스라인 설정의 통계적 타당성).

#### (4) 에포크 전체 기하학적 정합성 ($0.2\,\text{s} + 0.8\,\text{s} = 1.0\,\text{s}$)
- 사후 자극 구간 $0.8\,\text{s}$ ($800\,\text{ms}$)는 잠복기($200\,\text{ms}$), 주 새케이드($50\,\text{ms}$), 교정 새케이드($200\,\text{ms}$) 및 최종 안착 주시($350\,\text{ms}$)를 모두 수용하는 최소 생리학적 시간입니다.
- 사전 자극 $0.2\,\text{s}$와 결합하여 총 윈도우 길이가 정확히 $1.0\,\text{초}$($120$ 샘플)로 정합되며, 이는 $32$개의 시간 빈으로 CWT 리샘플링 후 신경망 입력 크기인 $256 \times 256$으로 8배 정수 업스케일링할 때 시간 왜곡을 방지하는 정수 그리드 구조를 형성합니다.

---

## 4. 요약

| 파라미터 / 단계 | 설정값 및 수식 | 목적 | 핵심 근거 |
|---|---|---|---|
| **오차 신호 계산** | $e(t) = \text{eye}(t) - \text{target}(t)$<br>(Anti 과제: $\text{target} \leftarrow -\text{target}$) | 목표 추적 편차 추출 | 신경계 제어 변수(망막 운동 오차) 정렬 (Sparks, 2002) |
| **사전 자극 구간 길이** | **$0.2\,\text{s}$ ($200\,\text{ms}$, $N=24$)** | 안정 주시 구간 표본화 | ① 단일 주시 지속 시간 일치 (Rayner, 1998)<br>② 이전 반응 종료 및 신규 반응 미도달 (Fischer & Weber, 1993)<br>③ $N=24$ 노이즈 80% 상쇄 및 표류 배제 (Luck, 2014) |
| **사전 자극 평균 추정** | $\mu_{\text{pre}} = \frac{1}{N_{\text{pre}}} \sum_{k=1}^{N_{\text{pre}}} e(t_k)$ | 측정 장비의 기저 DC 편향 추정 | 주시 구간 내 헤드셋 밀림 및 기저 사시 오프셋 분리 (Holmqvist et al., 2011) |
| **베이스라인 감산** | $\bar{e}(t) = e(t) - \mu_{\text{pre}}$ | 순수 동적 반응 추출 및 영점 정렬 | CWT $15\text{–}60\,\text{Hz}$ 대역의 저주파 누설 및 경계 아티팩트 차단 (Torrence & Compo, 1998) |

