# 평가 설계 감사

2026-09-10에 코드, 두 Notebook, 원고 버전과 실제 뉴스 CSV를 대조했습니다. 실험의 핵심 알고리즘을 소급 변경하지 않고, 확인된 사실과 후속 개선을 구분합니다.

| 항목 | 실제 구현 | 영향 |
|---|---|---|
| 전처리 | 전체 거래량으로 MinMaxScaler fit | 평가 구간의 분포가 학습에 포함 |
| AE 학습 | 전체 X로 fit/predict | 정상 전용 학습과 독립 재구성 평가가 아님 |
| AE 선택 | 같은 라벨에서 window/epoch F1 최대화 | 설정 선택과 평가의 재사용 |
| 분류기 분할 | stratified random 70:30, seed42 | 시간 순서가 분리되지 않음 |
| threshold | validation F1로 선택 | 같은 validation 평가의 낙관 편향 |
| 그림/표 | 평가0.5/0.9/best의3블록, 그림은 모두 best | 표와 그림의 동일 실험 여부 불분명 |
| 긍정 비율 | 일별 평균 감성>0인 이진값 | 실제 기사별 긍정 비율과 다름 |
| 텍스트량 | 집계 후 text 열 부재로0 | 해당 특징에 정보가 없음 |
| 결측 | 뉴스/시세/라벨 공백을 모두0 | 정보 없음과 정상/중립을 혼동 |
| AEMD 입력 | 뉴스2023~2025, 시세 요청2018~2021 | 같은 사건의 시점별 특징이 아님 |

## 코드 근거

- [`lstm_autoencoder_detect`](../src/legacy_experiment.py):86행 scaler fit,96~97행 전체 X로 fit/predict.
- `daily_sentiment`69행과 `make_sentiment_df`140행: 평균 집계 이후 원문 text가 사라지고, 긍정 이진값과 상수 텍스트 길이를 생성합니다.
- `train_test_split`341/391/441행: 같은 random70:30 분할을 사용합니다.
- `y_pred`352/402/452행: 각각0.5/0.9/best_thr로 표를 평가합니다. 전체 기간 그림356/406/456행은 모두 best_thr입니다.
- [`exploration.ipynb`](../notebooks/exploration.ipynb) 원본 cell30: Best threshold를 계산·출력하면서 실제 표의 y_pred에는0.5를 적용합니다.

`precision_recall_curve`의 precision/recall 배열은 thresholds보다 하나 더 깁니다. 그러나 마지막 precision=1, recall=0에서는 현재 식의 F1이0이고 `argmax`는 동률의 첫 위치를 고르므로, 배열 길이 차이만으로 현재 코드의 인덱스 오류를 단정하지 않습니다. 확인된 문제는 선택·평가 데이터 재사용입니다. [scikit-learn 공식 API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)를 기준으로 한 이전 정정을 유지합니다.

## 원고 버전의 차이

12쪽 제출본은 시간순7:1:2를 서술하지만, 이번에 공개한18쪽 원고의 원본p9는 전체 시계열 AE 학습과 XGBoost7:3 분할을 명시합니다. 원본p15~16은 일반화 한계도 인정합니다. 다만 앞부분의 정상 구간 학습 설명과 상충합니다. 자세한 대조는 [연구 원고 정정](research-errata.md)에 있습니다.

## 다음 실험의 조건

입력 기간·기사 계보·수집 시각을 확정한 뒤 시간순 train/validation/test를 분리해야 합니다. scaler와 AE는 train에서만 학습하고, window/epoch와 threshold는 validation에서만 선택하며, 최종 test는 한 번 평가합니다. 시간순 split 하나만 바꿔서는 상류 AE 학습과 설정 선택의 누수가 해결되지 않습니다.

이번 작업에서 모델을 재학습하거나 과거 성능을 갱신하지 않았습니다. [입력 품질 검사](dataset.md)와 [재현성 범위](reproducibility.md)를 참고하세요.
