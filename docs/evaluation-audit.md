# 평가 감사: 2026-09-08 포트폴리오 검토

| 논문/기획의 설명 | 확보 코드의 동작 | 영향 |
| --- | --- | --- |
| 시간순 7:1:2 분할 | stratify와 random_state=42의 70:30 random split | 미래 정보 혼입 가능, 독립 테스트 부재 |
| 정상 구간 AE 학습 | 전체 X로 fit/predict | 평가 구간 포함, 일반화 성능으로 해석 불가 |
| FinBERT 파인튜닝 | from_pretrained 후 pipeline 추론 | 추가 파인튜닝 주장 제외 |
| 뉴스 긍정 비율 | sentiment_mean > 0의 이진값 | 문장별 긍정 비율과 다름 |
| 텍스트량 | 일별 집계 후 text 없음 → 0 | 해당 특징이 정보 없음 |
| 검증 임계값 후 테스트 평가 | 같은 validation의 F1 최대화 및 평가 | 평가 낙관 편향 |
| 통합 모델 단일 결과 | 0.5/0.9/선택 임계값 블록 반복 및 다른 저장 출력 | 정확한 실험 버전 식별 필요 |

`precision_recall_curve`의 precision/recall은 thresholds보다 하나 더 깁니다. 다만 공식 API의 마지막 precision=1, recall=0에서는 현재 식의 F1이 0이고 `argmax`는 동률의 첫 위치를 고르므로, 길이 차이만으로 현재 코드에서 인덱스 오류가 발생한다고 단정할 수 없습니다. 확인된 핵심 문제는 같은 validation으로 임계값을 선택하고 평가한 편향입니다. [scikit-learn 공식 API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)를 기준으로 2026-09-09 감사 설명을 정정했습니다. 데이터 부족 시 best window가 None인 경로, 긴 뉴스의 truncation 처리, latin1/skip에 의한 조용한 데이터 손실도 점검 대상입니다.

공개본의 변경은 파일명·입력 경로·Notebook 메타데이터·출력 분리에 한정합니다. 실험 설계 교정은 별도의 신규 실험으로 진행해야 하며, 과거 수치와 섞으면 안 됩니다.

## 코드에서 검증할 위치

- `src/legacy_experiment.py:59`, `run_finbert`: FinBERT 모델 로딩·추론. 수면 LLM 학습과 별개다.
- `src/legacy_experiment.py:86`, `lstm_autoencoder_detect`: 전체 시계열로 scaler fit. 96~97행에서 같은 X로 AE fit/predict.
- `src/legacy_experiment.py:71`, `daily_sentiment`: 날짜별 평균 후 원문 텍스트가 사라짐. `make_sentiment_df` 142~143행의 긍정 여부/상수 텍스트 길이를 함께 확인해야 한다.
- `src/legacy_experiment.py:341`: random stratified split. 350행의 임계값 선택은 독립 테스트 검증이 아니다. 391/441행에 유사 블록이 반복된다.

2026-09-09 검토에서는 모델 재학습이나 과거 성능 갱신을 하지 않았다. 뉴스 원자료·검증된 실행 환경이 없고, 시간순 분할만 바꾸어도 상류 AE와 튜닝의 누수까지 자동으로 해결되지 않기 때문이다. 별도 실험에서는 train-only 전처리/AE, validation-only 선택, 최종 test 1회 평가를 전 구간에 적용해야 한다. 데이터 수집시각과 뉴스 공개시각, 날짜 병합에 따른 시점 누수도 함께 검증한다.
