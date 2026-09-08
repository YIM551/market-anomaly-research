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

`precision_recall_curve`의 precision/recall은 thresholds보다 하나 더 길 수 있습니다. 현재 코드의 `thr[np.argmax(f1s)]`는 마지막 원소가 최대인 경우 범위를 벗어날 수 있습니다. 데이터 부족 시 best window가 None인 경로, 긴 뉴스의 truncation 처리, latin1/skip에 의한 조용한 데이터 손실도 점검 대상입니다.

공개본의 변경은 파일명·입력 경로·Notebook 메타데이터·출력 분리에 한정합니다. 실험 설계 교정은 별도의 신규 실험으로 진행해야 하며, 과거 수치와 섞으면 안 됩니다.
