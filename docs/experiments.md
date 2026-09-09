# 실험 기록과 결과의 해석

## 실험 설정

주가 거래량의 슬라이딩 윈도우를 LSTM Autoencoder로 재구성하고, MSE와 뉴스 감성 특징을 XGBoost에 입력합니다. AE 은닉 크기는 80, batch는 64이며 Adam/MSE를 사용합니다. window 후보는 5/10/20/50/100/250, epoch 후보는 10/35/70입니다. 원고에 적힌 Close·체결강도, EarlyStopping·dropout·L2, FinBERT 자체 학습은 확보 코드에 없습니다.

분류기는 `scale_pos_weight`를 사용하며 XGBoost의 random_state는 42입니다. 탐색 Notebook 일부 블록은 전체 y에서 가중치를 계산하고, 통합 코드는 train y에서 계산합니다. AE random seed, GPU/VRAM, 패키지 lock, 뉴스·시세 snapshot은 복구되지 않았습니다. FinBERT 저장 출력에는 CPU 사용 문구가 있으나 이는 전체 학습 환경의 증명이 아닙니다.

## 저장 출력의 버전

| 자료 | 위치(원본0-based cell) | 해석 |
|---|---|---|
| 탐색 Notebook |13/17/30|README 표의 감성/AE/XGB 출처|
| 탐색 Notebook |16|5종목×6window×3epoch의 90개 기록, 동일 데이터에서 선택|
| 탐색 Notebook |26|단일 validation PR곡선/F1 .667/AUC .787, 해당 stock 식별 불충분|
| 통합 Notebook |9/10/11/12/13|감성/AE/XGB의 다른 반복 출력|
| 주변 출력 CSV |Out_26.csv|별도 5행 AE 결과, 예: NBDR F1 .867. 실험 설정 연결 부족|
| 주변 JPEG |XGBoost Table.jpeg|5종목 지표 모두 0인 별도 산출물, 설정 미복구|

원본 핵심 코드와 셀 순서를 보존했으며 실행출력의 개인PC경로는 제거하고 수치만 [historical-results.txt](historical-results.txt)에 발췌했습니다. 서로 다른 실행에서 최고 수치만 골라 하나의 성능으로 합치지 않습니다.

## 평가 단위와 threshold

- AE는 거래일별 플래그, 감성은 뉴스가 있는 날, 통합은 달력 날짜 join의 validation을 평가합니다. 분모가 같지 않습니다.
- `legacy_experiment.py`352행은 0.5,402행은 0.9,452행은 best_thr로 validation을 예측합니다. 각 블록의 전체 기간 그림(356/406/456행)은 모두 best_thr를 사용합니다.
- 탐색 Notebook cell30도 Best threshold를 출력하지만 y_pred는 0.5입니다. 표에 표시된 Best threshold로 해당 표를 계산했다고 주장하면 안 됩니다.
- 표의 F1과 그림의 붉은 구간은 서로 다른 실행·임계값일 수 있습니다. 원본 그림은 정성적 실행 증거로 공개하며 표의 검증 근거로 대체하지 않습니다.

## 실패 사례: GME 감성

탐색 Notebook cell14에 저장된 9개 뉴스 일자 라벨과 플래그에서 TP3, FP5, FN0, TN1을 직접 집계했습니다. Precision=3/8=.375, Recall=3/3=1, F1=6/11=.54545로 cell13과 일치합니다. Recall이 높아도 8번 경보 중 5번이 라벨 밖이라는 점을 설명할 수 있습니다.

원본 숫자에서 추출한 [9개 관측행](evidence/gme-sentiment-observations.csv)과 [집계](evidence/gme-sentiment-confusion.json)를 제공합니다. 뉴스 본문은 포함하지 않습니다. 이는 새 모델 실행이나 독립 test가 아니며 시장 전체 오탐률로 일반화하지 않습니다.

## Why It Underperformed / What I Learned

GME/NBDR에서는 대표 표의 통합 F1이 AE보다 낮고, AEMD AE는 목표 구간을 놓칩니다. 뉴스 희소성이나 문맥의 영향이 가능하지만 인과 원인을 확정할 통제 실험은 없습니다. 특히 AEMD는 뉴스 기간 불일치가 확인되어 '감성으로 문맥 문제를 해결했다'고 설명할 근거가 부족합니다.

모델 결합보다 먼저 시점, schema, 기사 계보, 분모를 확인해야 한다는 점이 이번 검토에서 배운 내용입니다. 전체 데이터 AE 학습과 동일 validation 튜닝은 [평가 감사](evaluation-audit.md)에 정리했습니다. 후속 실험은 정상 train 전용 학습과 시간순 분할, validation 전용 튜닝, 최종 test 1회 평가로 별도 기록해야 합니다.
