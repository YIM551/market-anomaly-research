# 실행 조건과 재현성

## 네트워크 없이 실행하는 검증

Python 3 표준 라이브러리만 사용합니다. 모델이나 ML 패키지를 설치할 필요가 없습니다.

```bash
python scripts/check_sources.py
python -m unittest discover -s tests -v
python scripts/validate_news.py --data-dir data/raw
```

첫 명령은 소스 1개와 Notebook 2개의 문법·출력 상태만 확인합니다. 두 번째 명령은 합성 CSV로14개 테스트를 실행합니다. 세 번째는 사용 권한이 있는 실제 뉴스의 schema, 날짜, 기간을 검사하며 입력이 없거나 결함이 있으면 exit1을 반환합니다.

```bash
python scripts/validate_news.py --data-dir data/raw --ticker GME
python scripts/validate_news.py --data-dir data/raw --ticker AEMD --encoding latin1
```

인코딩은 검토자가 명시해야 합니다. 검사기는 데이터를 고치거나 건너뛰지 않으며 진단에 뉴스 본문·요약·개인PC 절대경로를 포함하지 않습니다. 출처, 재배포권, 실제 기사 존재 여부와 모델 평가 누수는 별도 확인 사항입니다.

## 원래 모델 실험의 조건

`requirements.txt`는 실제 import에서 정리한 미고정 의존성 목록이며 검증된 lock 환경이 아닙니다. 원래 Notebook의 Python 2.7.6 메타데이터는 f-string 등 Python 3 코드와 상충해 공개본에서는 Python 3 커널로 표시했습니다. 정확한 Python, TensorFlow, CUDA, Transformers 버전과 GPU는 미확인입니다.

`src/legacy_experiment.py`는 import 시 FinBERT 다운로드와 시세 요청, 다수 학습을 실행합니다. 코드 살펴보기만 하려는 경우 import하지 마세요. 현재 입력을 그대로 복사하면 AEMD 헤더 오류 등이 발생합니다. 외부 시세 가용성, 뉴스 권한, 환경을 확정하고 입력 검사를 통과한 후 별도 신규 실험으로 실행해야 합니다.

탐색 Notebook은 역사적 작업 흐름을 보존한 자료입니다. `all_signals`, `all_data`, `df_xgb_pred` 등의 초기화가 없거나 사용이 앞서고 실행 count가 비순차입니다. 새 커널의 Run All로 과거 출력이 복원되는 Notebook이라고 주장하지 않습니다. 셀을 재배열해 결과를 소급 변경하지 않았습니다.

## 이번 검증 결과

2026-09-10 정적 소스 검사와 입력 검사기 단위 테스트14개가 통과했습니다. 실제 입력 검사에서는 AEMD schema·날짜와 GME 열 수 결함을 보고하고 예상대로 exit1을 반환했습니다. 파생 PDF는 원본 보존, 민감 텍스트·이미지 제거, 메타데이터,19쪽 렌더를 확인했습니다.

FinBERT 새 추론, LSTM/XGBoost 재학습, yfinance 재수집, 시간순 holdout 실험, 전체 모델 성능 재현은 실행하지 않았습니다. 검사 도구의 성공을 원본 모델 실행 성공으로 표현하지 않습니다.
