# 뉴스 입력 계약

사용 권한이 있는 뉴스만 `data/raw/{TICKER}_augmented.csv`에 준비합니다. TICKER는 AEMD, APPB, GME, NBDR, TRBO입니다. 이 디렉터리는 Git에서 제외됩니다.

필수 열은 대소문자를 구분하는 `Date,Headline,Summary`입니다. Date는 YYYY-MM-DD 형식이고, 제목·요약 중 하나는 비어 있지 않아야 합니다. Source URL 등 추가 열은 허용하지만 중복 헤더나 행별로 다른 열 수는 오류입니다. 쉼표·줄바꿈이 있는 필드는 CSV 규칙대로 따옴표로 묶어야 합니다.

```bash
python scripts/validate_news.py --data-dir data/raw
```

원본 입력에서 실제 오류가 발견되어 현재 자료를 그대로 복사한 학습은 재현되지 않습니다. [원본59행·증강280레코드와 기간/schema 감사](../docs/dataset.md)를 먼저 확인하세요. 샘플을 임의로 고쳐 원래 실험 입력으로 취급하지 않습니다. 주가 거래량은 이 CSV에 없으며 원래 스크립트가 yfinance에서 별도로 요청합니다.
