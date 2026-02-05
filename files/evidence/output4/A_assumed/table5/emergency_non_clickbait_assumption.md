# Emergency Non-Clickbait Assumption (Non-clickbait-by-design)

- Dataset: 재난안전데이터 공유 플랫폼을 통해 국가(공공) 목적으로 공개된 기사/텍스트를 사용하였다.
- 본 실험의 목적은 개별 기사에 대한 정답 라벨을 검증하는 것이 아니라,
  '의미적으로 정당한 긴급 정보 맥락'에서 모델이 표면적 낚시성 신호에 과잉 반응(over-triggering)하는지 측정하는 것이다.
- 따라서 본 실험에서는 전체 입력을 비낚시성(non-clickbait-by-design)으로 간주하고,
  모델이 산출한 clickbait 확률(p1) 및 logit(1)-logit(0)의 분포/비율을 스트레스 테스트 지표로 사용한다.
- Slice name: Emergency_NationalSafety
- Thresholds: p1 >= 0.5, logit_diff > 0.0
