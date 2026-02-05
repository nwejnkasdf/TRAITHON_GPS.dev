
# Dataset Source Log — emergency_news_yonhap_v1_2026

## Dataset Snapshot ID
emergency_news_yonhap_v1_2026

## Source (Org / Portal)
- 재난안전데이터 공유 플랫폼
- 연합뉴스 (dataSn=46)
- Contact: 02-398-3556

## Collection Method & Window
- 시스템 연계 기반 자동 수집
- 갱신주기: 5분
- 최근 갱신: 2026-01-13 (KST)

## Annotation (Guideline / IRR / QA)
- 공식 언론 보도 원문 사용 (라벨링 없음)
- 재난 정보 전달 목적 데이터
- QA: 수집 누락, 중복, 필드 정합성 점검

## Preprocessing Steps
1. CSV 포맷 검증 및 스키마 고정
2. 중복 기사 제거 (ID·제목·본문 해시)
3. 재난 유형 태깅(산사태, 지진, 감염병 등)
4. 제목·본문 텍스트 정규화
5. 시간·유형 메타데이터 정합성 확인

## Licensing & Rights (SPDX / Notes)
- 공공 공개 데이터
- 대국민 공개 / 연구·분석 목적 사용
- 저작권: 연합뉴스 원문 유지
- SPDX: CC-BY 계열 준용(출처 명시 필수)

## Maintainer (Role / Contact)
- 연합뉴스 데이터 담당 부서
- 02-398-3556

## Version (Tag / Date / Changes)
- v1.0 (2026-01-13)
- 재난 기사 스트레스 테스트용 스냅샷 정의
