# -*- coding: utf-8 -*-
"""
장기 어휘 드리프트(JSD + CUSUM) 파이프라인 (일 단위 count 재활용 + 부트스트랩 보강)

추가 반영:
- 윈도우 내 샘플(기사 수 / nonzero day)이 부족하면 bootstrap(with replacement)로 보강
- bootstrap 발생 row는 row_color="RED"
- dup_rate(중복률) 기록
- 윈도우 샘플 수/토큰 수/nonzero day 수를 “배경 정보”로 CSV에 함께 기록

CUSUM:
S_t = max(0, S_{t-1} + (JSD_t - tau_noise))
tau_noise = 0.0036 (고정)
"""

import os
import json
import math
import csv
import importlib.util
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

# -------------------------
# ✅ 0) 하드코딩 설정값 (여기만 바꿔서 사용)
# -------------------------

# (A) 입력 time_map jsonl (각 줄에 abs_path(or path), dt_date가 있어야 함)
TIME_MAP_JSONL_PATH = r"C:\TRAITHON\data\146.낚시성 기사 탐지 데이터\validation_file_time_map.jsonl"

# (B) 전처리기
PREPROCESS_PY_PATH = r"C:\Users\gywnd\OneDrive\바탕 화면\clickbait_preprocess.py"

# (C) 토크나이저 로컬 경로
TOKENIZER_DIR = r"C:\TRAITHON\model\A.X-4.0-Light"

# (D) 출력 루트
OUT_ROOT = r"C:\TRAITHON\outputs\jsd_final"

# (E) 전처리+시간정렬 기사 jsonl 출력
OUT_ARTICLES_JSONL = os.path.join(OUT_ROOT, "articles_sorted_by_date.jsonl")

# (F) 일 단위 count 배열 저장 디렉토리
OUT_DAILY_COUNTS_DIR = os.path.join(OUT_ROOT, "daily_counts")

# (F-2) 일 단위 “문서수/토큰수” 메타(재활용)
OUT_DAILY_META_CSV = os.path.join(OUT_DAILY_COUNTS_DIR, "_daily_meta.csv")
# columns: dt_date, n_docs, n_tokens

# (G) 분석 결과 CSV
OUT_METRICS_CSV = os.path.join(OUT_ROOT, "jsd_cusum_metrics.csv")

# ---- 토크나이징/카운트 옵션 ----
MAX_TOKENS = 4096
USE_TITLE_AND_BODY_SEP = "\n"

# ---- 윈도우 옵션 ----
DATE_START = "2021-01-01"
DATE_END   = "2022-08-24"
WINDOW_DAYS = 56
STRIDE_DAYS = 7

# ---- CUSUM 옵션 ----
TAU_NOISE = 0.0036
CUSUM_H = 0.020  # 바꿔가면서 테스트

# ---- 부트스트랩 옵션(핵심) ----
BOOTSTRAP_ENABLED = True

# (1) 윈도우가 "너무 빈약"하다고 판단하는 기준
MIN_NONZERO_DAYS_PER_WINDOW = 10   # 윈도우 내 실제 데이터 존재한 일수 (doc>0인 날짜 수)
MIN_DOCS_PER_WINDOW = 400          # 윈도우 내 기사 수(문서 수)

# (2) 보강 목표: 기사 수를 최소 이만큼까지 “복제”로 끌어올림
TARGET_DOCS_PER_WINDOW = 800

# (3) 무한루프 방지용
MAX_BOOTSTRAP_EXTRA_DRAWS = 5000

# (4) 부트스트랩 random seed (윈도우 시작일에 따라 결정적으로 돌아가게)
BOOTSTRAP_SEED = 42

# ---- 기타 ----
SKIP_IF_MISSING_DT = True
SKIP_IF_CATEGORY_ETC = True
ENCODING = "utf-8"

# ---- 주당 고정 샘플링 옵션 ----
USE_WEEKLY_BALANCED = True

SECTIONS = ["PO", "EC", "SO", "LC", "GB", "IS", "ET"]
WEEKLY_TOTAL = 2100
PER_SECTION = WEEKLY_TOTAL // len(SECTIONS)

OUT_WEEKLY_COUNTS_DIR = os.path.join(OUT_ROOT, "weekly_counts")
OUT_WEEKLY_META_CSV   = os.path.join(OUT_WEEKLY_COUNTS_DIR, "_weekly_meta.csv")

# 윈도우도 '주'로 굴릴 거면 이렇게 쓰는 게 더 직관적
WINDOW_WEEKS = 8      # 4주(=28일) 추천
STRIDE_WEEKS = 1      # 1주 이동

# -------------------------
# 1) 유틸
# -------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def parse_yyyy_mm_dd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def daterange(d0: date, d1: date):
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)

def jsonl_iter(path: str):
    with open(path, "r", encoding=ENCODING) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def jsonl_write(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding=ENCODING) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def dynamic_import_module(py_path: str, module_name: str = "cb_preprocess"):
    py_path = str(py_path)
    if not os.path.isfile(py_path):
        raise FileNotFoundError(f"PREPROCESS_PY_PATH not found: {py_path}")
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import preprocess module from: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def safe_read_json(abs_path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(abs_path, "r", encoding=ENCODING) as f:
            return json.load(f)
    except Exception:
        return None

def daily_count_path(out_dir: str, d: date) -> str:
    return os.path.join(out_dir, f"{d.isoformat()}.npy")

def jsd_from_count_vectors(c1: np.ndarray, c2: np.ndarray, eps: float = 1e-12) -> float:
    s1 = float(c1.sum())
    s2 = float(c2.sum())
    if s1 <= 0 or s2 <= 0:
        return float("nan")

    p = c1 / s1
    q = c2 / s2
    m = 0.5 * (p + q)

    mask = m > 0
    p_m = p[mask]
    q_m = q[mask]
    m_m = m[mask]

    def kl(a, b):
        a_mask = a > 0
        aa = a[a_mask]
        bb = b[a_mask]
        return float(np.sum(aa * np.log((aa + eps) / (bb + eps))))

    return 0.5 * kl(p_m, m_m) + 0.5 * kl(q_m, m_m)

# -------------------------
# 2) 기사 전처리 + 날짜 오름차순 jsonl 생성
# -------------------------

def build_sorted_articles_jsonl(time_map_jsonl: str, preprocess_py_path: str, out_jsonl: str) -> Tuple[int, int]:
    cbp = dynamic_import_module(preprocess_py_path, module_name="cb_preprocess")

    rows = []
    n_total = 0
    n_kept = 0

    for r in jsonl_iter(time_map_jsonl):
        n_total += 1

        # 키가 abs_path일 수도, path일 수도 있음
        abs_path = r.get("abs_path") or r.get("path")
        dt_date = r.get("dt_date") or r.get("date")

        if not abs_path or not isinstance(abs_path, str):
            continue

        if dt_date is None:
            if SKIP_IF_MISSING_DT:
                continue
        else:
            try:
                _ = parse_yyyy_mm_dd(dt_date)
            except Exception:
                if SKIP_IF_MISSING_DT:
                    continue

        raw = safe_read_json(abs_path)
        if raw is None:
            continue

        flat = cbp.preprocess_article(raw)

        if SKIP_IF_CATEGORY_ETC and flat.get("newsCategory") == "ETC":
            continue

        out = {
            "dt_date": dt_date,
            "dt_iso": r.get("dt_iso"),
            "abs_path": abs_path,
            "rel_path": r.get("rel_path"),
            "newsID": flat.get("newsID"),
            "newsCategory": flat.get("newsCategory"),
            "newsTitle": flat.get("newsTitle"),
            "newsContent": flat.get("newsContent"),
            "clickbaitClass": flat.get("clickbaitClass"),
            "match_status": r.get("match_status"),
            "dt_source": r.get("dt_source"),
        }
        rows.append(out)
        n_kept += 1

    def sort_key(x):
        d = x.get("dt_date")
        if d is None:
            return (1, "9999-99-99")
        return (0, d)

    rows.sort(key=sort_key)

    ensure_dir(str(Path(out_jsonl).parent))
    jsonl_write(out_jsonl, rows)
    return n_total, n_kept

# -------------------------
# 3) 일 단위 count 배열 + (doc/token meta) 저장
# -------------------------

def build_daily_counts_from_articles_jsonl(articles_jsonl: str, tokenizer_dir: str, out_daily_dir: str) -> Tuple[int, int]:
    ensure_dir(out_daily_dir)

    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=False)

    vocab_size = int(tok.vocab_size)

    cur_date: Optional[str] = None
    token_ids_bucket: List[int] = []
    doc_count_bucket: int = 0

    n_docs = 0
    n_days = 0
    daily_meta_rows: List[Dict[str, Any]] = []

    def flush_bucket(date_str: str, token_ids: List[int], n_docs_day: int):
        nonlocal n_days
        if n_docs_day <= 0:
            return
        d = parse_yyyy_mm_dd(date_str)
        arr = np.bincount(np.array(token_ids, dtype=np.int64), minlength=vocab_size).astype(np.int64)
        np.save(daily_count_path(out_daily_dir, d), arr)
        daily_meta_rows.append({
            "dt_date": d.isoformat(),
            "n_docs": int(n_docs_day),
            "n_tokens": int(arr.sum()),
        })
        n_days += 1

    for r in jsonl_iter(articles_jsonl):
        dt_date = r.get("dt_date")
        if dt_date is None:
            continue

        title = (r.get("newsTitle") or "").strip()
        body = (r.get("newsContent") or "").strip()
        text = title + USE_TITLE_AND_BODY_SEP + body

        enc = tok(
            text,
            truncation=True,
            max_length=MAX_TOKENS,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        ids = enc.get("input_ids", [])
        if not ids:
            continue

        if cur_date is None:
            cur_date = dt_date

        if dt_date != cur_date:
            flush_bucket(cur_date, token_ids_bucket, doc_count_bucket)
            token_ids_bucket = []
            doc_count_bucket = 0
            cur_date = dt_date

        token_ids_bucket.extend(ids)
        doc_count_bucket += 1
        n_docs += 1

    if cur_date is not None:
        flush_bucket(cur_date, token_ids_bucket, doc_count_bucket)

    # meta 저장
    meta = {
        "tokenizer_dir": tokenizer_dir,
        "vocab_size": vocab_size,
        "max_tokens": MAX_TOKENS,
        "articles_jsonl": articles_jsonl,
        "created_at": datetime.now().isoformat(),
    }
    with open(os.path.join(out_daily_dir, "_meta.json"), "w", encoding=ENCODING) as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # daily meta CSV 저장 (재활용 핵심!)
    with open(OUT_DAILY_META_CSV, "w", newline="", encoding=ENCODING) as f:
        writer = csv.DictWriter(f, fieldnames=["dt_date", "n_docs", "n_tokens"])
        writer.writeheader()
        for row in daily_meta_rows:
            writer.writerow(row)

    return n_docs, n_days

def load_daily_meta(meta_csv: str) -> Dict[str, Dict[str, int]]:
    """
    return dict[dt_date] = {"n_docs": int, "n_tokens": int}
    """
    mp: Dict[str, Dict[str, int]] = {}
    if not os.path.isfile(meta_csv):
        return mp
    with open(meta_csv, "r", encoding=ENCODING) as f:
        reader = csv.DictReader(f)
        for r in reader:
            d = r["dt_date"]
            mp[d] = {
                "n_docs": int(r["n_docs"]),
                "n_tokens": int(r["n_tokens"]),
            }
    return mp

def week_start_monday(d: date) -> date:
    # 월요일을 주 시작으로 맞춤
    return d - timedelta(days=d.weekday())


def build_weekly_counts_balanced_from_articles_jsonl(
    articles_jsonl: str,
    tokenizer_dir: str,
    out_weekly_dir: str,
) -> Tuple[int, int]:
    """
    - 주 단위로 기사들을 모은 뒤
    - 섹션별 144개씩(총 1008) 샘플링
    - 부족하면 섹션 내부 bootstrap(with replacement)
    - 주 단위 토큰 count vector 저장 (week_start.npy)
    """
    ensure_dir(out_weekly_dir)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=True)
    vocab_size = int(tok.vocab_size)

    # week_bucket[week_start][section] = list[article_row]
    week_bucket: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    n_total_docs = 0
    for r in jsonl_iter(articles_jsonl):
        dt_date = r.get("dt_date")
        sec = r.get("newsCategory")
        if not dt_date or sec not in SECTIONS:
            continue

        d = parse_yyyy_mm_dd(dt_date)
        ws = week_start_monday(d).isoformat()

        week_bucket.setdefault(ws, {s: [] for s in SECTIONS})
        week_bucket[ws][sec].append(r)
        n_total_docs += 1

    # meta 저장용
    meta_rows = []
    n_weeks_saved = 0

    for ws, sec_map in sorted(week_bucket.items()):
        rng = np.random.default_rng(BOOTSTRAP_SEED + int(ws.replace("-", "")))

        selected_rows = []
        boot_applied = 0
        dup_ids = []

        base_total = 0
        final_total = 0

        for sec in SECTIONS:
            pool = sec_map.get(sec, [])
            base_n = len(pool)
            base_total += base_n

            if base_n == 0:
                # 아예 없으면 이 섹션은 못 채움 → 전체도 못 채움
                # (원하면 여기서 "다른 섹션에서 보충" 정책도 가능)
                boot_applied = 1
                continue

            if base_n >= PER_SECTION:
                idx = rng.choice(base_n, size=PER_SECTION, replace=False)
                chosen = [pool[i] for i in idx]
            else:
                # 부족 → bootstrap
                boot_applied = 1
                idx = rng.choice(base_n, size=PER_SECTION, replace=True)
                chosen = [pool[i] for i in idx]

            selected_rows.extend(chosen)
            final_total += len(chosen)

            # dup rate 측정용 id(경로 기준)
            dup_ids.extend([c.get("abs_path") or c.get("newsID") or "" for c in chosen])

        if final_total == 0:
            continue

        # dup rate 계산
        uniq = len(set([x for x in dup_ids if x]))
        dup_rate = 1.0 - (uniq / max(1, len(dup_ids)))

        # 토큰화 + 카운트
        token_ids = []
        for r in selected_rows:
            title = (r.get("newsTitle") or "").strip()
            body  = (r.get("newsContent") or "").strip()
            text = title + USE_TITLE_AND_BODY_SEP + body

            enc = tok(
                text,
                truncation=True,
                max_length=MAX_TOKENS,
                add_special_tokens=True,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            ids = enc.get("input_ids", [])
            if ids:
                token_ids.extend(ids)

        arr = np.bincount(np.array(token_ids, dtype=np.int64), minlength=vocab_size).astype(np.int64)
        np.save(os.path.join(out_weekly_dir, f"{ws}.npy"), arr)

        meta_rows.append({
            "week_start": ws,
            "base_total_docs": base_total,
            "final_total_docs": final_total,   # 보통 1008 근처
            "boot_applied": boot_applied,
            "dup_rate": float(dup_rate),
            "n_tokens": int(arr.sum()),
        })
        n_weeks_saved += 1

    with open(os.path.join(out_weekly_dir, "_meta.json"), "w", encoding=ENCODING) as f:
        json.dump({
            "tokenizer_dir": tokenizer_dir,
            "vocab_size": vocab_size,
            "max_tokens": MAX_TOKENS,
            "weekly_total": WEEKLY_TOTAL,
            "per_section": PER_SECTION,
            "created_at": datetime.now().isoformat(),
        }, f, ensure_ascii=False, indent=2)

    with open(OUT_WEEKLY_META_CSV, "w", newline="", encoding=ENCODING) as f:
        w = csv.DictWriter(f, fieldnames=["week_start","base_total_docs","final_total_docs","boot_applied","dup_rate","n_tokens"])
        w.writeheader()
        for row in meta_rows:
            w.writerow(row)

    return n_total_docs, n_weeks_saved


# -------------------------
# 4) 윈도우 sum 만들기 + (부트스트랩 옵션) + dup_rate 계산
# -------------------------

def window_days_list(all_days: List[date], start_idx: int, window_days: int) -> List[date]:
    return all_days[start_idx:start_idx + window_days]

def load_daily_counts(daily_dir: str, d: date, vocab_size: int) -> np.ndarray:
    p = daily_count_path(daily_dir, d)
    if os.path.isfile(p):
        arr = np.load(p)
        if arr.shape[0] != vocab_size:
            raise ValueError(f"Vocab mismatch: {p} shape={arr.shape}, vocab={vocab_size}")
        return arr.astype(np.int64, copy=False)
    return np.zeros((vocab_size,), dtype=np.int64)

def build_window_sum_with_bootstrap(
    daily_dir: str,
    daily_meta: Dict[str, Dict[str, int]],
    vocab_size: int,
    days_in_window: List[date],
    tau_seed_key: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    1) base window sum = 각 날짜 count 합
    2) base window가 너무 희소하면 -> bootstrap으로 추가 day를 복제해서 sum 보강
    3) meta로 샘플 수/dup_rate/부트스트랩 여부 반환
    """

    # base 합
    base_sum = np.zeros((vocab_size,), dtype=np.int64)
    base_docs = 0
    base_tokens = 0
    nonzero_days = 0

    # bootstrap 대상 pool(실제 데이터 있는 날짜)
    pool_days: List[date] = []
    weights: List[float] = []

    for d in days_in_window:
        key = d.isoformat()
        info = daily_meta.get(key, {"n_docs": 0, "n_tokens": 0})
        n_docs = int(info["n_docs"])
        n_tokens = int(info["n_tokens"])

        if n_docs > 0:
            nonzero_days += 1
            pool_days.append(d)
            weights.append(float(n_docs))

        base_docs += n_docs
        base_tokens += n_tokens

        if n_docs > 0:
            base_sum += load_daily_counts(daily_dir, d, vocab_size)

    need_bootstrap = False
    if BOOTSTRAP_ENABLED:
        if nonzero_days < MIN_NONZERO_DAYS_PER_WINDOW:
            need_bootstrap = True
        if base_docs < MIN_DOCS_PER_WINDOW:
            need_bootstrap = True

    meta = {
        "base_docs": base_docs,
        "base_tokens": base_tokens,
        "base_nonzero_days": nonzero_days,
        "boot_applied": 0,
        "boot_extra_draws": 0,
        "dup_rate_extra": 0.0,
        "dup_rate_total": 0.0,
        "final_docs": base_docs,
        "final_tokens": int(base_sum.sum()),
    }

    # 부트스트랩 불가(풀 자체가 없음)
    if not need_bootstrap or len(pool_days) == 0:
        return base_sum, meta

    # bootstrap 수행
    rng = np.random.default_rng(BOOTSTRAP_SEED + tau_seed_key)
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()

    target_docs = max(TARGET_DOCS_PER_WINDOW, base_docs)
    cur_sum = base_sum.copy()
    cur_docs = base_docs

    chosen_extra: List[str] = []
    extra_draws = 0

    while cur_docs < target_docs and extra_draws < MAX_BOOTSTRAP_EXTRA_DRAWS:
        idx = int(rng.choice(len(pool_days), p=w))
        d = pool_days[idx]
        key = d.isoformat()

        info = daily_meta.get(key, {"n_docs": 0, "n_tokens": 0})
        n_docs = int(info["n_docs"])

        cur_sum += load_daily_counts(daily_dir, d, vocab_size)
        cur_docs += n_docs
        chosen_extra.append(key)
        extra_draws += 1

    unique_extra = len(set(chosen_extra)) if chosen_extra else 0
    dup_rate_extra = 0.0
    if extra_draws > 0:
        dup_rate_extra = 1.0 - (unique_extra / float(extra_draws))

    # total dup rate: (base_nonzero_days + extra_draws)에서 unique가 얼마나 되나
    total_contrib = nonzero_days + extra_draws
    unique_total = len(set([d.isoformat() for d in pool_days] + chosen_extra))
    dup_rate_total = 0.0
    if total_contrib > 0:
        dup_rate_total = 1.0 - (unique_total / float(total_contrib))

    meta.update({
        "boot_applied": 1,
        "boot_extra_draws": int(extra_draws),
        "dup_rate_extra": float(dup_rate_extra),
        "dup_rate_total": float(dup_rate_total),
        "final_docs": int(cur_docs),
        "final_tokens": int(cur_sum.sum()),
    })
    return cur_sum, meta

# -------------------------
# 5) JSD + CUSUM 실행
# -------------------------

def run_jsd_cusum(
    daily_dir: str,
    tokenizer_dir: str,
    daily_meta_csv: str,
    date_start: str,
    date_end: str,
    window_days: int,
    stride_days: int,
    tau_noise: float,
    cusum_h: float,
    out_csv: str,
) -> None:
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=False)
    vocab_size = int(tok.vocab_size)

    daily_meta = load_daily_meta(daily_meta_csv)

    d0 = parse_yyyy_mm_dd(date_start)
    d1 = parse_yyyy_mm_dd(date_end)

    all_days = list(daterange(d0, d1))
    if len(all_days) < window_days + stride_days:
        raise ValueError("기간이 너무 짧아서 윈도우/stride를 굴릴 수 없음")

    ensure_dir(str(Path(out_csv).parent))

    # 초기 prev 윈도우
    prev_start = 0
    prev_days = window_days_list(all_days, prev_start, window_days)
    prev_ws = prev_days[0].isoformat()
    prev_sum, prev_meta = build_window_sum_with_bootstrap(
        daily_dir=daily_dir,
        daily_meta=daily_meta,
        vocab_size=vocab_size,
        days_in_window=prev_days,
        tau_seed_key=int(prev_days[0].strftime("%Y%m%d")),
    )

    S = 0.0
    t = 0

    with open(out_csv, "w", newline="", encoding=ENCODING) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "t",
                "prev_window_start", "prev_window_end",
                "cur_window_start", "cur_window_end",
                "jsd",
                "tau_noise",
                "cusum_S",
                "H",
                "state",

                # ✅ 배경 정보(샘플/토큰/일수)
                "prev_base_docs", "prev_final_docs",
                "prev_base_tokens", "prev_final_tokens",
                "prev_base_nonzero_days",
                "prev_boot_applied", "prev_boot_extra_draws",
                "prev_dup_rate_extra", "prev_dup_rate_total",

                "cur_base_docs", "cur_final_docs",
                "cur_base_tokens", "cur_final_tokens",
                "cur_base_nonzero_days",
                "cur_boot_applied", "cur_boot_extra_draws",
                "cur_dup_rate_extra", "cur_dup_rate_total",

                # ✅ 빨간 표시용 컬럼
                "row_color",
            ],
        )
        writer.writeheader()

        cur_start = prev_start + stride_days

        while cur_start + window_days - 1 < len(all_days):
            cur_days = window_days_list(all_days, cur_start, window_days)
            cur_sum, cur_meta = build_window_sum_with_bootstrap(
                daily_dir=daily_dir,
                daily_meta=daily_meta,
                vocab_size=vocab_size,
                days_in_window=cur_days,
                tau_seed_key=int(cur_days[0].strftime("%Y%m%d")),
            )

            jsd = jsd_from_count_vectors(prev_sum, cur_sum)

            if not (jsd is None or math.isnan(jsd)):
                S = max(0.0, S + (jsd - tau_noise))

            if S >= cusum_h:
                state = "ALERT"
            elif S > 0:
                state = "WATCH"
            else:
                state = "CLEAR"

            # ✅ 빨간 표시 규칙: 부트스트랩이 “prev 또는 cur”에서 발생한 step이면 RED
            row_color = "RED" if (prev_meta["boot_applied"] == 1 or cur_meta["boot_applied"] == 1) else "OK"

            prev_ws = prev_days[0].isoformat()
            prev_we = prev_days[-1].isoformat()
            cur_ws = cur_days[0].isoformat()
            cur_we = cur_days[-1].isoformat()

            writer.writerow({
                "t": t,
                "prev_window_start": prev_ws,
                "prev_window_end": prev_we,
                "cur_window_start": cur_ws,
                "cur_window_end": cur_we,
                "jsd": jsd,
                "tau_noise": tau_noise,
                "cusum_S": S,
                "H": cusum_h,
                "state": state,

                "prev_base_docs": prev_meta["base_docs"],
                "prev_final_docs": prev_meta["final_docs"],
                "prev_base_tokens": prev_meta["base_tokens"],
                "prev_final_tokens": prev_meta["final_tokens"],
                "prev_base_nonzero_days": prev_meta["base_nonzero_days"],
                "prev_boot_applied": prev_meta["boot_applied"],
                "prev_boot_extra_draws": prev_meta["boot_extra_draws"],
                "prev_dup_rate_extra": prev_meta["dup_rate_extra"],
                "prev_dup_rate_total": prev_meta["dup_rate_total"],

                "cur_base_docs": cur_meta["base_docs"],
                "cur_final_docs": cur_meta["final_docs"],
                "cur_base_tokens": cur_meta["base_tokens"],
                "cur_final_tokens": cur_meta["final_tokens"],
                "cur_base_nonzero_days": cur_meta["base_nonzero_days"],
                "cur_boot_applied": cur_meta["boot_applied"],
                "cur_boot_extra_draws": cur_meta["boot_extra_draws"],
                "cur_dup_rate_extra": cur_meta["dup_rate_extra"],
                "cur_dup_rate_total": cur_meta["dup_rate_total"],

                "row_color": row_color,
            })

            # 다음 step
            prev_sum = cur_sum
            prev_meta = cur_meta
            prev_days = cur_days

            cur_start += stride_days
            t += 1

# -------------------------
# 6) main
# -------------------------

def main():
    ensure_dir(OUT_ROOT)

    # (1) 기사 jsonl 생성
    if not os.path.isfile(OUT_ARTICLES_JSONL):
        n_total, n_kept = build_sorted_articles_jsonl(
            time_map_jsonl=TIME_MAP_JSONL_PATH,
            preprocess_py_path=PREPROCESS_PY_PATH,
            out_jsonl=OUT_ARTICLES_JSONL,
        )
        print(f"[articles] total={n_total}, kept={n_kept}")
    else:
        print(f"[articles] exists: {OUT_ARTICLES_JSONL}")

    # (2) 주 단위 count + meta 생성
    if USE_WEEKLY_BALANCED:
        if not os.path.isfile(OUT_WEEKLY_META_CSV):
            n_docs, n_weeks = build_weekly_counts_balanced_from_articles_jsonl(
                articles_jsonl=OUT_ARTICLES_JSONL,
                tokenizer_dir=TOKENIZER_DIR,
                out_weekly_dir=OUT_WEEKLY_COUNTS_DIR,
            )
            print(f"[weekly_counts] docs_seen={n_docs}, weeks_saved={n_weeks}")
        else:
            print(f"[weekly_counts] exists: {OUT_WEEKLY_COUNTS_DIR}")


    # (3) JSD + CUSUM 실행
    run_jsd_cusum(
        daily_dir=OUT_DAILY_COUNTS_DIR,
        tokenizer_dir=TOKENIZER_DIR,
        daily_meta_csv=OUT_DAILY_META_CSV,
        date_start=DATE_START,
        date_end=DATE_END,
        window_days=WINDOW_DAYS,
        stride_days=STRIDE_DAYS,
        tau_noise=TAU_NOISE,
        cusum_h=CUSUM_H,
        out_csv=OUT_METRICS_CSV,
    )
    print(f"[done] metrics saved: {OUT_METRICS_CSV}")
    
    plot_path = os.path.join(OUT_ROOT, "jsd_cusum_plot.png")
    plot_jsd_cusum(OUT_METRICS_CSV, plot_path)
    print(f"[plot] saved: {plot_path}")


# -------------------------
# 7) 시각화 (JSD / CUSUM + bootstrap RED 표시 + 샘플 수 배경 bar)
# -------------------------

def load_metrics_for_plot(metrics_csv: str):
    """
    metrics CSV 읽어서 플롯용 배열로 반환
    x축은 cur_window_end(YYYY-MM-DD) 사용
    """
    xs = []
    jsd = []
    S = []
    row_color = []
    cur_base_docs = []
    cur_final_docs = []
    cur_base_nonzero_days = []
    cur_dup_rate_total = []

    with open(metrics_csv, "r", encoding=ENCODING) as f:
        reader = csv.DictReader(f)
        for r in reader:
            x = r["cur_window_end"]
            xs.append(parse_yyyy_mm_dd(x))
            jsd.append(float(r["jsd"]) if r["jsd"] not in ("", "nan", "NaN") else float("nan"))
            S.append(float(r["cusum_S"]) if r["cusum_S"] not in ("", "nan", "NaN") else float("nan"))
            row_color.append(r.get("row_color", "OK"))

            cur_base_docs.append(int(r.get("cur_base_docs", 0)))
            cur_final_docs.append(int(r.get("cur_final_docs", 0)))
            cur_base_nonzero_days.append(int(r.get("cur_base_nonzero_days", 0)))
            cur_dup_rate_total.append(float(r.get("cur_dup_rate_total", 0.0)))

    return {
        "x": np.array(xs),
        "jsd": np.array(jsd, dtype=np.float64),
        "S": np.array(S, dtype=np.float64),
        "row_color": np.array(row_color),
        "cur_base_docs": np.array(cur_base_docs, dtype=np.int64),
        "cur_final_docs": np.array(cur_final_docs, dtype=np.int64),
        "cur_base_nonzero_days": np.array(cur_base_nonzero_days, dtype=np.int64),
        "cur_dup_rate_total": np.array(cur_dup_rate_total, dtype=np.float64),
    }


def plot_jsd_cusum(metrics_csv: str, out_png: str):
    """
    - 상단: JSD line + tau_noise line + bootstrap RED 점
    - 하단: CUSUM S line + H line + bootstrap RED 점
    - 배경 bar: cur_base_docs (윈도우 샘플 수)
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    d = load_metrics_for_plot(metrics_csv)

    x = d["x"]
    jsd = d["jsd"]
    S = d["S"]
    row_color = d["row_color"]
    cur_base_docs = d["cur_base_docs"]

    # bootstrap 표시 마스크
    red_mask = (row_color == "RED")

    # 2행 1열 subplot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # -----------------
    # (1) JSD plot
    # -----------------
    ax1 = axes[0]
    ax1.plot(x, jsd, linewidth=1.5, label="JSD")
    ax1.axhline(TAU_NOISE, linestyle="--", linewidth=1.2, label=f"tau_noise={TAU_NOISE}")

    # bootstrap step: 빨간 점
    ax1.scatter(x[red_mask], jsd[red_mask], color="red", s=28, label="bootstrap window")

    ax1.set_ylabel("JSD")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # 배경 bar: 샘플 수(cur_base_docs)
    ax1b = ax1.twinx()
    ax1b.bar(x, cur_base_docs, alpha=0.15, width=5)  # width=5 days 정도
    ax1b.set_ylabel("cur_base_docs (window sample size)")
    ax1b.set_ylim(0, max(1, int(cur_base_docs.max() * 1.15)))

    # -----------------
    # (2) CUSUM plot
    # -----------------
    ax2 = axes[1]
    ax2.plot(x, S, linewidth=1.8, label="CUSUM S")
    ax2.axhline(CUSUM_H, linestyle="--", linewidth=1.2, label=f"H={CUSUM_H}")

    ax2.scatter(x[red_mask], S[red_mask], color="red", s=28, label="bootstrap window")

    ax2.set_ylabel("CUSUM S")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    # 배경 bar도 같이 깔기(같은 샘플 수 기준)
    ax2b = ax2.twinx()
    ax2b.bar(x, cur_base_docs, alpha=0.15, width=5)
    ax2b.set_ylabel("cur_base_docs (window sample size)")
    ax2b.set_ylim(0, max(1, int(cur_base_docs.max() * 1.15)))

    # -----------------
    # x축 포맷
    # -----------------
    ax2.set_xlabel("cur_window_end (date)")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

    fig.suptitle("Long-term Lexical Drift: JSD + CUSUM (bootstrap marked in RED)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    ensure_dir(str(Path(out_png).parent))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    main()