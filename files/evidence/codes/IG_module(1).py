# IG_module.py
# A.X-4.0 Light 기반 낚시성 기사 토큰 기여도(Integrated Gradients) 전용 모듈.

from typing import Dict, Any, Tuple, List

import torch
import numpy as np

# 점수 산정용 공통 로직은 기존 모듈에서 가져온다.
try:
    from ax4_clickbait_scorer import init_model, _build_prefix_ids_for_binary, build_article_text
except ImportError:
    from ax4_clickbait_score import init_model, _build_prefix_ids_for_binary, build_article_text  # type: ignore

# 모델 / 토크나이저 / 디바이스는 score 모듈과 완전히 공유한다.
_MODEL, _TOKENIZER, _DEVICE, _DTYPE = init_model()

ARTICLE_START = "[본문]"
ARTICLE_END = "이 기사가 낚시성이 강하면"


def _make_attention_mask(prefix_ids: torch.Tensor) -> torch.Tensor:
    """
    prefix_ids (1, L)에 대해 attention mask를 만든다.
    """
    if _TOKENIZER.pad_token_id is not None:
        attention_mask = (prefix_ids != _TOKENIZER.pad_token_id).long()
    else:
        attention_mask = torch.ones_like(prefix_ids)
    return attention_mask


def analyze_article_with_ig(article: Dict[str, Any], m_steps: int = 50) -> Dict[str, Any]:
    """
    한 기사에 대해 Integrated Gradients (IG)를 사용해
      - p0, p1 (다음 토큰이 0/1일 확률)
      - score_logit_diff = logit(1) - logit(0)
      - 토큰별 IG 기여도 (token_scores)
    를 한 번에 계산해서 반환한다.

    Args:
        article: newsCategory/newsTitle/newsContent 필드를 가진 기사 JSON dict
        m_steps: IG 적분 스텝 수 (클수록 정확하지만 느려짐)
    """
    # 1) prefix 토큰 시퀀스
    prefix_ids = _build_prefix_ids_for_binary(article)  # [1, L] (이미 _DEVICE에 올라가 있다고 가정)
    attention_mask = _make_attention_mask(prefix_ids).to(_DEVICE)

    # 2) 0/1 토큰 id
    id0 = _TOKENIZER("0", add_special_tokens=False).input_ids[0]
    id1 = _TOKENIZER("1", add_special_tokens=False).input_ids[0]

    # 3) 임베딩 레이어에서 원본 입력 임베딩(real_embeds) 얻기
    emb_layer = _MODEL.get_input_embeddings()
    with torch.no_grad():
        real_embeds = emb_layer(prefix_ids)  # [1, L, d]
    real_embeds = real_embeds.to(_DEVICE)

    # 4) 원본 입력(alpha=1.0)에 대한 p0, p1, score 계산
    with torch.no_grad():
        outputs = _MODEL(
            inputs_embeds=real_embeds,
            attention_mask=attention_mask,
        )
        last_logits = outputs.logits[0, -1, :]  # [V]

    two = torch.stack([last_logits[id0], last_logits[id1]], dim=0)  # [2]
    probs = torch.softmax(two, dim=0)
    p0 = probs[0].item()
    p1 = probs[1].item()
    score_logit_diff = (last_logits[id1] - last_logits[id0]).item()

    # 5) Integrated Gradients (IG) 계산
    baseline_embeds = torch.zeros_like(real_embeds)
    accumulated_grads = torch.zeros_like(real_embeds)

    alphas = torch.linspace(0.0, 1.0, steps=m_steps).to(_DEVICE)

    for alpha in alphas:
        interpolated_embeds = baseline_embeds + alpha * (real_embeds - baseline_embeds)
        interpolated_embeds.requires_grad_(True)

        _MODEL.zero_grad()

        ig_outputs = _MODEL(
            inputs_embeds=interpolated_embeds,
            attention_mask=attention_mask,
        )
        ig_last_logits = ig_outputs.logits[0, -1, :]

        ig_score = ig_last_logits[id1] - ig_last_logits[id0]

        ig_score.backward()

        accumulated_grads += interpolated_embeds.grad.detach()

    # 6) 평균 그래디언트, IG = (real - baseline) * 평균 grad
    avg_grads = accumulated_grads / float(m_steps)
    ig_attrib = (real_embeds - baseline_embeds) * avg_grads  # [1, L, d]

    # 🔧 여기서 bfloat16 → float32 변환 (numpy가 지원하는 dtype)
    ig_attrib = ig_attrib.to(torch.float32)

    # 7) 토큰별 스칼라 점수: 임베딩 차원에 대해 합산
    token_scores = ig_attrib.sum(dim=-1).squeeze(0)  # [L]
    token_scores = token_scores.detach().cpu().numpy()

    token_ids = prefix_ids[0].tolist()
    tokens = _TOKENIZER.convert_ids_to_tokens(token_ids)

    return {
        "p0": p0,
        "p1": p1,
        "score_logit_diff": score_logit_diff,
        "p_clickbait": p1,
        "tokens": tokens,
        "token_ids": token_ids,
        "token_scores": token_scores,
    }


def aggregate_to_words(attrib_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    토큰 단위 IG 점수를 "단어" 단위로 합쳐주는 헬퍼.
    - build_article_text에서 사용한 [본문] ~ "이 기사가 낚시성이 강하면 ..." 구간만 사용
    """
    tokens: List[str] = attrib_result["tokens"]
    token_ids: List[int] = attrib_result["token_ids"]
    scores: np.ndarray = np.asarray(attrib_result["token_scores"])

    words: List[Tuple[str, float]] = []
    cur_text = ""
    cur_score = 0.0

    running_text = ""
    in_article = False
    finished = False

    for tok, tid, s in zip(tokens, token_ids, scores):
        piece = _TOKENIZER.decode([tid])
        running_text += piece

        # [본문] 이전 토큰은 무시
        if not in_article:
            if ARTICLE_START in running_text:
                in_article = True
            else:
                continue

        # 지시문이 시작되면 본문 종료
        if ARTICLE_END in running_text:
            finished = True
        if finished:
            break

        # 대괄호 마커는 단어에서 제외
        if "[" in piece or "]" in piece:
            if cur_text:
                words.append((cur_text, cur_score))
                cur_text, cur_score = "", 0.0
            continue

        piece_strip = piece.strip()
        if not piece_strip:
            continue

        new_word = tok.startswith("▁") or piece.startswith(" ")

        if new_word:
            if cur_text:
                words.append((cur_text, cur_score))
            cur_text = piece_strip
            cur_score = float(s)
        else:
            cur_text += piece_strip
            cur_score += float(s)

    if cur_text:
        words.append((cur_text, cur_score))

    word_tokens = [w for (w, _) in words]
    word_scores = [float(v) for (_, v) in words]

    result = dict(attrib_result)
    result["word_tokens"] = word_tokens
    result["word_scores"] = word_scores
    return result


def print_top_words(attrib_result: Dict[str, Any], top_k: int = 20) -> None:
    """
    aggregate_to_words 결과를 이용해 IG 절댓값 기준 상위 단어를 출력하는 유틸.
    """
    agg = aggregate_to_words(attrib_result)
    pairs = list(zip(agg["word_tokens"], agg["word_scores"]))

    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)

    print(f"=== Top {top_k} words by |IG score| ===")
    for word, score in pairs_sorted[:top_k]:
        print(f"{word}\t{score:.4f}")


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 2:
        print("Usage: python IG_module.py /path/to/article.json")
        sys.exit(0)

    path = sys.argv[1]

    with open(path, "r", encoding="utf-8") as f:
        article = json.load(f)

    attrib = analyze_article_with_ig(article, m_steps=50)
    print_top_words(attrib, top_k=30)
