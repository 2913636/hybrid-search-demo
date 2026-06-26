"""
混合检索演示：向量检索 + BM25 -> RRF 融合 -> Cross-Encoder 重排序
==============================================================
完整演示混合检索的四个阶段，输出对比结果。

运行：py hybrid.py [关键词]
"""

import sys
import re
import math


def tokenize(text: str) -> list[str]:
    """简单分词：中文按字符+词组，英文按空格"""
    # 提取中文词（2-3字组合）
    chinese_chars = re.findall(r'[一-鿿]+', text)
    tokens = []
    for chars in chinese_chars:
        tokens.append(chars)  # 整个中文片段
        tokens.extend(chars)  # 单字
        tokens.extend(chars[i:i+2] for i in range(len(chars)-1))  # 双字
    # 英文小写
    tokens.extend(w.lower() for w in re.findall(r'[a-zA-Z]+', text))
    return tokens


# ── 文档库 ────────────────────────────

DOCS = [
    "特斯拉 Model 2 售价 2.5 万美元，续航 500 公里。",
    "特斯拉的入门车型定价很有竞争力，约 2.5 万美元。",
    "比亚迪海鸥全球畅销，2026年 Q1 销量突破 100 万辆。",
    "Model 2 是特斯拉最便宜车型，2026年 Q1 开始交付。",
    "充电桩国标 GB/T 20234 已更新，2026 年实施新标准。",
    "比亚迪海鸥售价约 1.5 万美元，续航 400 公里，面向低端市场。",
    "新能源汽车市场竞争激烈，各品牌纷纷降价促销。",
    "MCP 协议定义三种传输方式：stdio、SSE、Streamable HTTP。",
    "特斯拉 2026 年计划推出 Model 2 和 Cybertruck 量产版。",
    "AI Agent 开发框架包括 LangGraph、CrewAI 和 OpenAI Agents SDK。",
]

DOC_TOKENS = [tokenize(d) for d in DOCS]


# ── 阶段 1：向量检索 ──────────────────

def vector_search(query: str, top_k: int = 3) -> list[tuple[float, str, int]]:
    """
    向量检索：用 Jaccard 相似度模拟语义搜索。
    真实场景替换为 ChromaDB / embedding 查询。
    """
    q_tokens = set(tokenize(query))
    scores = []
    for i, doc_tokens in enumerate(DOC_TOKENS):
        d_set = set(doc_tokens)
        if not q_tokens or not d_set:
            continue
        jaccard = len(q_tokens & d_set) / len(q_tokens | d_set)
        if jaccard > 0:
            scores.append((round(jaccard, 3), DOCS[i], i))
    scores.sort(reverse=True)
    return scores[:top_k]


# ── 阶段 2：BM25 关键词检索 ────────────

def bm25_search(query: str, top_k: int = 3, k1: float = 1.5, b: float = 0.75) -> list[tuple[float, str, int]]:
    """
    BM25 关键词检索：对精确关键词匹配效果好。
    """
    q_terms = tokenize(query)
    n = len(DOCS)
    avgdl = sum(len(dt) for dt in DOC_TOKENS) / max(n, 1)

    # IDF 计算
    idf = {}
    for term in q_terms:
        df = sum(1 for dt in DOC_TOKENS if term in dt)
        idf[term] = math.log((n - df + 0.5) / (df + 0.5) + 1)

    # BM25 评分
    scores = []
    for idx, doc_tokens in enumerate(DOC_TOKENS):
        dl = len(doc_tokens)
        score = 0.0
        for term in q_terms:
            if term not in idf:
                continue
            tf = doc_tokens.count(term)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / avgdl)
            score += idf[term] * numerator / max(denominator, 0.1)
        if score > 0:
            scores.append((round(score, 3), DOCS[idx], idx))
    scores.sort(reverse=True)
    return scores[:top_k]


# ── 阶段 3：RRF 融合 ──────────────────

def rrf_fusion(results_a: list, results_b: list, k: int = 60) -> list[tuple[float, str, int]]:
    """
    RRF (Reciprocal Rank Fusion)：融合两个排序列表。
    公式：score(d) = sum(1 / (k + rank_i(d))) for i in [A, B]
    优点：无需调参，自动平衡两个信号。
    """
    scores = {}
    doc_map = {}  # idx -> text
    for rank, (_, text, idx) in enumerate(results_a):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
        doc_map[idx] = text
    for rank, (_, text, idx) in enumerate(results_b):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
        doc_map[idx] = text
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(round(s, 4), doc_map[idx], idx) for idx, s in ranked]


# ── 阶段 4：Cross-Encoder 重排序 ───────

def rerank(query: str, candidates: list[tuple[float, str, int]], top_k: int = 3) -> list[tuple[float, str]]:
    """
    Cross-Encoder 重排序：对候选文档精细打分。
    真实场景用 Cross-Encoder 模型（如 bge-reranker），这里用词覆盖度模拟。
    """
    q_tokens = set(tokenize(query))
    scored = []
    for _, doc, _ in candidates:
        d_tokens = set(tokenize(doc))
        overlap = len(q_tokens & d_tokens) / max(len(q_tokens), 1)
        scored.append((round(overlap, 3), doc))
    scored.sort(reverse=True)
    return scored[:top_k]


# ── 演示 ──────────────────────────────

def demo(query: str):
    """完整演示混合检索四个阶段"""
    print(f"\nQuery: {query}")
    print("=" * 65)

    # 阶段 1
    vec_results = vector_search(query)
    print(f"\n[Stage 1] Vector Retrieval (Jaccard similarity):")
    for rank, (score, doc, _) in enumerate(vec_results, 1):
        print(f"  #{rank} [{score:.3f}] {doc[:60]}")

    # 阶段 2
    bm25_results = bm25_search(query)
    print(f"\n[Stage 2] BM25 Keyword Retrieval:")
    for rank, (score, doc, _) in enumerate(bm25_results, 1):
        print(f"  #{rank} [{score:.3f}] {doc[:60]}")

    # 阶段 3
    fused = rrf_fusion(vec_results, bm25_results)
    print(f"\n[Stage 3] RRF Fusion (k=60):")
    for rank, (score, doc, _) in enumerate(fused, 1):
        print(f"  #{rank} [{score:.4f}] {doc[:60]}")

    # 阶段 4
    final = rerank(query, fused[:5])
    print(f"\n[Stage 4] Cross-Encoder Rerank (Top 3):")
    for rank, (score, doc) in enumerate(final, 1):
        print(f"  #{rank} [{score:.3f}] {doc}")


if __name__ == "__main__":
    queries = sys.argv[1:] if len(sys.argv) > 1 else [
        "特斯拉 Model 2 售价",
        "比亚迪 销量",
        "MCP 协议",
    ]
    for q in queries:
        demo(q)
    print(f"\n{'='*65}")
    print("Tip: py hybrid.py 'your keyword' for custom queries")
