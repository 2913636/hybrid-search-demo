"""混合检索演示：向量+BM25→RRF→Cross-Encoder"""
import math

DOCS = [
    "特斯拉Model 2售价2.5万美元，续航500公里。",
    "特斯拉的入门车型定价很有竞争力。",
    "比亚迪海鸥全球畅销，Q1销量突破100万辆。",
    "Model 2是特斯拉最便宜的车型，售价约2.5万美元。",
    "充电桩国标GB/T 20234已更新。",
    "新能源汽车市场竞争激烈，各品牌降价促销。",
]

def vector_search(query, top_k=3):
    q = set(query)
    return sorted([(len(q&set(d))/len(q),d) for d in DOCS if q&set(d)], reverse=True)[:top_k]

def bm25_search(query, top_k=3):
    k, b = 1.5, 0.75; avgdl = sum(len(d) for d in DOCS)/len(DOCS)
    scored = []
    for d in DOCS:
        score = sum(d.count(t)*(k+1)/(d.count(t)+k*(1-b+b*len(d)/avgdl)) for t in query if t in d)
        if score > 0: scored.append((score, d))
    return sorted(scored, reverse=True)[:top_k]

def rrf_fusion(a, b, k=60):
    scores = {}
    for rank, (_, d) in enumerate(a): scores[d] = scores.get(d,0) + 1/(k+rank+1)
    for rank, (_, d) in enumerate(b): scores[d] = scores.get(d,0) + 1/(k+rank+1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def rerank(query, candidates, top_k=2):
    q = set(query)
    return sorted([(len(q&set(d))/len(d),d) for d in candidates], reverse=True)[:top_k]

if __name__ == "__main__":
    q = "特斯拉 售价"
    v = vector_search(q); b = bm25_search(q)
    print("向量:", [d[:20] for _,d in v])
    print("BM25:", [d[:20] for _,d in b])
    fused = rrf_fusion(v, b)
    print("RRF:", [d[:20] for d,_ in fused[:3]])
    candidates = [d for d,_ in fused[:5]]
    final = rerank(q, candidates)
    print("精排:", [d[:30] for _,d in final])
