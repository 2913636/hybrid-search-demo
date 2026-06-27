# 混合检索 Demo

> 向量检索 + BM25 -> RRF 融合 -> Cross-Encoder 重排序 全流程演示。

## 运行

```bash
py hybrid.py                    # 默认 3 个查询
py hybrid.py "特斯拉 售价"       # 自定义查询
```

## 四阶段

| 阶段 | 方法 | 说明 |
|------|------|------|
| 1 | Vector Retrieval | Jaccard 相似度（真实场景用 embedding） |
| 2 | BM25 | 关键词精确匹配（IDF + TF） |
| 3 | RRF Fusion | 倒数排名融合，无需调参 |
| 4 | Cross-Encoder | 精细打分，取 Top 3 |

