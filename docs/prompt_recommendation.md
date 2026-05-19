# Prompt Recommendation Design

This project produces the precomputed graph features needed for an AI drawing
prompt recommendation system.

## Strategy 1: Confidence Autocomplete

Use directional confidence as conditional probability:

```text
confidence(X -> Y) = co_count(X, Y) / count(X)
```

When a user enters `karin_(blue_archive)`, the recommender can rank outgoing
neighbors by `confidence` to suggest canonical traits and companion concepts.
This is useful for preventing character out-of-character failures.

## Strategy 2: Discounted PPMI Exploration

Raw co-count over-recommends generic tags. Discounted PPMI rewards informative
associations while shrinking rare coincidences:

```text
discounted_ppmi = max(PMI, 0) * co_count / (co_count + k)
```

For a prompt containing `sword`, this favors tags such as scene, pose, weapon,
and action descriptors instead of globally frequent stop-words.

## Strategy 3: Community Motif Injection

When multiple input tags belong to the same detected community, use that
community's general-tag summary as a style pack.

Example:

```text
input tags: asuna_(school_uniform)_(blue_archive), neru_(blue_archive)
dominant community: C&C-like strict cluster
motif suggestions:
  cleaning_&_clearing_(blue_archive)
  sukajan
  aqua_leotard
  sig_mpx
```

This turns graph topology into context-aware prompt expansion.

## Online API Sketch

```python
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()


class PromptRequest(BaseModel):
    current_tags: list[str]
    target_category: str = "general"
    top_k: int = 10


@app.post("/api/v1/recommend")
async def recommend_tags(request: PromptRequest):
    scores: dict[str, float] = {}
    for tag in request.current_tags:
        for neighbor, score in fetch_neighbors(tag, request.target_category):
            if neighbor not in request.current_tags:
                scores[neighbor] = scores.get(neighbor, 0.0) + score

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {"recommended_tags": ranked[: request.top_k]}
```

## Storage Sketch

Redis sorted sets are enough for a fast first version:

```text
ZREVRANGE tag:karin_(blue_archive):related:general 0 20 WITHSCORES
ZREVRANGE tag:sword:related:general 0 20 WITHSCORES
ZREVRANGE community:234:general 0 20 WITHSCORES
```

## Representation Learning Extension

The sparse graph cannot solve zero co-occurrence by itself. A next-stage system
can run weighted random walks over the graph and train Node2Vec / Skip-gram
embeddings. FAISS or another vector index can then serve dense nearest-neighbor
queries, allowing recommendations even when two tags never directly co-occurred.
