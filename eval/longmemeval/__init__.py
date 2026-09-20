"""LongMemEval-S head-to-head between Memonative and mem0.

Two metrics:
  - recall@5  — does the top-5 retrieved memory set include any memory
                tagged with one of the question's needle session ids?
  - answer accuracy — does an LLM answering with the retrieved memories
                      match the gold answer (judged by gpt-4o-mini)?

Out-of-the-box defaults on both systems. No per-system tuning. The point
is a fair read on retrieval quality, not a leaderboard.
"""
