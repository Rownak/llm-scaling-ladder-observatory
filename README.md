# llm-scaling-ladder-observatory
Offline-first LLM evaluation system studying perplexity, scaling behavior, prompt sensitivity, and cross-framework parity across the Pythia scaling ladder.

# Description
Scaling Ladder Observatory evaluates a ladder of Pythia checkpoints (70M–1B, across training steps) on perplexity and downstream benchmarks, then asks how well perplexity predicts capability as models scale. It implements sliding-window perplexity with bits-per-byte from scratch, verifies its own benchmark implementations against lm-evaluation-harness with per-example discrepancy analysis, and measures how much prompt formatting alone moves reported accuracy — with every run reproducible from stored metadata and every metric backed by a hand-computed test. Built as a self-contained, offline-first system with SQLite storage and a single CLI (ladderctl).
