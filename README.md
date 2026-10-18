# Trajectory-Level Safety Monitoring for LLM Agents
APS360 project proposal — Joeun Yook

A lightweight monitor that classifies LLM agent trajectories
(prompt → tool calls → actions → final answer) as safe or unsafe.

## Plan
- **Data:** AgentDoG (Liu et al., 2026), trajectory-level safe/unsafe labels.
  Split 80/10/10 by trajectory, then chunk to ≤512 tokens.
- **Model:** DistilBERT (66M) → pooling over chunks → linear head, BCE loss.
- **Baseline:** TF-IDF + logistic regression.
- **Metrics:** precision, recall, F1, PR-AUC, inference latency.

## Status
Proposal stage. Code to follow.

## Setup
```
pip install -r requirements.txt
```
