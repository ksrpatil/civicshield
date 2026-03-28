# CivicShield

**A Cross-Domain Defense-in-Depth Framework for Securing Government-Facing AI Chatbots Against Multi-Turn Adversarial Attacks**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

CivicShield is a seven-layer defense-in-depth framework that secures government AI chatbots against jailbreak attacks, prompt injection, and multi-turn social engineering. It adapts proven security techniques from network security, software analysis, and formal verification to the AI chatbot domain.

**Key Results:**
- 72.9% TPR on real benchmarks (HarmBench, JailbreakBench, XSTest)
- 2.9% effective false positive rate
- 100% multi-turn attack detection
- 1,436 scenarios across 13 attack categories
- NIST SP 800-53 compliance mapping across 14 control families

## Architecture

| Layer | Defense | NIST Controls |
|-------|---------|---------------|
| L1 | Input validation + adversarial embedding detection | SI-10 |
| L2 | Semantic similarity scoring | SI-3 |
| L3 | Multi-turn taint tracking | AU-10 |
| L4 | Risk-based response classification | SI-4 |
| L5 | Graduated response (flag/block/escalate) | IR-4 |
| L6 | Multi-model consensus verification | SA-11 |
| L7 | Human escalation | CP-2 |

## Quick Start

```bash
pip install -r requirements.txt
python civicshield_simulation_v4.py
```

## Files

| File | Description |
|------|-------------|
| `civicshield_simulation.py` | Main simulation (definitive version) |
| `jbb_failure_analysis.py` | JailbreakBench failure mode analysis |
| `llm_in_loop_test.py` | LLM-in-the-loop evaluation |

## Requirements

- Python 3.12+
- sentence-transformers
- numpy, scipy, torch
- No GPU required (CPU only)

## Citation

```bibtex
@article{patil2026civicshield,
  title={CivicShield: A Cross-Domain Defense-in-Depth Framework for Securing Government-Facing AI Chatbots Against Multi-Turn Adversarial Attacks},
  author={Patil, KrishnaSaiReddy},
  year={2026}
}
```

## Related Work

This is Paper 1 in a trilogy on government AI security:
1. **CivicShield** — Conversational AI chatbot defense (this repo)
2. [RAGShield](https://github.com/ksrpatil/ragshield) — Knowledge base / RAG pipeline defense
3. [SentinelAgent](https://github.com/ksrpatil/sentinelagent) — Autonomous AI agent delegation security

## License

MIT License. See [LICENSE](LICENSE).
