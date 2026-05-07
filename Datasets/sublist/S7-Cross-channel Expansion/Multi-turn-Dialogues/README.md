# Multi-turn Dialogues

This folder groups the processed S7 multi-turn dialogue CSVs in one place.

- `HW-Vishing-Multi.csv`: existing HW multi-turn set from the AIFraud pipeline
- `HW-Vishing-Multi-ScamBaiter.csv`: hand-written scam-baiting multi-turn threads from `scambaitermailbox/scambaiting_dataset`
- `LLM-Vishing-Multi.csv`: original AIFraud LLM multi-turn scam set
- `LLM-Vishing-Multi-BothBosu.csv`: scam-only multi-turn set from `BothBosu/multi-agent-scam-conversation`
- `LLM-Vishing-Multi-Augmented.csv`: `LLM-Vishing-Multi.csv` plus `BothBosu` scam rows
