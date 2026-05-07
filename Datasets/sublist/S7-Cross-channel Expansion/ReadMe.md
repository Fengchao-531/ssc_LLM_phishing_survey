## S7 Cross-channel Expansion

This stage contains processed datasets for quishing and vishing, including both
single-turn and multi-turn variants.

### Canonical multi-turn CSVs

Use these files as the source of truth:

- `HW-Vishing-Multi.csv`
- `HW-Vishing-Multi-ScamBaiter.csv`
- `LLM-Script-Multi.csv`
- `LLM-Script-Multi-BothBosu.csv`
- `LLM-Script-Multi-Augmented.csv`
- `LLM-Vishing-Multi.csv`
- `LLM-Vishing-Multi-BothBosu.csv`
- `LLM-Vishing-Multi-Augmented.csv`

### Why `Multi-turn-Dialogues/` is small

The `Multi-turn-Dialogues/` folder is now a lightweight index with a README that
points back to the canonical CSVs above. The previous duplicate copies were
removed to keep the repository smaller and avoid storing the same processed data
twice.
