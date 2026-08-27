# Systematization

This directory is the documentation entry point for the SoK systematization used by the artifact. The paper organizes LLM-generated phishing around generation controls, observed characteristics/user effects, and defenses. The benchmark data under `../data/benchmark/` uses the paper's current manipulation-stage assignments rather than the older lifecycle-style directory names that appeared in earlier development versions of the repository.

The empirical benchmark does not require a standalone dataset for every stage in the nine-stage taxonomy. Only stages/settings represented in the released benchmark are materialized under `../data/benchmark/`; S6 retains separate rewriting procedures and S8 retains controlled generator settings where relevant to the analysis.

Public study and dataset references should be interpreted together with the accompanying paper. This directory intentionally does not reconstruct missing study-coding records from inference; only source-backed systematization artifacts should be added here.
