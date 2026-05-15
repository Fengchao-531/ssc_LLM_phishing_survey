# Datasets

This directory contains the dataset collection used for the SSC LLM phishing
survey. The datasets are organized to cover different stages of the
LLM-enabled phishing lifecycle, from basic phishing-content generation to
cross-channel expansion and model-driven automation.

The repository keeps the public or reproducible data sources under
`Generic-Data`, `HW-Phishing`, `LLM-Phishing`, and processed stage-specific
subsets under `sublist/`. The stage-specific subsets are the primary inputs
used by the benchmarking and detector-evaluation scripts.

## Label Convention

Each processed dataset is assigned one of the following labels when the
underlying source and preprocessing make the distinction clear:

| Label | Meaning |
|---|---|
| `HW-P` | Human-written phishing content |
| `HW-B` | Human-written benign content |
| `LLM-P` | LLM-generated phishing content |
| `LLM-B` | LLM-generated benign content |

For cross-channel datasets, especially vishing scripts, scam conversations,
malicious URLs, and QR-code data, the same human/LLM provenance is preserved
where possible, but not every file maps cleanly to a benign/phishing email
split.

## Lifecycle Stage Mapping

Datasets are mapped to stages according to their metadata, original dataset
descriptions, and the primary phishing capability captured by each dataset.

| Stage | Directory | Mapping rule |
|---|---|---|
| `S1` | `sublist/S1-Basic Instruction/` | Direct phishing or benign email generation from basic instructions. Datasets whose documentation indicates direct generation by malicious or phishing-oriented LLMs, including WormGPT-style generation, are assigned here. |
| `S2` | `sublist/S2-Role-Framed Prompting/` | Role-playing, jailbreak-style, or role-framed prompting datasets. Only phishing-related samples are retained after preprocessing because many jailbreak tasks are not phishing-specific. |
| `S3` | `sublist/S3-Multi-turn Task Decomposition/` | Multi-turn task decomposition where attackers refine prompts across turns, attempt to bypass safeguards, or produce intermediate artifacts such as phishing emails or sensitive-information requests. |
| `S4` | `sublist/S4-Scenarios-driven Adaptation/` | Scenario-driven phishing generation. Datasets are classified according to the provided scenario, communication context, or generated phishing situation. |
| `S5` | `sublist/S5-Personalization for Credibility/` | Business email compromise and targeted phishing data tailored to specific roles, organizations, individuals, or business contexts. |
| `S6` | `sublist/S6-Stealthy Rewriting/` | Rewriting, paraphrasing, or fluency-improvement datasets where existing phishing content is modified for contextual adaptation or detection evasion. |
| `S7` | `sublist/S7-Cross-channel Expansion/` | Cross-channel phishing data, including quishing, malicious URL to QR-code conversion, vishing scripts, and scam conversations. |
| `S8` | `sublist/S8-Model-driven Automation/` | Reconstructed model-driven automation datasets based on the procedures and examples reported in the literature. Reproduced outputs are not publicly released when release would create safety risk. |

## Public Dataset Sources

The collection uses public datasets to cover email, URL, QR-code, vishing, and
scam-conversation modalities.

### Email and Text Phishing Corpora

| Dataset | Source | Year | Main use |
|---|---|---:|---|
| Phishyai | https://github.com/GangGreenTemperTatum/phishyai/ | 2023 | Human-written and generated phishing/ham email material |
| E-PhishGen | https://github.com/pajola/e-phishGen | 2025 | LLM-generated phishing and benign email generation |
| Human-LLM generated phishing-legitimate emails | https://www.kaggle.com/datasets/francescogreco97/human-llm-generated-phishing-legitimate-emails | 2024 | Human vs. LLM phishing/legitimate email comparison |
| Scambaiting Dataset | https://github.com/scambaitermailbox/scambaiting_dataset | 2022 | Scam email and scam-conversation material |
| Multi Agent Scam Conversation | https://huggingface.co/datasets/BothBosu/multi-agent-scam-conversation | 2025 | Multi-turn scam conversation data |
| DataPhish Phish Fuzzer | https://github.com/DataPhish/PhishFuzzer/tree/main/DataSet_Creation | 2026 | Phishing-content fuzzing and rewriting |
| Customer care emails | https://www.kaggle.com/datasets/rtweera/customer-care-emails | 2025 | Human-written benign customer-service email baseline |
| Adversarial BEC Email Dataset | https://www.kaggle.com/datasets/yoadjei/adversarial-bec-email-dataset | 2025 | Business email compromise and targeted phishing |
| Nazario phishing corpus | https://monkey.org/~jose/phishing/ | 2026 | Human-written phishing benchmark corpus |
| Cornell University Phish Bowl | https://it.cornell.edu/phish-bowl | 2026 | Human-written institutional phishing examples |
| Millersmile archive | https://www.millersmiles.co.uk/archives.php | 2026 | Human-written phishing benchmark corpus |
| Phishbot corpus | https://github.com/rf-peixoto/phishing_pot/tree/main/email | 2026 | Human-written phishing email benchmark corpus |

Additional datasets referenced by the survey include PiMRef, Paladin, and Malla
Phishing. These are used where their released artifacts or described
preprocessing procedures match the corresponding lifecycle stage.

### URL, QR-Code, and Quishing Data

| Dataset | Source | Year | Main use |
|---|---|---:|---|
| MalURLBench | https://github.com/JiangYingEr/MalURLBench/tree/main/Examples | 2026 | Malicious URL examples used for URL and QR-code experiments |
| Phishing Site URLs | https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls | 2020 | Phishing URL baseline |

The survey also references QR-code phishing datasets such as fouadtrad QRcode
and QGuard where they support the quishing portion of S7.

### Vishing and Scam Transcript Data

| Dataset | Source | Year | Main use |
|---|---|---:|---|
| AI-automated-vishing/Viking | https://github.com/ai-automated-vishing/Viking | 2025 | Audio robocall and vishing-style data |
| Composite Scam Transcript Dataset | https://www.kaggle.com/datasets/ibrahimbagwan12/composite-scam-transcript-dataset | 2026 | Scam transcript benchmark data |
| Scambaiting Dataset | https://github.com/scambaitermailbox/scambaiting_dataset | 2022 | Multi-turn scam and scambaiting interactions |
| Multi Agent Scam Conversation | https://huggingface.co/datasets/BothBosu/multi-agent-scam-conversation | 2025 | LLM-assisted multi-agent scam conversations |

The survey also references AI-FraudCall-Detector for vishing, smishing, and
fraud-call examples.

## Preprocessing Notes

- Source datasets are normalized into stage-specific files under `sublist/`.
- Role-playing and jailbreak datasets are filtered to retain phishing-related
  samples only.
- Scenario-driven datasets are mapped to S4 when the source description
  emphasizes scenario, persona, or communication-context adaptation.
- Targeted phishing and BEC datasets are mapped to S5 when they encode specific
  roles, organizations, recipients, or business workflows.
- Rewriting and paraphrasing datasets are mapped to S6 when the main operation
  changes existing phishing content for fluency, adaptation, or evasion.
- Cross-channel datasets are mapped to S7 when the phishing behavior is carried
  through URLs, QR codes, calls, scripts, or multi-turn scam conversations.
- S8 data is reconstructed from published procedures and examples, but
  reproduced datasets are not publicly released when doing so would increase
  misuse risk.

## Safety and Release Policy

This directory is intended for controlled benchmarking and research
reproducibility. Some generated or reconstructed phishing outputs may be
withheld, redacted, or represented only through derived evaluation artifacts
when public release would create safety concerns. Users should follow the
license, terms of use, and citation requirements of each upstream dataset.
