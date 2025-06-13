# ssc_LLM_phishing_survey
# Table of Contents
- Table of Contents
- What's New
- Survey
- LLM-Driven Phishing Textual Attacks
  - Corpus-adapted Generation
  - Prompt-based Generation
  - Knowledge-enhanced Generation
- Attack Vectors Distinctions
  - Attacks Attributes
  - Textual Attributes
  - User Traits
-  Anti-Phishing Countermeasures
  - LLM-supported Detection
  - Deep Learning Models
  - Machine Learning Models
  - User-Centred Models
- Reference
- Statement

# What's New
# Survey
# LLM-Driven Phishing Textual Attacks
![image](https://github.com/user-attachments/assets/01a7c2c6-c7c4-49a5-a385-b81a8e4abd7a)

| Year | Paper             | Threats Model        | Exploitations                          | Dataset   | Prompt  | Model   | Output            | Resources |
|------|-------------------|----------------------|----------------------------------------|-----|-----|-----|---------------------------------|----------|
| 2023 | Mehdi et al.      | GPT-2                | Perturbation Evasion                   |     |     | ✓   | General Phishing Emails         |[Link]() |
| 2022 | Guo et al.        | BART, GPT-2          | Phishing Indicators                    | ✓   |     | ✓   | General Phishing Emails         |[Link]() |
| 2024 | Panda et al.      | GPT-4                | Prompt Leakage                         | ✓   |     | ✓   | Spear Phishing Bio              |[Link]() |
| 2022 | Karanjai et al.   | GPT-2, OPT etc.      | Phishing Indicators                    |     | ✓   | ✓   | General Phishing Emails         |[Link]() |
| 2023 | Emanuela et al.   | ChatGPT              | Persuasion Principles                  |     | ✓   |     | General/Spear Phishing Emails   |[Link]() |
| 2023 | Fredrik et al.    | GPT-4                | Phishing Indicators                    |     | ✓   |     | Spear Phishing Emails           |[Link]() |
| 2023 | Langford et al.   | ChatGPT              | Persuasion Principles                  |     | ✓   |     | Spear Phishing Emails           |[Link]() |
| 2023 | Utaliyeva et al.  | ChatGPT              | Prompt Engineering / Paraphrasing      | ✓   | ✓   |     | General Phishing Emails         |[Link]() |
| 2023 | Singh et al.      | Claude, BART, etc.   | Persuasion Principles                  |     | ✓   |     | General Phishing Emails         |[Link]() |
| 2024 | Greco et al.      | WormGPT              | Prompt Engineering                     |     | ✓   |     | General Phishing Emails         |[Link]() |
| 2024 | Wani              | ChatGPT              | Phishing Indicators / Prompt Eng.      |     | ✓   |     | Phishing Reviews                |[Link]() |
| 2024 | Ekekihl           | ChatGPT              | Phishing Indicators                    |     | ✓   |     | General Phishing Emails         |[Link]() |
| 2024 | Eze               | DeepAI               | Phishing Indicators / Prompt Eng.      |     | ✓   |     | General Phishing Emails         |[Link]() |
| 2024 | Francia et al.    | GPT-4                | Phishing Indicators                    |     | ✓   |     | Spear Phishing SMS              |[Link]() |
| 2024 | Bethany et al.    | GPT-4                | Persuasion Principles                  | ✓   | ✓   |     | Spear Phishing Emails           |[Link]() |
| 2024 | Carelli et al.    | GPT-3.5, WormGPT     | Persuasion Principles                  |     | ✓   |     | Multi-Subject Phishing Emails   |[Link]() |
| 2024 | Mahendru et al.   | GPT-4                | Persuasion Principles                  |     | ✓   |     | General Phishing SMS/Emails     |[Link]() |
| 2024 | Gryka et al.      | GPT-3.5-turbo        | Phishing Indicators                    |     | ✓   |     | General Phishing Emails         |[Link]() |
| 2024 | Roy et al.        | GPT-4, Claude, etc.  | Prompt Mining                          | ✓   | ✓   |     | Malicious Prompt Dataset        |[Link]() |
| 2024 | Kang et al.       | GPT2-XL, ChatGPT     | Prompt Leakage                         |     | ✓   |     | Spear Phishing Emails           |[Link]() |

## Corpus-adapted Generation
## Prompt-based Generation
## Knowledge-enhanced Generation

# Attack Vectors Distinctions
![image](https://github.com/user-attachments/assets/c43198a8-fe2c-4a77-981a-df455a922b71)

## Attacks Attributes
## Textual Attributes
## User Traits

# Anti-Phishing Countermeasures
![image](https://github.com/user-attachments/assets/ea07c796-951b-4ca0-b157-1f2da1886c14)
| Year | Paper | L | S | T | I | Model | Product | Distinguish Texts | Resources |
|------|-------|---|---|---|---|-------|---------|-------------------|-----------|
| 2023 | Heiding et al. |  |  |  | ✔️ | Bard, PaLM, etc. | Detector | LLM-generated Phishing Emails | [Link]() |
| 2024 | Bethany et al. | ✔️ | ✔️ |  |  | T5-LLMEmail | Detector | LLM-generated Phishing Emails | [Link]() |
| 2024 | Perik et al. |  |  |  |  |  |  |  | [Link]() |
| 2024 | Quinn et al. | ✔️ | ✔️ | ✔️ | ✔️ | Gemini | Protocol-based Analyzer | LLM-generated Phishing Emails |  [Link]()|
| 2024 | Mahendru et al. | ✔️ | ✔️ | ✔️ | ✔️ | Deberta V3 | Rule-based Analyzer | LLM-generated Phishing Emails | [Link]() |
| 2024 | Roy et al. |  |  |  | ✔️ | DistilBERT | Detector | LLM-generated Phishing Prompts | [Link]() |
| 2024 | Heiding et al. |  |  |  | ✔️ | Claude, Mistral, etc. | Agent-Detector | LLM-generated Spear Phishing Emails | [Link]() |
| 2025 | Malloy et al. |  |  |  |  | IBL Models | User Behavior Predictor | LLM-generated Phishing Emails |[Link]()  |
| 2024 | Elongha et al. | ✔️ | ✔️ |  |  | CNN | Ensemble-Detector | LLM-generated Phishing Emails |[Link]()  |
| 2024 | Greco et al. | ✔️ |  |  |  | LR | Classifier | LLM-generated + General Phishing Emails | [Link]() |
| 2024 | Gryka et al. | ✔️ |  |  |  | SVM, NB, etc. | Classifier | LLM-generated + General Phishing Emails | [Link]() |
| 2024 | Carelli | ✔️ |  |  |  | SVM, NB, etc. | Classifier | LLM-generated + General Phishing Emails |[Link]()  |
| 2024 | Jovanovic et al. | ✔️ |  |  |  | Boosting, NB, etc. | Detector | LLM-generated Phishing Emails | [Link]() |
| 2024 | Wani et al. | ✔️ | ✔️ |  |  | N.A. | Classifier | LLM-generated + General Phishing Reviews | [Link]() |
| 2024 | Eze et al. | ✔️ | ✔️ |  |  | UDAT, NB, etc. | Ensemble-Classifier | LLM-generated + General Phishing Emails |[Link]() |

## LLM-supported Detection
## Deep Learning Models
## Machine Learning Models
## User-Centred Models

# Reference
# Statement
