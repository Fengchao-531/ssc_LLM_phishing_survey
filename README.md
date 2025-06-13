# ssc_LLM_phishing_survey
# Table of Contents
- [Table of Contents](#Table-of-Contents)
- [What's New](#whats-new)
- [Survey](#Survey)
- [LLM-Driven Phishing Textual Attacks](#llm-driven-phishing-textual-attacks)
  - [Corpus-adapted Generation](#Corpus-adapted-Generation)
  - [Prompt-based Generation](#Prompt-based-Generation)
  - [Knowledge-enhanced Generation](#Knowledge-enhanced-Generation)
- [Attack Vectors Distinctions](#Attack-Vectors-Distinctions)
  - [Attacks Attributes](#Attacks-Attributes)
  - [Textual Attributes](#Textual-Attributes)
  - [User Traits](#User-Traits)
-  [Anti-Phishing Countermeasures](#Anti-Phishing-Countermeasures)
   - [LLM-supported Detection](#LLM-supported-Detection)
   - [Deep Learning Models](#Deep-Learning-Models)
   - [Machine Learning Models](#Machine-Learning-Models)
   - [User-Centred Models](#User-Centred-Models)
   - [Traditional Anti-Phishing Countermeasures (Part)](#Traditional-Anti-Phishing-Countermeasures-(Part))
- [Reference](*Reference)
- [Statement](#Statement)

# What's New
# Survey
# LLM-Driven Phishing Textual Attacks
![image](https://github.com/user-attachments/assets/01a7c2c6-c7c4-49a5-a385-b81a8e4abd7a)
## Corpus-adapted Generation
| Year | Paper             | Threats Model        | Exploitations                          | Dataset   | Prompt  | Model   | Output            | Resources |
|------|-------------------|----------------------|----------------------------------------|-----|-----|-----|---------------------------------|----------|
| 2023 | Mehdi et al.      | GPT-2                | Perturbation Evasion                   |     |     | ✔️   | General Phishing Emails         |[Link]() |
| 2022 | Guo et al.        | BART, GPT-2          | Phishing Indicators                    | ✔️  |     | ✔️   | General Phishing Emails         |[Link]() |

## Prompt-based Generation
### Instruction Prompting
| Year | Paper             | Threats Model        | Exploitations                          | Dataset   | Prompt  | Model   | Output            | Resources |
|------|-------------------|----------------------|----------------------------------------|-----|-----|-----|---------------------------------|----------|
| 2022 | Karanjai et al.   | GPT-2, OPT etc.      | Phishing Indicators                    |     | ✔️   | ✔️   | General Phishing Emails         |[Link]() |
| 2023 | Patel et al.      | ChatGPT              | Chatbot Conversations                  |     | ✔️  |     | General Phishing Emails         |[Link]() |
| 2023 | Singh et al.      | Claude, BART, etc.   | Persuasion Principles                  |     | ✔️  |     | General Phishing Emails         |[Link]() |
| 2023 | Emanuela et al.   | ChatGPT              | Persuasion Principles                  |     | ✔️  |     | General/Spear Phishing Emails   |[Link]() |
| 2023 | Fredrik et al.    | GPT-4                | Phishing Indicators                    |     | ✔️   |     | Spear Phishing Emails           |[Link]() |
| 2023 | Utaliyeva et al.  | ChatGPT              | Prompt Engineering / Paraphrasing      | ✔️  | ✔️   |     | General Phishing Emails         |[Link]() |
| 2024 | Wani              | ChatGPT              | Phishing Indicators / Prompt Eng.      |     | ✔️   |     | Phishing Reviews                |[Link]() |
| 2024 | Chen et al.       | Llama 3.1            | Persuasion Principles, Subject Diversity | ✔️ | ✔️ |    | Multi-Subject Phishing Emails   |[Link]() |
| 2024 | Greco et al.      | WormGPT              | Prompt Engineering                     |     | ✔️   |     | General Phishing Emails         |[Link]() |
| 2024 | Francia et al.    | GPT-4                | Phishing Indicators                    |     | ✔️   |     | Spear Phishing SMS              |[Link]() |
| 2024 | Ekekihl           | ChatGPT              | Phishing Indicators                    |     | ✔️   |     | General Phishing Emails         |[Link]() |
| 2024 | Eze               | DeepAI               | Phishing Indicators / Prompt Eng.      |     | ✔️   |     | General Phishing Emails         |[Link]() |
| 2024 | Bethany et al.    | GPT-4                | Persuasion Principles                  | ✔️  | ✔️   |     | Spear Phishing Emails           |[Link]() |
| 2024 | Mahendru et al.   | GPT-4                | Persuasion Principles                  |     | ✔️   |     | General Phishing SMS/Emails     |[Link]() |

### Template-based Prompting
| Year | Paper             | Threats Model        | Exploitations                          | Dataset   | Prompt  | Model   | Output            | Resources |
|------|-------------------|----------------------|----------------------------------------|-----|-----|-----|---------------------------------|----------|
| 2023 | Langford et al.   | ChatGPT              | Persuasion Principles                  |     | ✔️   |     | Spear Phishing Emails           |[Link]() |
| 2024 | Panda et al.      | GPT-4                | Prompt Leakage                         | ✔️  |      | ✔️  | Spear Phishing Bio              |[Link]() |
| 2024 | Gryka et al.      | GPT-3.5-turbo        | Phishing Indicators                    |     | ✔️   |     | General Phishing Emails         |[Link]() |
### Conditional Prompting
| Year | Paper             | Threats Model        | Exploitations                          | Dataset   | Prompt  | Model   | Output            | Resources |
|------|-------------------|----------------------|----------------------------------------|-----|-----|-----|---------------------------------|----------|
| 2024 | Kang et al.       | GPT2-XL, ChatGPT     | Prompt Leakage                         |     | ✔️   |     | Spear Phishing Emails           |[Link]() |
### Auto-prompting
| Year | Paper             | Threats Model        | Exploitations                          | Dataset   | Prompt  | Model   | Output            | Resources |
|------|-------------------|----------------------|----------------------------------------|-----|-----|-----|---------------------------------|----------|
| 2024 | Roy et al.        | GPT-4, Claude, etc.  | Prompt Mining                          | ✔️   | ✔️  |     | Malicious Prompt Dataset        |[Link]() |


## Knowledge-enhanced Generation
| Year | Paper             | Threats Model        | Exploitations                          | Dataset   | Prompt  | Model   | Output            | Resources |
|------|-------------------|----------------------|----------------------------------------|-----|-----|-----|---------------------------------|----------|
| 2021 | Khan et al.      | GPT-2                | Phishing Indicators                     | ✔️  |     |     | General Phishing Emails         |[Link]() |
| 2023 | Hazell           | GPT-3.5/4            | Persuasion Principles, Phishing Indicators |   | ✔️ |     | Spear Phishing Emails           |[Link]() |
| 2024 | Carelli et al.    | GPT-3.5, WormGPT     | Persuasion Principles                  |     | ✔️   |     | Multi-Subject Phishing Emails   |[Link]() |
| 2024 | Fairbanks et al. | GPT-3.5              | Paraphrasing Attack                     |     |     | ✔️  | General Phishing Emails         |[Link]() |
| 2024 | Guo et al.       | TFM-based LLMs       | Phishing Indicators                     |     | ✔️  | ✔️  | Multi-Lingual Phishing Emails   |[Link]() |
| 2024 | Fredrik et al.   | GPT-4, Mistral, etc. | Prompt Leakage                          | ✔️   |     |     | Spear Phishing Emails           |[Link]() |
| 2025 | Perik et al.     | GPT-4                | Prompt Engineering                      |       | ✔️ |     | Collusion Scam                  |[Link]() |


# Attack Vectors Distinctions
![image](https://github.com/user-attachments/assets/c43198a8-fe2c-4a77-981a-df455a922b71)

## Attacks Attributes
| Year | Paper | Lexical | Semantic | Tactics | Intention | Model | Product | Distinguish Texts | Resources |
|------|-------|---------|----------|---------|-----------|-------|---------|-------------------|-----------|

## Textual Attributes
| Year | Paper | Lexical | Semantic | Tactics | Intention | Model | Product | Distinguish Texts | Resources |
|------|-------|---------|----------|---------|-----------|-------|---------|-------------------|-----------|

## User Traits
| Year | Paper | Lexical | Semantic | Tactics | Intention | Model | Product | Distinguish Texts | Resources |
|------|-------|---------|----------|---------|-----------|-------|---------|-------------------|-----------|

# Anti-Phishing Countermeasures
![image](https://github.com/user-attachments/assets/ea07c796-951b-4ca0-b157-1f2da1886c14)
## LLM-supported Detection
### Feature-based Detection
| Year | Paper | Lexical | Semantic | Tactics | Intention | Model | Product | Distinguish Texts | Resources |
|------|-------|---------|----------|---------|-----------|-------|---------|-------------------|-----------|
| 2024 | Quinn et al. | ✔️ | ✔️ | ✔️ | ✔️ | Gemini | Protocol-based Analyzer | LLM-generated Phishing Emails |  [Link]()|
| 2024 | Mahendru et al. | ✔️ | ✔️ | ✔️ | ✔️ | Deberta V3 | Rule-based Analyzer | LLM-generated Phishing Emails | [Link]() |
| 2025 | Perik et al.     |✔️      |         |         |           | ChatGPT|         | Collusion Scam    |[Link]()  |
### Prompt-targeted Detection
| Year | Paper | Lexical | Semantic | Tactics | Intention | Model | Product | Distinguish Texts | Resources |
|------|-------|---------|----------|---------|-----------|-------|---------|-------------------|-----------|
| 2024 | Roy et al. |  |  |  | ✔️ | DistilBERT | Detector | LLM-generated Phishing Prompts | [Link]() |
### Malicious Intent Detection
| Year | Paper | Lexical | Semantic | Tactics | Intention | Model | Product | Distinguish Texts | Resources |
|------|-------|---------|----------|---------|-----------|-------|---------|-------------------|-----------|
| 2023 | Heiding et al. |  |  |  | ✔️ | Bard, PaLM, etc. | Detector | LLM-generated Phishing Emails | [Link]() |
| 2024 | Heiding et al. |  |  |  | ✔️ | Claude, Mistral, etc. | Agent-Detector | LLM-generated Spear Phishing Emails | [Link]() |
## Deep Learning Models
| Year | Paper | Lexical | Semantic | Tactics | Intention | Model | Product | Distinguish Texts | Resources |
|------|-------|---------|----------|---------|-----------|-------|---------|-------------------|-----------|
| 2024 | Elongha et al. | ✔️ | ✔️ |  |  | CNN | Ensemble-Detector | LLM-generated Phishing Emails |[Link]()  |
| 2024 | Wani et al. | ✔️ | ✔️ |  |  | N.A. | Classifier | LLM-generated + General Phishing Reviews | [Link]() |

## Machine Learning Models
| Year | Paper | Lexical | Semantic | Tactics | Intention | Model | Product | Distinguish Texts | Resources |
|------|-------|---------|----------|---------|-----------|-------|---------|-------------------|-----------|
| 2024 | Bethany et al. | ✔️ | ✔️ |  |  | T5-LLMEmail | Detector | LLM-generated Phishing Emails | [Link]() |
| 2024 | Greco et al. | ✔️ |  |  |  | LR | Classifier | LLM-generated + General Phishing Emails | [Link]() |
| 2024 | Gryka et al. | ✔️ |  |  |  | SVM, NB, etc. | Classifier | LLM-generated + General Phishing Emails | [Link]() |
| 2024 | Carelli | ✔️ |  |  |  | SVM, NB, etc. | Classifier | LLM-generated + General Phishing Emails |[Link]()  |
| 2024 | Eze et al. | ✔️ | ✔️ |  |  | UDAT, NB, etc. | Ensemble-Classifier | LLM-generated + General Phishing Emails |[Link]() |
| 2024 | Jovanovic et al. | ✔️ |  |  |  | Boosting, NB, etc. | Detector | LLM-generated Phishing Emails | [Link]() |
## User-Centred Models
| Year | Paper | Lexical | Semantic | Tactics | Intention | Model | Product | Distinguish Texts | Resources |
|------|-------|---------|----------|---------|-----------|-------|---------|-------------------|-----------|
| 2025 | Malloy et al. |  |  |  |  | IBL Models | User Behavior Predictor | LLM-generated Phishing Emails |[Link]()  |

## Traditional Anti-Phishing Countermeasures (Part)
| Year | Paper | Lexical | Semantic | Tactics | Intention | Model | Product | Distinguish Texts | Resources |
|------|-------|---------|----------|---------|-----------|-------|---------|-------------------|-----------|
| 2022 | Misra et al. | ✔️ | ✔️ |   |   | GPT-2 | Classifier | General Phishing/Benign Emails |  [Link]() |
| 2023 | Otieno et al. | ✔️ | ✔️ |   |   | BERT | Classifier | General Phishing/Benign Emails |  [Link]() |
| 2024 | Nguyen et al. | ✔️ | ✔️ | ✔️ |   | Flancon, Mistral, etc. | Warning Generator | General Phishing Emails |  [Link]() |
| 2024 | Uddin et al. | ✔️ | ✔️ |   |   | LIME + DistilBERT | X-Classifier | General Phishing/Benign Emails | [Link]()  |
| 2024 | Koide et al. | ✔️ | ✔️ | ✔️ | ✔️ | ChatGPT | SE-based Analyzer | Phishing Pretext Analyze | [Link]()  |
|  | | | | | | | | | |
| 2021 | Alhogail et al. | ✔️ | ✔️ |   |   | GCN | Classifier | General Phishing/Benign Emails | [Link]()  |
| 2021 | Lee et al. | ✔️ | ✔️ |   |   | BERT | Classifier | General Phishing/Benign Emails | [Link]()  |
| 2022 | Qachfar et al. | ✔️ | ✔️ |   |   | BERT | Classifier | Synthetic + General Phishing/Benign Emails |  [Link]() |
| 2023 | Doshi et al. | ✔️ | ✔️ |   |   | ANN, CNN, RNN | Classifier | General Phishing/Benign Emails | [Link]()  |
| 2024 | Zhou et al. | ✔️ | ✔️ |   |   | C-Net | Classifier | Multilingual Synthetic & General Spam Content |  [Link]() |
| 2024 | Smadi et al. | ✔️ |   |   |   | DENNuRL | Classifier | Zero-day Phishing Emails | [Link]()  |
| 2024 | Magdy et al. | ✔️ |   |   |   | ANN | Classifier | General Phishing/Benign Emails | [Link]()  |
|  | | | | | | | | | |
| 2017 | Holgado et al. | / | / | / | / | HMM | Multi-HMM Predictor | N.A. | [Link]()  |
| 2018 | Gutierrez et al. | ✔️ | ✔️ |   |   | SAFE-PC | Classifier | General Phishing/Benign Emails | [Link]()  |
| 2024 | Ramanathan et al. | ✔️ |   |   |   | CRF-Adaboost | Classifier | General Phishing/Benign Emails | [Link]()  |
| 2021 | Valecha et al. |   |   | ✔️ |   | SVM, RF, etc. | Classifier | General Phishing/Benign Emails |  [Link]() |
|  | | | | | | | | | |
| 2018 | Hu et al. | / | / | / | / | Email Providers | Evaluation Report | General Phishing/Benign Emails |  [Link]() |
| 2022 | Galdi et al. | ✔️ |   |   |   | N.A. | Agent-Analyzer | General Phishing/Benign Emails |  [Link]() |
| 2024 | Wang et al. | / | / | / | / | N.A. | VeriSMS System | Healthcare Phishing SMS Messages |  [Link]() |


# Reference
# Statement
