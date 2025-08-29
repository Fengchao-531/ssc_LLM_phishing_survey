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

| Year | Author            | Generation   | Characteristics  | Defense   | Resources |
|------|-------------------|--------------|-----------------|------------|-----------|
|  2017     |  Dou et al.              |   |   |✔️  |  [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8036198)   |
|  2024      |  Mathew  et al.  | ✔️          |                 | ✔️         |   [Link](https://d197for5662m48.cloudfront.net/documents/publicationstatus/228635/preprint_pdf/095e1e86e56492fce3118e1eba27f7d4.pdf) |
|  2024      | Schmitt \& Flechais |   |  ✔️ |  |  [Link](https://link.springer.com/content/pdf/10.1007/s10462-024-10973-2.pdf)         |
|  2023      |   Zhuo et al              |   |   | ✔️ |  [Link](https://dl.acm.org/doi/pdf/10.1145/3575797)          |
|  2024      |   Kumarage et al  |   |   | ✔️| [Link](https://arxiv.org/pdf/2403.01152)          |
|  2025      |   Wu et al.   |   |   |✔️  |  [Link](https://direct.mit.edu/coli/article/51/1/275/127462)          |
|2025| Chen et al.     |✔️ |✔️|✔️| -- |

# LLM-Driven Phishing Textual Attacks
[![Preview](docs/preview.png)](docs/paper.pdf)
## Corpus-adapted Generation
| Year | Author             | Threats Model        | Exploitations                          | Dataset   | Prompt  | Model   | Output            | Resources |
|------|-------------------|----------------------|----------------------------------------|-----|-----|-----|---------------------------------|----------|
| 2023 | Mehdi et al.      | GPT-2                | Perturbation Evasion                   |     |     | ✔️   | General Phishing Emails         |[Link](https://dl.acm.org/doi/abs/10.1145/3579987.3586567) |
| 2022 | Guo et al.        | BART, GPT-2          | Phishing Indicators                    | ✔️  |     | ✔️   | General Phishing Emails         |[Link](https://link.springer.com/chapter/10.1007/978-3-031-20029-8_26) |

## Prompt-based Generation
### Instruction Prompting
| Year | Author             | Threats Model        | Exploitations                          | Dataset   | Prompt  | Model   | Output            | Resources |
|------|-------------------|----------------------|----------------------------------------|-----|-----|-----|---------------------------------|----------|
| 2022 | Karanjai et al.   | GPT-2, OPT etc.      | Phishing Indicators                    |     | ✔️   | ✔️   | General Phishing Emails       |[Link](https://arxiv.org/abs/2301.00665) |
| 2023 | Patel et al.      | ChatGPT              | Chatbot Conversations                  |     | ✔️  |     | General Phishing Emails         |[Link](https://www.inuit.se/hubfs/WithSecure/files/WithSecure-Creatively-malicious-prompt-engineering.pdf) |
| 2023 | Singh et al.      | Claude, BART, etc.   | Persuasion Principles                  |     | ✔️  |     | General Phishing Emails         |[Link](https://ieeexplore.ieee.org/abstract/document/10386814/) |
| 2023 | Emanuela et al.   | ChatGPT              | Persuasion Principles                  |     | ✔️  |     | General/Spear Phishing Emails   |[Link](https://ieeexplore.ieee.org/abstract/document/10607415/) |
| 2023 | Fredrik et al.    | GPT-4                | Phishing Indicators                    |     | ✔️   |     | Spear Phishing Emails           |[Link](https://arxiv.org/abs/2308.12287) |
| 2023 | Utaliyeva et al.  | ChatGPT              | Prompt Engineering / Paraphrasing      | ✔️  | ✔️   |     | General Phishing Emails         |[Link](https://ieeexplore.ieee.org/abstract/document/10466952/) |
| 2024 | Wani              | ChatGPT              | Phishing Indicators / Prompt Eng.      |     | ✔️   |     | Phishing Reviews                |[Link](https://search.ebscohost.com/login.aspx?direct=true&profile=ehost&scope=site&authtype=crawler&jrnl=2073431X&AN=180558443&h=RJRe0AMI1v6u7kTHecJDx4GnVBiNYseiozuh1x%2FziOCv0GzxZ34RmuOAfqDuVXqTrPwhuPs1%2BpohNPb1wvBsSA%3D%3D&crl=c) |
| 2024 | Chen et al.       | Llama 3.1            | Persuasion Principles, Subject Diversity | ✔️ | ✔️ |    | Multi-Subject Phishing Emails   |[Link](https://arxiv.org/abs/2411.11389) |
| 2024 | Greco et al.      | WormGPT              | Prompt Engineering                     |     | ✔️   |     | General Phishing Emails         |[Link](https://www.researchgate.net/profile/Andrea-Esposito-12/publication/379697853_David_versus_Goliath_Can_Machine_Learning_Detect_LLM-Generated_Text_A_Case_Study_in_the_Detection_of_Phishing_Emails/links/661f8f2843f8df018d14631b/David-versus-Goliath-Can-Machine-Learning-Detect-LLM-Generated-Text-A-Case-Study-in-the-Detection-of-Phishing-Emails.pdf) |
| 2024 | Francia et al.    | GPT-4                | Phishing Indicators                    |     | ✔️   |     | Spear Phishing SMS              |[Link](https://arxiv.org/abs/2406.13049) |
| 2024 | Ekekihl           | ChatGPT              | Phishing Indicators                    |     | ✔️   |     | General Phishing Emails         |[Link](https://www.diva-portal.org/smash/record.jsf?pid=diva2:1879930) |
| 2024 | Eze               | DeepAI               | Phishing Indicators / Prompt Eng.      |     | ✔️   |     | General Phishing Emails         |[Link](https://www.mdpi.com/2079-9292/13/10/1839) |
| 2024 | Bethany et al.    | GPT-4                | Persuasion Principles                  | ✔️  | ✔️   |     | Spear Phishing Emails           |[Link](https://arxiv.org/abs/2401.09727) |
| 2024 | Mahendru et al.   | GPT-4                | Persuasion Principles                  |     | ✔️   |     | General Phishing SMS/Emails     |[Link](https://ieeexplore.ieee.org/abstract/document/10692765/) |

### Template-based Prompting
| Year | Author             | Threats Model        | Exploitations                          | Dataset   | Prompt  | Model   | Output            | Resources |
|------|-------------------|----------------------|----------------------------------------|-----|-----|-----|---------------------------------|----------|
| 2023 | Langford et al.   | ChatGPT              | Persuasion Principles                  |     | ✔️   |     | Spear Phishing Emails           |[Link](https://link.springer.com/chapter/10.1007/978-3-031-47454-5_13) |
| 2024 | Panda et al.      | GPT-4                | Prompt Leakage                         | ✔️  |      | ✔️  | Spear Phishing Bio              |[Link](https://arxiv.org/abs/2403.00871) |
| 2024 | Gryka et al.      | GPT-3.5-turbo        | Phishing Indicators                    |     | ✔️   |     | General Phishing Emails         |[Link](https://dl.acm.org/doi/abs/10.1145/3664476.3670465) |
### Conditional Prompting
| Year | Author             | Threats Model        | Exploitations                          | Dataset   | Prompt  | Model   | Output            | Resources |
|------|-------------------|----------------------|----------------------------------------|-----|-----|-----|---------------------------------|----------|
| 2024 | Kang et al.       | GPT2-XL, ChatGPT     | Prompt Leakage                         |     | ✔️   |     | Spear Phishing Emails           |[Link](https://ieeexplore.ieee.org/abstract/document/10579515/) |
### Auto-prompting
| Year | Author             | Threats Model        | Exploitations                          | Dataset   | Prompt  | Model   | Output            | Resources |
|------|-------------------|----------------------|----------------------------------------|-----|-----|-----|---------------------------------|----------|
| 2024 | Roy et al.        | GPT-4, Claude, etc.  | Prompt Mining                          | ✔️   | ✔️  |     | Malicious Prompt Dataset        |[Link](https://ieeexplore.ieee.org/abstract/document/10646856/) |


## Knowledge-enhanced Generation
| Year | Author             | Threats Model        | Exploitations                          | Dataset   | Prompt  | Model   | Output            | Resources |
|------|-------------------|----------------------|----------------------------------------|-----|-----|-----|---------------------------------|----------|
| 2021 | Khan et al.      | GPT-2                | Phishing Indicators                     | ✔️  |     |     | General Phishing Emails         |[Link](https://digital-library.theiet.org/doi/abs/10.1049/icp.2021.2422) |
| 2023 | Hazell           | GPT-3.5/4            | Persuasion Principles, Phishing Indicators |   | ✔️ |     | Spear Phishing Emails           |[Link](https://arxiv.org/abs/2305.06972) |
| 2024 | Carelli et al.    | GPT-3.5, WormGPT     | Persuasion Principles                  |     | ✔️   |     | Multi-Subject Phishing Emails   |[Link](https://www.researchgate.net/profile/Alessandro-Carelli-2/publication/382648027_Finding_Differences_Between_LLM-generated_And_Human-written_Text_A_Phishing_Emails_Case_Study/links/66a79ca14433ad480e845754/Finding-Differences-Between-LLM-generated-And-Human-written-Text-A-Phishing-Emails-Case-Study.pdf) |
| 2024 | Fairbanks et al. | GPT-3.5              | Paraphrasing Attack                     |     |     | ✔️  | General Phishing Emails         |[Link](https://ieeexplore.ieee.org/abstract/document/10825007/) |
| 2024 | Guo et al.       | TFM-based LLMs       | Phishing Indicators                     |     | ✔️  | ✔️  | Multi-Lingual Phishing Emails   |[Link](https://dl.acm.org/doi/abs/10.1145/3670402) |
| 2024 | Fredrik et al.   | GPT-4, Mistral, etc. | Prompt Leakage                          | ✔️   |     |     | Spear Phishing Emails           |[Link](https://arxiv.org/abs/2412.00586) |
| 2025 | Perik et al.     | GPT-4                | Prompt Engineering                      |       | ✔️ |     | Collusion Scam                  |[Link](http://essay.utwente.nl/104867/) |


# Attack Vectors Distinctions
![image](https://github.com/user-attachments/assets/c43198a8-fe2c-4a77-981a-df455a922b71)

## Attacks Attributes
| Year | Author | Lexical | Semantic | Tactics | Intention | Model | Product | Distinguish Texts | Resources |
|------|-------|---------|----------|---------|-----------|-------|---------|-------------------|-----------|

## Textual Attributes
| Year | Author | Lexical | Semantic | Tactics | Intention | Model | Product | Distinguish Texts | Resources |
|------|-------|---------|----------|---------|-----------|-------|---------|-------------------|-----------|

## User Traits
| Year | Author | Lexical | Semantic | Tactics | Intention | Model | Product | Distinguish Texts | Resources |
|------|-------|---------|----------|---------|-----------|-------|---------|-------------------|-----------|

# Anti-Phishing Countermeasures
![image](https://github.com/user-attachments/assets/ea07c796-951b-4ca0-b157-1f2da1886c14)
## LLM-supported Detection
### Feature-based Detection
| Year | Author | Lexical | Semantic | Tactics | Intention | Model | Product | Distinguish Texts | Resources |
|------|-------|---------|----------|---------|-----------|-------|---------|-------------------|-----------|
| 2024 | Quinn et al. | ✔️ | ✔️ | ✔️ | ✔️ | Gemini | Protocol-based Analyzer | LLM-generated Phishing Emails |  [Link](https://assets-eu.researchsquare.com/files/rs-4405206/v1_covered_c6529ffb-ac95-411b-ab4c-16538a029c64.pdf)|
| 2024 | Mahendru et al. | ✔️ | ✔️ | ✔️ | ✔️ | Deberta V3 | Rule-based Analyzer | LLM-generated Phishing Emails | [Link](https://ieeexplore.ieee.org/abstract/document/10692765/) |
| 2025 | Perik et al.     |✔️      |         |         |           | ChatGPT|         | Collusion Scam    |[Link](http://essay.utwente.nl/104867/)  |
### Prompt-targeted Detection
| Year | Author | Lexical | Semantic | Tactics | Intention | Model | Product | Distinguish Texts | Resources |
|------|-------|---------|----------|---------|-----------|-------|---------|-------------------|-----------|
| 2024 | Roy et al. |  |  |  | ✔️ | DistilBERT | Detector | LLM-generated Phishing Prompts | [Link](https://ieeexplore.ieee.org/abstract/document/10646856/) |
### Malicious Intent Detection
| Year | Author | Lexical | Semantic | Tactics | Intention | Model | Product | Distinguish Texts | Resources |
|------|-------|---------|----------|---------|-----------|-------|---------|-------------------|-----------|
| 2023 | Fredrik et al. |  |  |  | ✔️ | Bard, PaLM, etc. | Detector | LLM-generated Phishing Emails | [Link](https://arxiv.org/abs/2308.12287) |
| 2024 | Fredrik et al. |  |  |  | ✔️ | Claude, Mistral, etc. | Agent-Detector | LLM-generated Spear Phishing Emails | [Link](https://arxiv.org/abs/2412.00586) |
## Deep Learning Models
| Year | Author | Lexical | Semantic | Tactics | Intention | Model | Product | Distinguish Texts | Resources |
|------|-------|---------|----------|---------|-----------|-------|---------|-------------------|-----------|
| 2024 | Elongha et al. | ✔️ | ✔️ |  |  | CNN | Ensemble-Detector | LLM-generated Phishing Emails |[Link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4934129)  |
| 2024 | Wani et al. | ✔️ | ✔️ |  |  | N.A. | Classifier | LLM-generated + General Phishing Reviews | [Link](https://search.ebscohost.com/login.aspx?direct=true&profile=ehost&scope=site&authtype=crawler&jrnl=2073431X&AN=180558443&h=RJRe0AMI1v6u7kTHecJDx4GnVBiNYseiozuh1x%2FziOCv0GzxZ34RmuOAfqDuVXqTrPwhuPs1%2BpohNPb1wvBsSA%3D%3D&crl=c) |

## Machine Learning Models
| Year | Author | Lexical | Semantic | Tactics | Intention | Model | Product | Distinguish Texts | Resources |
|------|-------|---------|----------|---------|-----------|-------|---------|-------------------|-----------|
| 2024 | Bethany et al. | ✔️ | ✔️ |  |  | T5-LLMEmail | Detector | LLM-generated Phishing Emails | [Link](https://arxiv.org/abs/2401.09727) |
| 2024 | Greco et al. | ✔️ |  |  |  | LR | Classifier | LLM-generated + General Phishing Emails | [Link](https://www.researchgate.net/profile/Andrea-Esposito-12/publication/379697853_David_versus_Goliath_Can_Machine_Learning_Detect_LLM-Generated_Text_A_Case_Study_in_the_Detection_of_Phishing_Emails/links/661f8f2843f8df018d14631b/David-versus-Goliath-Can-Machine-Learning-Detect-LLM-Generated-Text-A-Case-Study-in-the-Detection-of-Phishing-Emails.pdf) |
| 2024 | Gryka et al. | ✔️ |  |  |  | SVM, NB, etc. | Classifier | LLM-generated + General Phishing Emails | [Link](https://dl.acm.org/doi/abs/10.1145/3664476.3670465) |
| 2024 | Carelli | ✔️ |  |  |  | SVM, NB, etc. | Classifier | LLM-generated + General Phishing Emails |[Link](https://www.researchgate.net/profile/Alessandro-Carelli-2/publication/382648027_Finding_Differences_Between_LLM-generated_And_Human-written_Text_A_Phishing_Emails_Case_Study/links/66a79ca14433ad480e845754/Finding-Differences-Between-LLM-generated-And-Human-written-Text-A-Phishing-Emails-Case-Study.pdf)  |
| 2024 | Eze et al. | ✔️ | ✔️ |  |  | UDAT, NB, etc. | Ensemble-Classifier | LLM-generated + General Phishing Emails |[Link](https://www.mdpi.com/2079-9292/13/10/1839) |
| 2024 | Jovanovic et al. | ✔️ |  |  |  | Boosting, NB, etc. | Detector | LLM-generated Phishing Emails | [Link](https://link.springer.com/chapter/10.1007/978-981-97-3191-6_46) |
## User-Centred Models
| Year | Author | Lexical | Semantic | Tactics | Intention | Model | Product | Distinguish Texts | Resources |
|------|-------|---------|----------|---------|-----------|-------|---------|-------------------|-----------|
| 2025 | Malloy et al. |  |  |  |  | IBL Models | User Behavior Predictor | LLM-generated Phishing Emails |[Link](https://arxiv.org/abs/2502.01764)  |

## Traditional Anti-Phishing Countermeasures (Part)
| Year | Author | Lexical | Semantic | Tactics | Intention | Model | Product | Distinguish Texts | Resources |
|------|-------|---------|----------|---------|-----------|-------|---------|-------------------|-----------|
| 2022 | Misra et al. | ✔️ | ✔️ |   |   | GPT-2 | Classifier | General Phishing/Benign Emails |  [Link](https://ieeexplore.ieee.org/abstract/document/10101955/) |
| 2023 | Otieno et al. | ✔️ | ✔️ |   |   | BERT | Classifier | General Phishing/Benign Emails |  [Link](https://ieeexplore.ieee.org/abstract/document/10197078/) |
| 2024 | Nguyen et al. | ✔️ | ✔️ | ✔️ |   | Flancon, Mistral, etc. | Warning Generator | General Phishing Emails |  [Link](https://dl.acm.org/doi/abs/10.1145/3665451.3665531) |
| 2024 | Uddin et al. | ✔️ | ✔️ |   |   | LIME + DistilBERT | X-Classifier | General Phishing/Benign Emails | [Link](https://arxiv.org/abs/2402.13871)  |
| 2024 | Koide et al. | ✔️ | ✔️ | ✔️ | ✔️ | ChatGPT | SE-based Analyzer | Phishing Pretext Analyze | [Link](https://arxiv.org/abs/2402.18093)  |
|  | | | | | | | | | |
| 2021 | Alhogail et al. | ✔️ | ✔️ |   |   | GCN | Classifier | General Phishing/Benign Emails | [Link](https://www.sciencedirect.com/science/article/pii/S0167404821002388?casa_token=y3J_4WqI8TIAAAAA:NzMhlLChIpaolzUxW8GUglsR_EP2pud7wK7dZAWV_fQirBo80zzpGQWglgT20z4_DWOLTIL0sw)  |
| 2021 | Lee et al. | ✔️ | ✔️ |   |   | BERT | Classifier | General Phishing/Benign Emails | [Link](https://ieeexplore.ieee.org/abstract/document/9581199/)  |
| 2022 | Qachfar et al. | ✔️ | ✔️ |   |   | BERT | Classifier | Synthetic + General Phishing/Benign Emails |  [Link](https://dl.acm.org/doi/abs/10.1145/3508398.3511524) |
| 2023 | Doshi et al. | ✔️ | ✔️ |   |   | ANN, CNN, RNN | Classifier | General Phishing/Benign Emails | [Link](https://www.sciencedirect.com/science/article/pii/S0167404823002882?casa_token=zetMU6QBe_gAAAAA:tgSxordHe-3Qt51cVdCckHeRKUlxniBGN0H9EZ7wItpeJQbNcoL8i72CF9a7YopwmNEynaxGxw)  |
| 2024 | Zhou et al. | ✔️ | ✔️ |   |   | C-Net | Classifier | Multilingual Synthetic & General Spam Content |  [Link](https://ieeexplore.ieee.org/abstract/document/10508945/) |
| 2024 | Smadi et al. | ✔️ |   |   |   | DENNuRL | Classifier | Zero-day Phishing Emails | [Link](https://www.sciencedirect.com/science/article/pii/S0167923618300010?casa_token=-ahdbD36QDYAAAAA:qYiT4gq3tDCaQqJvhxzWr0eREZ68bB_e8nAmx15Nhe44u19iMiMjEsy8H6Y3ouU5K08ypU63ZA)  |
| 2024 | Magdy et al. | ✔️ |   |   |   | ANN | Classifier | General Phishing/Benign Emails | [Link](https://www.sciencedirect.com/science/article/pii/S1389128622000469?casa_token=WEDDS-PBAosAAAAA:kOFc62c8heZdo2emfELFC_pDPDKHkXRsyB95QNoQCvbY-GvqqNldCY11YYOIv3ySwhRyxOZWAg)  |
|  | | | | | | | | | |
| 2017 | Holgado et al. | / | / | / | / | HMM | Multi-HMM Predictor | N.A. | [Link](https://ieeexplore.ieee.org/abstract/document/8031986/)  |
| 2018 | Gutierrez et al. | ✔️ | ✔️ |   |   | SAFE-PC | Classifier | General Phishing/Benign Emails | [Link](https://ieeexplore.ieee.org/abstract/document/8440723/)  |
| 2024 | Ramanathan et al. | ✔️ |   |   |   | CRF-Adaboost | Classifier | General Phishing/Benign Emails | [Link](https://www.sciencedirect.com/science/article/pii/S0167404812001812?casa_token=gX3eecBG7v4AAAAA:A7ZHEruEhcxSyFJ4tKJPIeqsb__IJLPpxr03S8JOauQkUwi7VBpoMToo4MQcalx5B4ARKSHX7A)  |
| 2021 | Valecha et al. |   |   | ✔️ |   | SVM, RF, etc. | Classifier | General Phishing/Benign Emails |  [Link](https://ieeexplore.ieee.org/abstract/document/9565347/) |
|  | | | | | | | | | |
| 2018 | Hu et al. | / | / | / | / | Email Providers | Evaluation Report | General Phishing/Benign Emails |  [Link](https://www.usenix.org/conference/usenixsecurity18/presentation/hu) |
| 2022 | Galdi et al. | ✔️ |   |   |   | N.A. | Agent-Analyzer | General Phishing/Benign Emails |  [Link](https://ceur-ws.org/Vol-3260/paper6.pdf) |
| 2024 | Wang et al. | / | / | / | / | N.A. | VeriSMS System | Healthcare Phishing SMS Messages |  [Link](https://dl.acm.org/doi/abs/10.1145/3613904.3642027) |


# Reference
# Statement
