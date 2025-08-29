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

[![Preview](<img width="1730" height="438" alt="image" src="https://github.com/user-attachments/assets/6091c539-1751-4d59-a312-311fde62ac41"/>)](diagrams/GED.pdf)
# RQ1 Exploitation Techniques for Phishing Text Generation across Different LLMs
<p align="center">
  <img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/61a83904-853b-42cf-a5c4-c148c4d4d90a" />
</p>


## Data-Guided
### Task Adaptation
| Year | Venue              | Paper                           | Threats Model          | Exploitations                                     | Data | Prompt | Model | Methods                 | 
|------|--------------------|---------------------------------|------------------------|---------------------------------------------------|---|---|---|-------------------------|
| 2023 | CODASPY'23         | [Mehdi et al.](https://dl.acm.org/doi/abs/10.1145/3579987.3586567)| GPT-2                  | Perturbation Evasion|   |   | ✓ | Task-Adaptation        | 
| 2022 | BWCCA              | [Guo et al.](https://link.springer.com/chapter/10.1007/978-3-031-20029-8_26) | BART, GPT-2   | Phishing Indicators| ✓ |   | ✓ | Task-Adaptation  |
| 2022 | arXiv              | [Karanjai et al.](https://arxiv.org/abs/2301.00665)              | GPT-2, etc.           | Phishing Indicators                               |   | ✓ | ✓ | Task-Adaptation         |
| 2024 | ICLR               | [Panda et al.](https://arxiv.org/abs/2403.00871)                | GPT-4                  | Prompt Leakage                                     | ✓ |   | ✓ | Task-Adaptation         |
| 2024 | ACM TALLIP         | [Guo et al.](https://dl.acm.org/doi/abs/10.1145/3670402)                    | LLMs (unspecified)     | Phishing Indicators                               |   | ✓ | ✓ | Task-Adaptation         | 
| 2024 | Computers          | [Wani](https://search.ebscohost.com/login.aspx?direct=true&profile=ehost&scope=site&authtype=crawler&jrnl=2073431X&AN=180558443&h=RJRe0AMI1v6u7kTHecJDx4GnVBiNYseiozuh1x%2FziOCv0GzxZ34RmuOAfqDuVXqTrPwhuPs1%2BpohNPb1wvBsSA%3D%3D&crl=c)                            | ChatGPT                | Phishing Indicators; Prompt Engineering           |   | ✓ |   | Task-Adaptation         | 
### Knowledge Augmented
| Year | Venue              | Paper                           | Threats Model          | Exploitations                                     | Data | Prompt | Model | Methods                 | 
|------|--------------------|---------------------------------|------------------------|---------------------------------------------------|---|---|---|-------------------------|
| 2023 | arXiv              | [Hazell](https://arxiv.org/abs/2305.06972)                          | GPT-3.5/4              | Persuasion Principles; Phishing Indicators        |   | ✓ |   | Knowledge Augmented     | 
| 2025 | arXiv              | [Perik et al.](http://essay.utwente.nl/104867/)                     | GPT-4                  | Prompt Engineering                                |   | ✓ |   | Knowledge Augmented     | 
| 2025 | arXiv              | [Bi et al.]                       | Multi-LLM (unspecified)| Persuasion Principles                             |   | ✓ | ✓ | Knowledge Augmented     | 

## Prompt-Guided
### Human Crafted Prompting
| Year | Venue              | Paper                           | Threats Model          | Exploitations                                     | Data | Prompt | Model | Methods                 | 
|------|--------------------|---------------------------------|------------------------|---------------------------------------------------|---|---|---|-------------------------|
| 2023 | ECAI               | [Emanuela et al.](https://ieeexplore.ieee.org/abstract/document/10607415/)                 | ChatGPT                | Persuasion Principles                             |   | ✓ |   | Human Crafted Prompting | 
| 2023 | IEEE Access        | [Heiding et al.](https://arxiv.org/abs/2308.12287)                  | GPT-4                  | Phishing Indicators                               |   | ✓ |   | Human Crafted Prompting | 
| 2023 | FTC                | [Langford et al.](https://link.springer.com/chapter/10.1007/978-3-031-47454-5_13)                 | ChatGPT                | Persuasion Principles                             |   | ✓ |   | Human Crafted Prompting | 
| 2024 | ICML Workshop      | [Heiding et al.](https://arxiv.org/abs/2412.00586)                  | GPT-4, etc.            | Prompt Leakage                                     | ✓ |   |   | Human Crafted Prompting | 
| 2023 | DependSys 2023     | [Utaliyeva et al.](https://ieeexplore.ieee.org/abstract/document/10466952/)                 | ChatGPT                | Prompt Engineering; Paraphrasing Attack           | ✓ | ✓ |   | Human Crafted Prompting | 
| 2023 | arXiv              | [Patel et al.](https://www.inuit.se/hubfs/WithSecure/files/WithSecure-Creatively-malicious-prompt-engineering.pdf)                    | ChatGPT                | Chatbot Conversations                             |   | ✓ |   | Human Crafted Prompting | 
| 2023 | BigData            | [Singh et al.](https://ieeexplore.ieee.org/abstract/document/10386814/)                    | Claude, BART, etc.     | Persuasion Principles                             |   | ✓ |   | Human Crafted Prompting | 
| 2024 | BigData            | [Afane et al.]                    | GPT-3.5, GPT-4, etc.   | Paraphrasing                                       |   | ✓ |   | Human Crafted Prompting | 
| 2024 | ITASEC             | [Greco et al.](https://www.researchgate.net/profile/Andrea-Esposito-12/publication/379697853_David_versus_Goliath_Can_Machine_Learning_Detect_LLM-Generated_Text_A_Case_Study_in_the_Detection_of_Phishing_Emails/links/661f8f2843f8df018d14631b/David-versus-Goliath-Can-Machine-Learning-Detect-LLM-Generated-Text-A-Case-Study-in-the-Detection-of-Phishing-Emails.pdf)                    | WormGPT                | Prompt Engineering                                |   | ✓ |   | Human Crafted Prompting |
| 2024 | arXiv              | [Ekekihl](https://www.diva-portal.org/smash/record.jsf?pid=diva2:1879930)                         | ChatGPT                | Phishing Indicators                               |   | ✓ |   | Human Crafted Prompting | 
| 2024 | Electronics        | [Eze](https://www.mdpi.com/2079-9292/13/10/1839)                             | DeepAI                 | Phishing Indicators; Prompt Engineering           |   | ✓ |   | Human Crafted Prompting |
| 2024 | arXiv              | [Francia et al.](https://arxiv.org/abs/2406.13049)                   | GPT-4                  | Phishing Indicators                               |   | ✓ |   | Human Crafted Prompting | 
| 2024 | IEEE Access        | [Bethany et al.](https://arxiv.org/abs/2401.09727)                  | GPT-4                  | Persuasion Principles                             | ✓ | ✓ |   | Human Crafted Prompting | 
| 2024 | arXiv              | [Carelli et al.](https://www.researchgate.net/profile/Alessandro-Carelli-2/publication/382648027_Finding_Differences_Between_LLM-generated_And_Human-written_Text_A_Phishing_Emails_Case_Study/links/66a79ca14433ad480e845754/Finding-Differences-Between-LLM-generated-And-Human-written-Text-A-Phishing-Emails-Case-Study.pdf)                   | GPT-3.5, WormGPT       | Persuasion Principles                             |   | ✓ |   | Human Crafted Prompting |
| 2024 | BDAI               | [Mahendru et al.](https://ieeexplore.ieee.org/abstract/document/10692765/)                 | GPT-4                  | Persuasion Principles                             |   | ✓ |   | Human Crafted Prompting |
| 2024 | S&P                | [Roy et al.](https://ieeexplore.ieee.org/abstract/document/10646856/)                      | GPT-4, Claude, etc.    | Prompt Mining                                      | ✓ | ✓ |   | Human Crafted Prompting | 
| 2024 | SPW                | [Kang et al.](https://ieeexplore.ieee.org/abstract/document/10579515/)                      | GPT2-XL, ChatGPT       | Prompt Leakage                                     |   | ✓ |   | Human Crafted Prompting |
| 2024 | arXiv              | [Chen et al.](https://arxiv.org/abs/2411.11389)                     | Llama 3.1              | Persuasion Principles; Subject Diversity          | ✓ | ✓ |   | Human Crafted Prompting |

### Model-Crafted Prompting
| Year | Venue              | Paper                           | Threats Model          | Exploitations                                     | Data | Prompt | Model | Methods                 | 
|------|--------------------|---------------------------------|------------------------|---------------------------------------------------|---|---|---|-------------------------|
| 2024 | ISDFS              | [Shibli et al.]                   | GPT-3.5                | Prompt Engineering                                |   | ✓ |   | Model Crafted Prompting |
| 2025 | arXiv              | [Xue et al.]                      | GPT-4o                 | Prompt Engineering                                | ✓ |   |   | Model Crafted Prompting | 
| 2025 | arXiv              | [Sniegowski et al.]               | GPT-4o, Llama 3.2      | Prompt Engineering                                |   | ✓ |   | Model Crafted Prompting | 
| 2025 | AsiaCCS            | [Weinz et al.]                    | LLMs (unspecified)     | Prompt Engineering                                |   | ✓ |   | Model Crafted Prompting | 
| 2025 | SPIE               | [Young et al.]                    | Mistral 7B             | Prompt Mining                                      |   | ✓ |   | Model Crafted Prompting | 

## Adversaially Guided
### Adversarial Training
| Year | Venue              | Paper                           | Threats Model          | Exploitations                                     | Data | Prompt | Model | Methods                 | 
|------|--------------------|---------------------------------|------------------------|---------------------------------------------------|---|---|---|-------------------------|
| 2021 | IET Conf. Proc.    | [Khan et al.](https://digital-library.theiet.org/doi/abs/10.1049/icp.2021.2422)                     | GPT-2                  | Phishing Indicators                               | ✓ |   |   | Adversarial Training     | 
| 2024 | BigData            | [Fairbanks et al.](https://ieeexplore.ieee.org/abstract/document/10825007/)                | GPT-3.5                | Paraphrasing Attack                                |   |   | ✓ | Adversarial Training     | 

| 2024 | Gryka et al.      | GPT-3.5-turbo        | Phishing Indicators                    |     | ✓   |     | General Phishing Emails         |[Link](https://dl.acm.org/doi/abs/10.1145/3664476.3670465) |


# RQ2 Pattern Analysis of Emerging Phishing Attacks Generated by LLMs
## Text Traits
### Textual Characteristics

| Year | Venue               | Paper                                                   | Model | Text | User | Analytical Approaches     | Prompt | Config | Version | MultiGen |
|------|---------------------|---------------------------------------------------------|:-----:|:----:|:----:|---------------------------|:------:|:------:|:-------:|:--------:|
| 2024 | BigData             | [Afane et al.](#afane2024next)                          |       |  ✓   |      | Experimental Evaluation   |        |   ✓    |         |    ✓     |
| 2024 | HOLISTICA           | [Bouchareb et al.](#bouchareb2024analyzing)             |       |  ✓   |      | User-Centric Analysis     |   ✓    |        |         |          |
| 2024 | arXiv               | [Carelli et al.](#carelli2024finding)                   |       |  ✓   |      | Conceptual Analysis       |   ✓    |        |    ✓    |          |
| 2024 | Electronics         | [Eze et al.](#eze2024analysis)                          |       |  ✓   |      | User-Centric Analysis     |   ✓    |        |         |          |
| 2024 | IEEE Access         | [Heiding et al.](#heiding2023devising)                  |   ✓   |  ✓   |      | User-Centric Analysis     |   ✓    |        |         |    ✓     |
| 2024 | ICML Workshop       | [Heiding et al.](#heiding2024evaluating)                |   ✓   |  ✓   |      | User-Centric Analysis     |   ✓    |        |         |    ✓     |

### Social Engineering Tactics
| Year | Venue               | Paper                                                   | Model | Text | User | Analytical Approaches     | Prompt | Config | Version | MultiGen |
|------|---------------------|---------------------------------------------------------|:-----:|:----:|:----:|---------------------------|:------:|:------:|:-------:|:--------:|
| 2024 | arXiv               | [Chen et al.](#chen2024adapting)                        |       |  ✓   |      | Conceptual Analysis       |   ✓    |        |         |          |
| 2024 | ECAI                | [Emanuela et al.](#emanuela2024ai)                      |       |  ✓   |      | User-Centric Analysis     |   ✓    |        |         |          |

## Human Factors
### Individual Characteristics
| Year | Venue               | Paper                                                   | Model | Text | User | Analytical Approaches     | Prompt | Config | Version | MultiGen |
|------|---------------------|---------------------------------------------------------|:-----:|:----:|:----:|---------------------------|:------:|:------:|:-------:|:--------:|
| 2025 | arXiv               | [Francia et al.](#francia2024assessing)                 |       |  ✓   |  ✓   | User-Centric Analysis     |   ✓    |        |         |          |
| 2024 | MCNA                | [Alahmed et al.](#alahmed2024exploring)                 |   ✓   |  ✓   |  ✓   | User-Centric Analysis     |        |        |         |          |

### Psychological Characteristics
| Year | Venue               | Paper                                                   | Model | Text | User | Analytical Approaches     | Prompt | Config | Version | MultiGen |
|------|---------------------|---------------------------------------------------------|:-----:|:----:|:----:|---------------------------|:------:|:------:|:-------:|:--------:|
| 2023 | IJCIC               | [Asfour et al.](#asfour2023harnessing)                  |       |      |  ✓   | Experimental Evaluation   |   ✓    |        |         |          |
| 2023 | EuroS&PW            | [Sharma et al.](#sharma2023well)                        |       |      |  ✓   | Experimental Evaluation   |        |        |         |          |
| 2024 | AIJMR               | [Bharati et al.](#bharati2024ai)                        |       |  ✓   |      | Conceptual Analysis       |   ✓    |        |         |          |

## Model Traits
### Compotentional Efficiency
| Year | Venue               | Paper                                                   | Model | Text | User | Analytical Approaches     | Prompt | Config | Version | MultiGen |
|------|---------------------|---------------------------------------------------------|:-----:|:----:|:----:|---------------------------|:------:|:------:|:-------:|:--------:|
| 2024 | Artif. Intell. Rev. | [Schmitt et al.](#schmitt2024digital)                   |   ✓   |  ✓   |  ✓   | Conceptual Analysis       |        |        |         |          |


# RQ3 Defense and Detection Techniques against LLM-Enabled Textual Phishing Campaigns



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
