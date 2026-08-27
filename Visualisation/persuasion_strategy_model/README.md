# Modeling Persuasive Strategies via Semi-Supervised Neural Nets on Crowdfunding Platforms

Modeling what makes a request persuasive—eliciting the desired response from a reader - is critical to the study of propaganda, behavioral economics, and advertising. Yet current models can't quantify the persuasiveness of requests or extract successful persuasive strategies. Building on theories of persuasion, we propose a neural network to quantify persuasiveness and identify the persuasive strategies in advocacy requests. Our semi-supervised hierarchical neural network model is supervised by the number of people persuaded to take actions and partially supervised at the sentence level with human-labeled rhetorical strategies. Our method outperforms several baselines, uncovers persuasive strategies—offering increased interpretability of persuasive speech and has applications for other situations with document-level supervision but only partial sentence supervision.

## Code Structure

* ./src/AttnModel.py: semi-supervised model component.    
* ./src/DataLoader.py: data loader for pre-processing and batching.     
* ./src/MessageLoss.py: semi-supervised loss functions.     
* ./src/train.py and test.py: main training and testing.

## Run

Run `python ./src/train.py --dataset-text ... --dataset-with-annotation ... --output-dir ...` to train and save the model.         
Run `python ./src/test.py` to test the model.

## Data

The mapping from label to persuasive strategies:

``` {'Other':0, 'Concreteness':1, 'Commitment':2, 'Emotional':3, 'Identity':4, 'Impact':5, 'Scarcity':6} ```

## Local Adaptation

This workspace adds an email-oriented preprocessing layer before training or inference:

- keep as many sentences as fit within a `3000`-word budget
- strip HTML / style / script blocks
- normalize URLs
- drop noisy token-list fragments such as CSS / comma-separated random identifiers
- preserve the six persuasion principles used in the paper by excluding `Other` at aggregation time

The new batch scorer is:

- `src/score_email_csv.py`

Expected training artifacts:

- `model_best.pt`
- `vocab.pkl`

The original repository does **not** include the training files referenced by `train.py`:

- `annotation_dataset.0720.public.csv`
- `persuasion_dataset_text.0720.public.csv`
- `mid2target_normal.pkl`


## Cite
If you use our tools for your work, please cite the following paper:

* Diyi Yang, Jiaao Chen, Zichao Yang, Dan Jurafsky, Eduard Hovy. "Let’s Make Your Request More Persuasive: Modeling Persuasive Strategies via Semi-Supervised Neural Nets on Crowdfunding Platforms" In NAACL. 2019.
