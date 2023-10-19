# Table of Contents
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Results](#results)

# Installation
To install necessary dependencies, run the following command:
```bash
conda env create -f environment.yml
```

# Data
Data path: `./data/`

# Usage
## Running sLLMs:
args:
- -v: variant, i.e., distilbert, BERT, RoBERTa, XLNet, ALBERT
- -d: dataset name, i.e., app, code (Gerrit dataset), github, so (StackOverflow dataset), and jira.

```bash
python run_slm.py -v distilbert -d app
```

## Running LLMs
args:
- -d: dataset name, i.e., app, code (Gerrit dataset), github, so (StackOverflow dataset), and jira.
- -m: model name, i.e. llama2, wizardlm, vicuna
- -p: prompt template

```bash
python run_llm.py -d app -m llama2 -p llama2-1 -s 1
```

## Evaluation
args:
- -d: dataset name, i.e., app, code (Gerrit dataset), github, so (StackOverflow dataset), and jira.
- -m: model name, i.e. llama2, wizardlm, vicuna
- -p: prompt template
- -s: shots, i.e., 0 for zero-shot, 1 for one-shot, 3 for three-shot, and 5 for five-shot.

```bash
python eval.py -d app -m vicuna -p vicuna-0 -s 0
```

## Results
The results are saved in `./results/`.
For instance, for the results of LLaMA2 on the APP dataset, the directory structure is as follows:

📦results
 ┣ 📂app
 ┃ ┣ 📂llama2
 ┃ ┃ ┣ 📂few-shot
 ┃ ┃ ┃ ┗ 📂llama2
 ┃ ┃ ┃ ┃ ┣ 📂1
 ┃ ┃ ┃ ┃ ┣ 📂3
 ┃ ┃ ┃ ┃ ┣ 📂5
 ┃ ┃ ┗ 📂zero-shot
 ┃ ┃ ┃ ┣ 📂llama2-0
 ┃ ┃ ┃ ┣ 📂llama2-1
 ┃ ┃ ┃ ┣ 📂llama2-2


## Additional Scripts
- `draw_figures.py`: draw figures for the paper.
- `preprocess.py`: preprocess the dataset.
- `analyze_errors.py`: analyze the errors of the models.
- `sample_few_shot.py`: sample few-shot examples from the dataset.
