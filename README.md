# Formalizing a Thread Model for Adversarial Attacks in Information Retrieval
This reposititory contain the work of Dennis Agafonov and Jelke Matthijsse For the course Information Retrieval 2, given at the University of Amsterdam (2023). 

This README contains all information necessary to reproduce the work as proposed in the "Formalizing a Thread Model for Adversarial Attacks in Information Retrieval" paper [[1]](#1).

## Directory structure

This project uses the Top1000 Dev subset of the MS MARCO Dataset for Passage Retrieval (top1000.dev). This dataset can be downloaded from the official MS MARCO github: [github:ms marco](https://microsoft.github.io/msmarco/). The relevance labels (qrels.dev.tsv) can also be downloaded from this source. Additionally, the language model and collision model, as used and provided by Song et al. (2020) [[2]](#2), used can be downloaded from here: https://drive.google.com/drive/folders/1XRwWZLgs1Pm_mbl16wyXoXo9q-Sbb4O6?usp=sharing. These models can also be downloaded from the original Semantic Collisions git page: https://github.com/csong27/collision-bert.

```tree
├── methods/
|    └── semantic_collisions.py
|    └── perturb_doc.py
├── models/
|    └── bert_layers.py
|    └── bert_models.py
|    └── bert/<language model>
|    └── msmarco_mb/<collision model>
├── data/
|    └── qrels.dev.tsv
|    └── top1000.dev
├── main.py
├── evaluation.py
├── dataloader.py
└── requirements.txt
```

## Environment and requirements
We have provided a `requirements.txt` file that contains all `pip` packages that are necessary for running the experiments. A virtual environment can be created with an environment manager of choice, e.g. conda, where the requirements can be installed the following way:

```sh
# Create env
conda create -n ir2 python=3.11 && conda activate ir2
# Install requirements
pip install -r requirements.txt
```

## Experiments
To reproduce the experiments, please run the `main.py` file. After selecting the desired perturbation method (and any other arguments) in the argparser, this method will perturb the most irrelevant documents from an original ranking (as provided by the CrossEncoder model) to produce a new ranking. Resulting metrics will be stored in results/results.txt, in the following order: Average success rate, average shift, average nDCG before the attack, average nDCG after the attack, difference in average nCDG.

## References
<a id="1">[1]our paper</a> 

<a id="2">[2] Song et al. (2020). Adversarial Semantic Collisions.</a> 
