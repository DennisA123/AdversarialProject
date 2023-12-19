# Formalizing a Thread Model for Adversarial Attacks in Information Retrieval
This reposititory contain the work of Dennis Agafonov and Jelke Matthijsse for the course Information Retrieval 2, given at the University of Amsterdam (2023). This work formalizes a general threat model that allows comparison of different adversarial attack methods as proposed in IR literature. This work focuses primarily on two existing adversarial attack methods: Semantic Collisions by Song et al. (2020) [[1]](#1), and Encoding Attack by Boucher et al. (2023) [[2]](#2). Much of their code (available at [this link](https://github.com/csong27/collision-bert) and [this link](https://github.com/nickboucher/search-engine-attacks), respectively) has been used and altered for this work.

This README contains all information necessary to reproduce the work as proposed in the "Towards a General Threat Model for Adversarial
Attacks in Information Retrieval" paper.

## Directory structure

This project uses the Top1000 Dev subset of the MS MARCO Dataset for Passage Retrieval (top1000.dev). This dataset can be downloaded from the [official MS MARCO GitHub](https://microsoft.github.io/msmarco/). The relevance labels (qrels.dev.tsv) can also be downloaded from this source. Additionally, the language model and collision model, as used and provided by Song et al. (2020) [[1]](#1), used can be downloaded from [here](https://drive.google.com/drive/folders/1XRwWZLgs1Pm_mbl16wyXoXo9q-Sbb4O6?usp=sharing). These models can also be downloaded from the original [Semantic Collisions GitHub page](https://github.com/csong27/collision-bert).

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
To reproduce the experiments, please run the `main.py` file. After selecting the desired perturbation method (and any other arguments) in the argparser, this method will perturb the most irrelevant documents from an original ranking (as provided by the CrossEncoder model) to produce a new ranking.

Argparse arguments:
- `perturbation_method`: use Aggressive Semantic Collisions (ASC) or Encoding Attack (EA)
- `nr_irrelevant_docs`: how many irrelevant documents (as provided by R) should be perturbed
- `nr_words`: length of the perturbation (number of tokens adapted or added in original document)
- `perturbation_type` [for EA only]: what kind of perturbation implementation method is used
- `choice_of_words` [for EA only]: how the tokens to be perturbed are selected
- `verbosity`: if parsed, will print additional information during the adversarial attack process
- `max_iter` [for ASC only]: how many iterations should be done to find the best collision

 Resulting metric values will be stored in results/results-{perturbation size}.txt, in the following order: average normalized rank shift, average nDCG before the attack, average nDCG after the attack, difference in average nCDG.

## References

<a id="1">[1] Song et al. (2020). Adversarial Semantic Collisions.</a> 

<a id="2">[2] Boucher et al. (2023). Boosting Big Brother: Attacking Search Engines with Encodings.</a> 
