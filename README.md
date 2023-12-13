# Formalizing a Thread Model for Adversarial Attacks in Information Retrieval
This reposititory contain the work of Dennis Agafonov and Jelke Matthijsse For the course Information Retrieval 2, given at the University of Amsterdam (2023). 

This README contains all information necessary to reproduce the work as proposed in the "Formalizing a Thread Model for Adversarial Attacks in Information Retrieval" paper [[1]](#1). First and overview of the file structure is given. Then, instructions on the requirement are provided. And lastly, this README provides detailed instructions on how to run the experiments from the paper. 

## Directory structure

This project uses the Top1000 Dev subset of the MS MARCO Dataset for Passage Retrieval. This dataset can be downloaded from the official MS MARCO github: [github:ms marco](https://microsoft.github.io/msmarco/). Make sure the data is available in the directory where the rest of the repository is stored.

TODO --> Github structuren en hier goed inzetten (maar dat komt later)
```tree
├── Collision files
|    └── Semantic collision.py
|    └── maybe nog wat anders.py
├── Encoding files
|    └── Semantic collision.py
|    └── maybe nog wat anders.py
├── requirements.txt
├── evaluation.py
└── main.py
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
To reproduce the experiments it is possible to run the main.py file. This method perturbs the most irrelevant documents from an original ranking (as provided by the CrossEncoder model) to produce a new ranking. By default, this method uses the semantic collisions method. This reranking can be evaluated using the `evaluation.py` file TODOOOO.

The syntax for running this file is as follows, where the choice of method, number of irrelevant documents and top documents can be made. 

```txt
main.py [-h] [--use-cuda] [--perturbation-method METHOD] [--no-irrelevant-docs IRR_DOCS][--no-top-docs TOP_DOCS] 
```
## File description
- `main.py`: running the re-ranking with adversarial attacks for one of the two perturbation methods
- `evaluation.py`: evaluating the new ranking TODO!
- `perturb_doc.py`: perturbs documents based on the adversarial encoding attack
- `semantic_collision.py`: generates collision based on the semantic collision adversarial attack

## References
<a id="1">[1]</a> 
LINK TO OUR PAPER
