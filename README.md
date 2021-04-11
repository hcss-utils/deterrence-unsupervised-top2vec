# deterrence-unsupervised-top2vec

This repository contains a top2vec model trained on 
ProQuest Deterrence dataset containing 26,525 non-empty documents.


## Usage

To use or contribute to this repository, first checkout the code. 
Then create a new virtual environment:

```console
$ git clone https://github.com/hcss-utils/deterrence-unsupervised-top2vec.git
$ cd deterrence-unsupervised-top2vec
$ python3 -m venv env
$ . env/bin/activate
$ pip install -r requirements.txt
```

To train the model, run [`train.py`](examples/train.py): 

```console
$ python examples/train.py --help
```

The model that we're using for the paper was trained as follows:
```console
$ python examples/train.py data/raw/210119_en_deter_preprocessed.json models/pq-model doc2vec --speed deep-learn --workers 128
```

To replicate out analysis, follow [`examples/notebooks`](examples/) in consecutive order

## Project Organization
```console
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── examples           <- Jupyter notebooks mostly. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0_hp_initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    └── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
                              generated with `pip freeze > requirements.txt`
```