.PHONY: train

check: black flake

train: 
	python examples/train.py \
		data/raw/210119_en_deter_preprocessed.json \
		models/pq-model --embedding-model \
		doc2vec --training-speed deep-learn --workers 128

install:
	python -m pip install -r requirements.txt

install-dev: install
	python -m pip install -r requirements-dev.txt

black:
	black examples/train.py

flake:
	flake8 --ignore=E501 examples/train.py

