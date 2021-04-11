install:
	python -m pip install -r requirements.txt

install-dev: install
	python -m pip install -r requirements-dev.txt

black:
	black examples/train.py

flake:
	flake8 --ignore=E501 examples/train.py

check: black flake
