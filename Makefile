install:
	python -m pip install -U pip setuptools wheel
	python -m pip install -r requirements.txt
	python -m pip install -e .

build-dist: ## [Continuous integration] Build package for pypi
	python3.6 -m venv .env
	. .env/bin/activate && pip install -U pip setuptools wheel
	. .env/bin/activate && python setup.py sdist
	rm -rf .env
