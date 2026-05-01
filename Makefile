.PHONY: venv install clean docker run

PYTHON ?= python3
VENV   ?= .venv

venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	@echo "→ activate with: source $(VENV)/bin/activate"

install:
	$(PYTHON) -m pip install --user -r requirements.txt

clean:
	rm -rf $(VENV) __pycache__ *.pyc _warped.png template_overlay.png

docker:
	docker build -t omr-reader .

# Process the directory in $DIR using the Docker image.
# Example: make run DIR=/path/to/scans
run:
	docker run --rm \
	    -v "$(DIR)":/data \
	    omr-reader /data
