.PHONY: install ui run test benchmark

install:     ## Python deps + UI deps
	pip install -r requirements.txt
	cd web && npm install

ui:          ## Build the React console (served by FastAPI at /)
	cd web && npm run build

run:         ## Start the API + console on http://localhost:8000
	uvicorn app.main:app --reload --port 8000

test:        ## Unit tests (no models or API key needed)
	pytest -q

benchmark:   ## Compare pipelines on the sample dataset
	python -m app.cli benchmark --dataset sample
