.PHONY: all clean install data-pipeline data-pipeline-rebuild training-pipeline streaming-inference run-all mlflow-ui mlflow-clean promote-to-staging promote-to-production help

# Default Python interpreter
PYTHON = python
VENV = .venv/bin/activate

# MLflow settings
MLFLOW_HOST = 127.0.0.1
MLFLOW_PORT = 5000
MODEL_NAME = churn_prediction_model

# Default target
all: help

# Help target
help:
	@echo "Available targets:"
	@echo "  make install                - Install project dependencies and set up environment"
	@echo "  make data-pipeline          - Run the data pipeline"
	@echo "  make data-pipeline-rebuild  - Force rebuild data pipeline from scratch"
	@echo "  make training-pipeline      - Run the training pipeline"
	@echo "  make streaming-inference    - Run streaming inference with sample data"
	@echo "  make run-all                - Run all pipelines in sequence (data + training)"
	@echo "  make mlflow-ui              - Launch MLflow tracking UI"
	@echo "  make mlflow-clean           - Remove MLflow tracking data (mlruns/)"
	@echo "  make promote-to-staging     - Promote latest model to Staging (VERSION=X)"
	@echo "  make promote-to-production  - Promote model to Production (VERSION=X)"
	@echo "  make clean                  - Clean up artifacts and temporary files"
	@echo "  make clean-all              - Clean artifacts, MLflow data, and remove virtual environment"

# Install project dependencies and set up environment
install:
	@echo "Installing project dependencies and setting up environment..."
	@echo "Creating virtual environment..."
	@python3 -m venv .venv
	@echo "Activating virtual environment and installing dependencies..."
	@source .venv/bin/activate && pip install --upgrade pip
	@source .venv/bin/activate && pip install -r requirements.txt
	@echo "Installation completed successfully!"
	@echo "To activate the virtual environment, run: source .venv/bin/activate"

# Clean up artifacts
clean:
	@echo "Cleaning up artifacts and temporary files..."
	rm -rf artifacts/data/*
	rm -rf artifacts/models/*
	rm -rf artifacts/encode/*
	rm -f temp_imputed.csv
	rm -rf data/processed/*
	@echo "Cleanup completed!"

# Clean MLflow tracking data
mlflow-clean:
	@echo "Removing MLflow tracking data..."
	rm -rf mlruns/
	@echo "MLflow data cleaned!"

# Clean everything including virtual environment and MLflow
clean-all: clean mlflow-clean
	@echo "Removing virtual environment..."
	rm -rf .venv
	rm -rf src/__pycache__
	rm -rf utils/__pycache__
	rm -rf pipelines/__pycache__
	@echo "Full cleanup completed!"

# Run data pipeline
data-pipeline:
	@echo "Start running data pipeline..."
	@source $(VENV) && $(PYTHON) pipelines/data_pipeline.py
	@echo "Data pipeline completed successfully!"

# Force rebuild data pipeline
data-pipeline-rebuild:
	@echo "Cleaning artifacts before rebuild..."
	@make clean
	@echo "Running data pipeline with force rebuild..."
	@source $(VENV) && $(PYTHON) -c "from pipelines.data_pipeline import data_pipeline; data_pipeline(force_rebuild=True)"
	@echo "Data pipeline rebuild completed successfully!"

# Run training pipeline
training-pipeline:
	@echo "Start running training pipeline..."
	@source $(VENV) && $(PYTHON) pipelines/training_pipeline.py
	@echo "Training pipeline completed successfully!"

# Run streaming inference
streaming-inference:
	@echo "Start running streaming inference..."
	@source $(VENV) && $(PYTHON) pipelines/streaming_inference_pipeline.py
	@echo "Streaming inference completed successfully!"

# Run all pipelines in sequence
run-all:
	@echo "Running all pipelines in sequence..."
	@echo "========================================"
	@echo "Step 1: Running data pipeline"
	@echo "========================================"
	@source $(VENV) && $(PYTHON) pipelines/data_pipeline.py
	@echo "\n========================================"
	@echo "Step 2: Running training pipeline"
	@echo "========================================"
	@source $(VENV) && $(PYTHON) pipelines/training_pipeline.py
	@echo "\n========================================"
	@echo "All pipelines completed successfully!"
	@echo "========================================"

# Launch MLflow UI
mlflow-ui:
	@echo "Launching MLflow tracking UI..."
	@echo "Access the UI at: http://$(MLFLOW_HOST):$(MLFLOW_PORT)"
	@source $(VENV) && mlflow ui --host $(MLFLOW_HOST) --port $(MLFLOW_PORT)

# Promote model to Staging
promote-to-staging:
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION not specified. Usage: make promote-to-staging VERSION=1"; \
		exit 1; \
	fi
	@echo "Promoting model $(MODEL_NAME) version $(VERSION) to Staging..."
	@source $(VENV) && mlflow models transition-model-version-stage \
		--name $(MODEL_NAME) \
		--version $(VERSION) \
		--stage Staging \
		--archive-existing-versions
	@echo "Model promoted to Staging successfully!"

# Promote model to Production
promote-to-production:
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION not specified. Usage: make promote-to-production VERSION=1"; \
		exit 1; \
	fi
	@echo "Promoting model $(MODEL_NAME) version $(VERSION) to Production..."
	@source $(VENV) && mlflow models transition-model-version-stage \
		--name $(MODEL_NAME) \
		--version $(VERSION) \
		--stage Production \
		--archive-existing-versions
	@echo "Model promoted to Production successfully!"
