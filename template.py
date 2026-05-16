from pathlib import Path

# -------------------- #
# Directories to create
# -------------------- #
directories = [
    # source
    "src/components",
    "src/configuration",
    "src/cloud_storage",
    "src/data_access",
    "src/constants",
    "src/entity",
    "src/exception",
    "src/logger",
    "src/pipeline",
    "src/utils",

    # notebooks
    "notebooks",

    # data
    "data/raw",
    "data/processed",
    "data/interim",
    "data/external",

    # artifacts
    "artifacts/model",
    "artifacts/reports",
    "artifacts/logs",

    # config
    "config",
]

# -------------------- #
# Files to create
# -------------------- #
files = [
    # src files
    "src/__init__.py",

    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_validation.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluation.py",
    "src/components/model_pusher.py",

    "src/configuration/__init__.py",
    "src/configuration/mongo_db_connection.py",
    "src/configuration/aws_connection.py",

    "src/cloud_storage/__init__.py",
    "src/cloud_storage/aws_storage.py",

    "src/data_access/__init__.py",
    "src/data_access/vehicle_insurance_data.py",

    "src/constants/__init__.py",

    "src/entity/__init__.py",
    "src/entity/config_entity.py",
    "src/entity/artifact_entity.py",
    "src/entity/estimator.py",
    "src/entity/s3_estimator.py",

    "src/exception/__init__.py",
    "src/logger/__init__.py",

    "src/pipeline/__init__.py",
    "src/pipeline/training_pipeline.py",
    "src/pipeline/prediction_pipeline.py",

    "src/utils/__init__.py",
    "src/utils/main_utils.py",

    # notebook
    "notebooks/Used_Cars_Analysis.ipynb",

    # root files
    "app.py",
    "demo.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "setup.py",
    "pyproject.toml",

    # config files
    "config/model.yaml",
    "config/schema.yaml",
]

# -------------------- #
# Create directories
# -------------------- #
for dir_path in directories:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# -------------------- #
# Create files
# -------------------- #
for file_path in files:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()
        print(f"Created: {file_path}")
    else:
        print(f"Already exists: {file_path}")