import os
import logging.config
import yaml

def create_project_structure():
    directories = [
        "config",
        "data/raw",
        "data/processed",
        "data/simulated",
        "logs",
        "models",
        "notebooks",
        "scripts",
        "utils"
    ]
    files = {
        "config/config.yaml": "# Logging and other configurations\n",
        "config/model_config.yaml": "# Model parameters for ARIMA, LSTM, etc.\n",
        "logs/app.log": "",
        "models/arima_model.py": "# ARIMA model logic\n",
        "models/lstm_model.py": "# LSTM model logic\n",
        "models/hybrid_model.py": "# Hybrid model logic combining all techniques\n",
        "scripts/data_preprocessing.py": "# Data cleaning and preprocessing logic\n",
        "scripts/monte_carlo.py": "# Monte Carlo simulation logic\n",
        "scripts/spline_regression.py": "# Spline regression logic\n",
        "utils/logger.py": "# Logger setup using dictConfig\n",
        "utils/data_loader.py": "# Data loading and saving utilities\n",
        "main.py": "# Main script to orchestrate the entire project\n",
        "requirements.txt": "numpy\npandas\nscipy\nmatplotlib\ntensorflow\nstatsmodels\npmdarima\nyaml\n",
        "README.md": "# Stock Price Prediction Project\n"
    }
    # Create directories
    for directory in directories:
        os.makedirs(os.path.join(directory), exist_ok=True)

    # Create files
    for filepath, content in files.items():
        full_path = os.path.join(filepath)
        with open(full_path, "w") as f:
            f.write(content)

def setup_logging_config():
    logging_config = {
        "version": 1,
        "formatters": {
            "detailed": {
                "format": "[%(asctime)s]: Line No.: %(lineno)d - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "formatter": "detailed",
                "filename": "logs/app.log",
                "mode": "a"
            },
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "detailed",
                "stream": "ext://sys.stdout"
            }
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["file", "console"]
        }
    }
    with open("config/config.yaml", "w") as f:
        yaml.dump(logging_config, f)

#if __name__ == "__main__":
#    create_project_structure()
#    setup_logging_config()
#    print(f"Project structure created")