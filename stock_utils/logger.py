# Logger setup using dictConfig
# do not forget to use Logger instance (use like logger = get_logger(__name__) in the file, and then use logger.info("Your message") etc.)
import logging.config
import yaml

def setup_logging_config(config_path="config/config.yaml"):
    """
    Sets up logging configuration using a YAML file.

    Args:
        config_path (str): Path to the logging configuration YAML file.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logging.config.dictConfig(config)
    except Exception as e:
        print(f"Failed to load logging configuration: {e}")
        logging.basicConfig(level=logging.INFO)  # Fallback basic config

def get_logger(name):
    """
    Returns a logger instance with the specified name.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)
