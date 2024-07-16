import yaml
import os

# Get the path of the script and the configuration file
script_dir = os.path.dirname(os.path.abspath(__file__))
config_file_name = "config.yml"
config_path = os.path.join(script_dir, config_file_name)


def load_config() -> dict:
    """Load the configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


config = load_config()
