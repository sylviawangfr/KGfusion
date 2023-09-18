from pykeen.pipeline import pipeline_from_config
from pykeen.utils import load_configuration, normalize_path


def train_transR():
    path = normalize_path(f"settings/TransR_DB15K.json")
    config = load_configuration(path)
    pipeline_result = pipeline_from_config(config=config)
    return pipeline_result


if __name__ == '__main__':
    train_transR()