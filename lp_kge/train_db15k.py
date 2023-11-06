from pykeen.pipeline import pipeline_from_config
from pykeen.utils import load_configuration, normalize_path


def train_DB15K_rotate(work_dir):
    path = normalize_path(f"settings/RotatE_DB15K.json")
    config = load_configuration(path)
    pipeline_result = pipeline_from_config(config=config)
    pipeline_result.save_to_directory(work_dir + "/checkpoint/", save_metadata=True)
    return pipeline_result


if __name__ == '__main__':
    train_DB15K_rotate("../outputs/db15k/rotate/")