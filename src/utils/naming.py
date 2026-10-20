import os

from src.config import Config
cfg = Config()


def pretrained_model_location(model_name=cfg.model):
    return f"models/{model_name}/pretrained.pth"


def model_location(checkpoint_number, model_name=cfg.model):
    return f"models/{model_name}/epoch_{checkpoint_number}.pth"


def results_location(model_name=cfg.model):
    job_id = os.environ.get("JOB_ID", "local")
    return f"results/{model_name}/{job_id}"