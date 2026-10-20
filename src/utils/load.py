
import timm
import torch

from src.utils.naming import pretrained_model_location, model_location
from src.config import Config
cfg = Config()


def load_model(model=cfg.model, epoch=0):
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    
    print(f"Loading {model} at checkpoint {epoch}")
    location = pretrained_model_location() if epoch == 0 else model_location(epoch)

    model = timm.create_model(model, pretrained=True, num_classes=101)
    state = torch.load(location, map_location = "cpu", weights_only=False)
    
    model.load_state_dict(state)
    model.to(device)

    return model