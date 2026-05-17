
import re
import os

import torch
import timm

from train import train
from data import get_base, get_ood
from evaluate import measure

from dataclasses import dataclass
from config import Config

from torch.utils.data import ConcatDataset, Subset
import torch.multiprocessing

# avoid overloading filedescriptor count
torch.multiprocessing.set_sharing_strategy('file_system')
cfg = Config()

def checkpoint(models_dir="models") -> int:
    pat = re.compile(r"^ood_(\d+)\.pth$")
    nums = []

    for name in os.listdir(models_dir):
        m = pat.match(name)
        if m:
            nums.append(int(m.group(1)))

    return max(nums) if nums else 0

def mix_datasets(a, b, p=0.2, total_size=None, seed=0):
    assert 0.0 <= p <= 1.0

    if total_size is None:
        if p == 0:
            total_size = len(a)
        elif p == 1:
            total_size = len(b)
        else:
            total_size = min(int(len(a) / (1 - p)), int(len(b) / p))

    n_b = int(round(total_size * p))
    n_a = total_size - n_b

    if n_a > len(a) or n_b > len(b):
        raise ValueError(
            f"Requested n_a={n_a}, n_b={n_b}, but len(a)={len(a)}, len(b)={len(b)}"
        )

    g = torch.Generator().manual_seed(seed)
    idx_a = torch.randperm(len(a), generator=g)[:n_a].tolist()
    idx_b = torch.randperm(len(b), generator=g)[:n_b].tolist()

    return ConcatDataset([
        Subset(a, idx_a),
        Subset(b, idx_b),
    ])


def main():
    device = "cuda" if torch.cuda.is_available() else Exception("No GPU detected")

    print("loading resnet")
    model = timm.create_model("resnet50", pretrained=False, num_classes=101)
    print("sending to gpu")
    model.to(device)

    for eopch in range(12):
        base_train, base_test = get_base()
        train(model, base_train, base_test, epochs=5)

    test = Subset(base_test, range(1000))   

    measure(model, test, "trained")
    ood_train, ood_test = get_ood()


    for i in range(60):
        mix_train = mix_datasets(base_train, ood_train)
        mix_test = mix_datasets(base_test, ood_test)
        
        torch.save(model.state_dict(), f"models/ood_{i}.pth")
        train(model, mix_train, mix_test, epochs=1)
    
    for i in range(0, 28):
        model = timm.create_model("resnet50", pretrained=True, num_classes=101)
        state = torch.load(f"models/ood_{i}.pth", map_location = "cpu", weights_only=False)
        
        model.load_state_dict(state)
        model.to(device)

        measure(model, test, f'trained_ood')


if __name__ == "__main__":
    main()