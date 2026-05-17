import numpy as np
import torch
import torch as tt
import torch.nn as nn
import statistics
from torch.utils.data import Dataset, DataLoader, Subset
import os
import csv


from rex_xai.lib import ReX
from rex_xai.input.input_data import Data
from rex_xai.output import visualisation

from train import evaluate
from data import _get_mean_std

from PIL import Image

def tensor_to_pil(x, mean, std):
    # x: [3, H, W]
    x = x.detach().cpu().clone()
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    x = x * std + mean
    x = x.clamp(0, 1)
    x = (x * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x)



def record(data, path="results/ood.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    write_header = not os.path.exists(path) or os.path.getsize(path) == 0

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(data)

def measure(model, data, path='results.csv'):
    device = 'cuda' if torch.cuda.is_available() else Exception("not")

    model.eval().to(device)

    results = []
    correct = 0
    total = 0

    subset = Subset(data, range(min(100, len(data))))

    for i, (x, y) in enumerate(subset):
        x = x.to(device)

        y = torch.as_tensor(y, device=device)

        rex = ReX(model, ["N", 3, 224, 224], device, "RGB")

        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]

        d = Data(
            input=tensor_to_pil(
                x,
                mean=mean,
                std=std,
            ),
            model_shape=["N", 3, 224, 224],
            device=device,
            mode="RGB",
            process=False,
        )

        if x.ndim == 3:
            x = x.unsqueeze(0)

        d.data = x
        rex.data = d

        rex.set_target().show_target()
        print(rex.data.target.__dict__)

        # Accuracy from ReX target
        pred = torch.as_tensor(rex.data.target.classification, device=device)
        correct += (pred == y).item()

        print(pred, y)

        total += 1

        rex.args.minimum_confidence_threshold = 0.5
        rex.args.batch_size = 64
        rex.args.seed = 0
        rex.calculate_responsibility()
        rex.generate_explanation_object()

        rex.extract_sufficient_explanation()
        mask = rex.explanation.sufficiency_mask

        # if not i: visualisation.surface_plot(rex.data.input,rex.args,rex.explanation.target_map,rex.data.target)
        size = torch.count_nonzero(mask).item() / mask.numel()
        results.append(size)

        if len(results) > 2:
            print("mean: ", sum(results) / len(results))
            print("stdev:", statistics.stdev(results))
            print("accuracy:", correct / total)

    mean = sum(results) / len(results)
    stdev = statistics.stdev(results) if len(results) > 1 else 0.0
    accuracy = correct / total

    record({
        "mean": mean,
        "stdev": stdev,
        "accuracy": accuracy,
    }, f"results/{path}")