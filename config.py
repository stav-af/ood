from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # data
    data_root: str = "./data"
    imagenet_root: str = "./imagenet"
    num_workers: int = 8
    batch_size: int = 256

    # training
    epochs: int = 30
    lr: float = 3e-4
    wd: float = 0.05

    # model
    num_classes: int = 101
    ood_label: int = 100

    # misc
    seed: int = 42
