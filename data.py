import torch

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
from collections import defaultdict

from tqdm import tqdm
from config import Config
cfg = Config()

IN_CIFAR_CLASSES = [
    0, 1, 2, 3, 4, 5, 6, 107, 118, 119, 120, 121, 122, 124, 147, 148, 150,
    25, 26, 28, 29, 30, 31, 33, 34, 35, 36, 37,
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
    70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    300, 301, 302, 303, 304, 305, 306, 307,
    308, 309, 310, 311, 312, 314, 315, 319, 320,
    321, 322, 323, 324, 325, 326,
    333, 334, 337, 360, 361,
    269, 270, 271, 272, 273, 274, 275,
    277, 278, 279, 280,
    286, 288, 291, 292, 293,
    365, 366, 367,
    385, 386,
    80, 81, 82, 83, 84, 85, 86,
    127, 128, 129, 130,
    144, 145, 146,
    429, 430, 444, 487, 504, 610, 620, 621, 673,
    717, 779, 805, 829, 846, 847, 851, 866, 923, 968
]

dataloader_kwargs = {
    "batch_size": cfg.batch_size,
    "shuffle": True,
    "num_workers": cfg.num_workers,
    "pin_memory": True,
    "persistent_workers": (cfg.num_workers > 0),
}



class FilterByClassIdx(Dataset):
    def __init__(self, base_ds, excluded_idxs, k=10_000):
        print("initializing filtered imagenet")
        self.base_ds = base_ds
        self.excluded = set(excluded_idxs)
        self.ood_label = 100

        counts = defaultdict(int)
        self.keep_indices = []

        keep_classes = set(base_ds.targets) - self.excluded

        for i, y in enumerate(base_ds.targets):
            if y in self.excluded:
                continue
            if counts[y] < k:
                counts[y] += 1
                self.keep_indices.append(i)
            
                if all(counts[y] >= k for y in keep_classes):
                    break
        
        print("initialized filtered imagenet")

    def __len__(self):
        return len(self.keep_indices)

    def __getitem__(self, idx):
        x, _ = self.base_ds[self.keep_indices[idx]]
        return x, self.ood_label



def _get_transforms(mean, std):
    train_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, test_tf

def _get_mean_std(ds):
    # for testing
    # return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    print("calculating stats")
    assert torch.cuda.is_available(), "No cuda!"
    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    n_pixels = 0

    for x, _ in tqdm(ds):
        b, c, h, w = x.shape
        pixels = b * h * w
        channel_sum += x.sum(dim=(0, 2, 3))
        channel_sq_sum += x.square().sum(dim=(0, 2, 3))
        n_pixels += pixels

    mean = (channel_sum / n_pixels).tolist()
    std = (channel_sq_sum / n_pixels - torch.tensor(mean) ** 2).sqrt().tolist()

    print("calculated stats")
    return mean, std


def get_ood(
    imagenet_root="./imagenet",
    excluded_idxs=IN_CIFAR_CLASSES,
    batch_size=256
):
    stats_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    print("loading imagenet")
    stats_train_base = datasets.ImageNet(root=imagenet_root, split="train", transform=stats_tf)

    print("filtering imagenet")
    stats_train_ds = DataLoader(
        FilterByClassIdx(stats_train_base, excluded_idxs),
        **dataloader_kwargs
    )

    mean, std = _get_mean_std(stats_train_ds)
    train_tf, test_tf = _get_transforms(mean, std)
    
    train_base = datasets.ImageNet(root=imagenet_root, split="train", transform=train_tf)
    val_base   = datasets.ImageNet(root=imagenet_root, split="val",   transform=test_tf)

    ood_train_ds = FilterByClassIdx(train_base, excluded_idxs)
    ood_val_ds   = FilterByClassIdx(val_base, excluded_idxs)

    return ood_train_ds, ood_val_ds



def get_base(data_root="./data", batch_size=256):
    stats_tf = transforms.Compose([transforms.ToTensor()])
    stats_train_ds = datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=stats_tf
    )

    loader = DataLoader(
        stats_train_ds,
        **dataloader_kwargs
    )

    mean, std = _get_mean_std(loader)
    train_tf, test_tf = _get_transforms(mean, std)

    train_ds = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=train_tf,
    )

    test_ds = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=test_tf,
    )

    return train_ds, test_ds

