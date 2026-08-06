import torch
import torchvision
import torchvision.transforms as T


def _tv_version() -> tuple:
    return tuple(int(x) for x in torchvision.__version__.split('.')[:2])


def get_train_transforms(img_size: int = 224) -> T.Compose:
    """Augmentation pipeline for Phase 1 and Phase 2 training.

    Applies geometric distortions that mimic real signing variation:
    affine (rotation, translation, scale), perspective warp, elastic
    deformation (torchvision ≥ 0.14), and mild Gaussian noise.
    """
    steps = [
        T.Resize((img_size, img_size)),
        T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.93, 1.07), fill=255),
        T.RandomPerspective(distortion_scale=0.2, p=0.4, fill=255),
    ]
    if _tv_version() >= (0, 14):
        steps.append(T.ElasticTransform(alpha=30.0, sigma=4.0, fill=255))
    steps += [
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
        T.Lambda(lambda x: x + 0.015 * torch.randn_like(x)),
    ]
    return T.Compose(steps)


def get_eval_transforms(img_size: int = 224) -> T.Compose:
    """Deterministic transform for single-image evaluation (no augmentation)."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_tta_transforms(img_size: int = 224) -> T.Compose:
    """Light stochastic transform for Test-Time Augmentation (TTA).

    Milder than training augmentation — enough to diversify views but
    not so strong that it corrupts the signature structure.
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.96, 1.04), fill=255),
        T.RandomPerspective(distortion_scale=0.1, p=0.5, fill=255),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
