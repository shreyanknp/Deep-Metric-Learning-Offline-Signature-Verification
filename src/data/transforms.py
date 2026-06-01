from typing import Callable
from PIL import Image
import torchvision.transforms as T

def get_default_transforms(size: int = 224) -> Callable:
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
