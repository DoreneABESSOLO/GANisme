"""
Dataset partagé entre les notebooks de prétraitement et d'entraînement du GAN.
Reprend la logique validée dans preprocessing_images.ipynb (validation,
filtrage des formats extrêmes, center-crop + resize, normalisation [-1, 1]).
"""
import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from scrape_images import target_path


def center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((size, size), Image.BICUBIC)


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def filter_by_format(paths, min_ratio=0.4, max_ratio=2.5, min_side=64):
    
    kept = []
    for p in paths:
        if not is_valid_image(p):
            continue
        with Image.open(p) as im:
            w, h = im.size
        if min_ratio <= (w / h) <= max_ratio and min(w, h) >= min_side:
            kept.append(p)
    return kept


def list_valid_images(images_dir, min_ratio=0.4, max_ratio=2.5, min_side=64):
    """Tous les .jpg valides sous images_dir (scan brut, sans filtrage par sous-échantillon)."""
    return filter_by_format(sorted(Path(images_dir).rglob("*.jpg")), min_ratio, max_ratio, min_side)


def subset_image_paths(csv_path, images_dir, min_ratio=0.4, max_ratio=2.5, min_side=64):
  
    images_dir = Path(images_dir)
    with open(csv_path, encoding="utf-8") as f:
        urls = [row["URL"] for row in csv.DictReader(f)]

    candidates = [target_path(u, images_dir) for u in urls]
    on_disk = [p for p in candidates if p.exists()]
    return filter_by_format(on_disk, min_ratio, max_ratio, min_side)


class PaintingsDataset(Dataset):
    def __init__(self, paths, size=64):
        self.paths = list(paths)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.Lambda(lambda im: center_crop_resize(im, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            with Image.open(self.paths[idx]) as im:
                return self.transform(im)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))
