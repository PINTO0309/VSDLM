from pathlib import Path

from PIL import Image
from torchvision import transforms

from vsdlm.pipeline import RandomCLAHE


DEBUG_DIR = Path("debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

img_path = Path("002_0005_front_028001_000000_pitchm005_yawm005_mouth.png")  # 48×28px の例
img = Image.open(img_path)
gray = transforms.RandomGrayscale(p=1.0)
gray_only_img = gray(img)

for size in [(4, 4), (6, 6), (8, 8)]:
    clahe = RandomCLAHE(tile_grid_size=size, p=1.0)
    clahe_img = clahe(img)
    clahe_path = DEBUG_DIR / f"{img_path.stem}_clahe_{size[0]}x{size[1]}.png"
    clahe_img.save(clahe_path)
    gray_only_img.save(DEBUG_DIR / f"{img_path.stem}_gray_{size[0]}x{size[1]}.png")
    gray(clahe_img).save(DEBUG_DIR / f"{clahe_path.stem}_gray{clahe_path.suffix}")
