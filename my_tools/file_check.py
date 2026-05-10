import torch
from pathlib import Path
from tqdm import tqdm

cache_dir = Path("/home/misko/projects/BP/data/SeeingThroughFog/cache")
bad_files = []

for f in tqdm(sorted(cache_dir.glob("*.pt")), desc="Checking cache"):
    try:
        torch.load(f, weights_only=False)
    except Exception as e:
        print(f"Corrupt: {f.name} — {e}")
        bad_files.append(f)

print(f"\nFound {len(bad_files)} corrupt files.")
for f in bad_files:
    f.unlink()
    print(f"Deleted: {f.name}")