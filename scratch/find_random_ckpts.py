import yaml
from pathlib import Path

ckpt_base = Path('assets/data/nerf_checkpoints/fern/fern')
for model_dir in ckpt_base.iterdir():
    if not model_dir.is_dir(): continue
    for ts_dir in model_dir.iterdir():
        if not ts_dir.is_dir(): continue
        config_path = ts_dir / 'config.yml'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    content = f.read()
                    if 'train_num_images_to_sample_from: 10' in content and "split_strategy: random" in content:
                        print(f"Model: {model_dir.name}, TS: {ts_dir.name}, 10 views, Random")
            except Exception as e:
                pass
