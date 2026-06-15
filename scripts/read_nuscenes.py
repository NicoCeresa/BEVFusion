import yaml
from pathlib import Path
from nuscenes.nuscenes import NuScenes

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

nusc = NuScenes(
    version=cfg["data"]["version"],
    dataroot=cfg["data"]["root"],
    verbose=True,
)

def visualize_sample(sample_token):
    nusc.render_sample(sample_token)

if __name__ == "__main__":
    sample_token = nusc.sample[0]["token"]
    visualize_sample(sample_token)