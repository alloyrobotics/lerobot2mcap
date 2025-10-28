"""LeRobot to MCAP converter."""

import argparse
from importlib.metadata import version
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tabular2mcap import McapConverter
from tabular2mcap.loader import McapConversionConfig
from pathlib import Path
import yaml
from jinja2 import Template
from tqdm import tqdm
__version__ = version("lerobot2mcap")

# Get the package root directory
PACKAGE_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = str(PACKAGE_ROOT / "configs" / "config.yaml")
DEFAULT_CONVERTER_FUNCTIONS = str(PACKAGE_ROOT / "configs" / "converter_functions.yaml")


def download_dataset(dataset_id: str, output_dir: str, episodes: list[int] | None = None) -> bool:
    """Download a lerobot dataset from Hugging Face Hub."""    
    print(f"📥 Downloading: {dataset_id}")
    if episodes:
        print(f"   Episodes: {episodes}")
    print(f"   Output: {output_dir}")
    
    try:
        dataset = LeRobotDataset(dataset_id, root=output_dir, episodes=episodes)
        print(f"✓ Episodes: {dataset.num_episodes}, Frames: {dataset.num_frames}, FPS: {dataset.fps}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def modified_load_mcap_conversion_config(config_path: Path, episode_id: str) -> McapConversionConfig:
    """Load and validate mapping configuration from YAML file"""
    with open(config_path) as f:
        template = Template(f.read())
        config = yaml.safe_load(template.render(episode_id=episode_id))
    return McapConversionConfig.model_validate(config)

def convert_dataset(input_dir: Path, output_dir: Path,
                    config_path: Path, 
                    converter_functions_path: Path,
                    episodes: list[int] | None = None) -> bool:
    """Convert a lerobot dataset to MCAP format."""
    print(f"🔄 Converting: {input_dir}")
    if episodes:
        print(f"   Episodes: {episodes}")
    print(f"   Output: {output_dir}")
    print(f"   Config: {config_path}")
    print(f"   Converter functions: {converter_functions_path}")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    if episodes is None:
        episode_ids = [p.stem for p in input_dir.glob("**/episode_*.parquet")]
    else:
        episode_ids = [f"episode_{episode_id:06d}" for episode_id in episodes]
    for episode_id in tqdm(episode_ids, desc="Converting episodes"):
        mcap_converter = McapConverter(config_path, converter_functions_path)
        # mcap_config is loaded twice, once in the constructor which is overwritten by the modified_load_mcap_conversion_config function
        mcap_converter.mcap_config = modified_load_mcap_conversion_config(config_path, episode_id)
        mcap_converter.convert(input_dir, output_dir / f"{episode_id}.mcap")

    return True


def main():
    parser = argparse.ArgumentParser(prog="lerobot2mcap", description="Convert LeRobot datasets to MCAP format")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    download_parser = subparsers.add_parser("download", help="Download a LeRobot dataset")
    download_parser.add_argument("dataset_id", help="Dataset ID (e.g., lerobot/pusht)")
    download_parser.add_argument("-o", "--output-dir", default=None, help="Output directory (default: dataset_id)")
    download_parser.add_argument("-e", "--episodes", type=int, nargs="+", help="Episode IDs to download (e.g., 0 1 2). If not specified, all episodes will be downloaded.")
    
    convert_parser = subparsers.add_parser("convert", help="Convert a LeRobot dataset to MCAP format")
    convert_parser.add_argument("input_dir", help="Input directory containing LeRobot dataset")
    convert_parser.add_argument("-o", "--output-dir", default=None, help="Output directory for MCAP files")
    convert_parser.add_argument("-e", "--episodes", type=int, nargs="+", help="Episode IDs to convert (e.g., 0 1 2). If not specified, all episodes will be converted.")
    convert_parser.add_argument("-c", "--config", default=DEFAULT_CONFIG, help=f"Path to config.yaml file (default: {DEFAULT_CONFIG})")
    convert_parser.add_argument("-f", "--converter-functions", default=DEFAULT_CONVERTER_FUNCTIONS, help=f"Path to converter_functions.yaml file (default: {DEFAULT_CONVERTER_FUNCTIONS})")
    
    args = parser.parse_args()
    if args.command == "download":
        if args.output_dir is None:
            args.output_dir = Path("./data") / args.dataset_id
        else:
            args.output_dir = Path(args.output_dir)
        return 0 if download_dataset(args.dataset_id, args.output_dir, args.episodes) else 1
    elif args.command == "convert":
        if args.output_dir is None:
            args.output_dir = Path(args.input_dir) / "mcap"
        else:
            args.output_dir = Path(args.output_dir)
        return 0 if convert_dataset(Path(args.input_dir), args.output_dir, Path(args.config), Path(args.converter_functions), args.episodes) else 1
    parser.print_help()
    return 0
