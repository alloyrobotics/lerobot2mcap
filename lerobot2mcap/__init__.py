"""LeRobot to MCAP converter."""

import argparse
import logging
from importlib.metadata import version
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .converter import LeRobotConverter

__version__ = version("lerobot2mcap")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Get the package root directory
PACKAGE_ROOT = Path(__file__).parent.parent
DEFAULT_CONVERTER_FUNCTIONS = str(PACKAGE_ROOT / "configs" / "converter_functions.yaml")


def download_dataset(dataset_id: str, output_dir: Path, episodes: list[int] | None = None) -> bool:
    """Download a lerobot dataset from Hugging Face Hub."""
    print(f"📥 Downloading: {dataset_id}")
    if episodes:
        print(f"   Episodes: {episodes}")
    print(f"   Output: {output_dir}")

    try:
        dataset = LeRobotDataset(dataset_id, root=str(output_dir), episodes=episodes)
        print(f"✓ Episodes: {dataset.num_episodes}, Frames: {dataset.num_frames}, FPS: {dataset.fps}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def convert_dataset(
    dataset_root: Path,
    output_dir: Path,
    converter_functions_path: Path,
    chunks: list[int] | None = None,
    episodes: list[int] | None = None,
) -> bool:
    """
    Convert a LeRobot dataset to MCAP format (episode-based).
    Args:
        dataset_root: Root directory of the LeRobot dataset
        output_dir: Output directory for MCAP files
        converter_functions_path: Path to converter_functions.yaml
        chunks: List of chunk indices to convert (None = all chunks)
        episodes: List of episode indices to convert (None = all episodes)
    Returns:
        True if conversion succeeded, False otherwise
    """
    print(f"🔄 Converting: {dataset_root}")
    if chunks:
        print(f"   Chunks: {chunks}")
    if episodes:
        print(f"   Episodes: {episodes}")
    print(f"   Output: {output_dir}")
    print(f"   Converter functions: {converter_functions_path}")

    try:
        # Initialize the converter with new OOP architecture
        converter = LeRobotConverter(
            dataset_root=dataset_root,
            converter_functions_path=converter_functions_path
        )

        # Show conversion plan
        print("\n" + converter.get_conversion_plan(chunks))

        # Perform conversion
        success = converter.convert(
            output_dir=output_dir,
            chunks=chunks,
            episodes=episodes
        )

        return success

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(prog="lerobot2mcap", description="Convert LeRobot datasets to MCAP format")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    download_parser = subparsers.add_parser("download", help="Download a LeRobot dataset")
    download_parser.add_argument("dataset_id", help="Dataset ID (e.g., lerobot/pusht)")
    download_parser.add_argument("-o", "--output-dir", default=None, help="Output directory (default: dataset_id)")
    download_parser.add_argument("-e", "--episodes", type=int, nargs="+", help="Episode IDs to download (e.g., 0 1 2). If not specified, all episodes will be downloaded.")
    
    convert_parser = subparsers.add_parser("convert", help="Convert a LeRobot dataset to MCAP format")
    convert_parser.add_argument("input_dir", help="Input directory containing LeRobot dataset (dataset root with meta/info.json)")
    convert_parser.add_argument("-o", "--output-dir", default=None, help="Output directory for MCAP files (default: input_dir/mcap)")
    convert_parser.add_argument("-c", "--chunks", type=int, nargs="+", help="Chunk IDs to convert (e.g., 0 1 2). If not specified, all chunks will be converted.")
    convert_parser.add_argument("-e", "--episodes", type=int, nargs="+", help="Episode IDs to convert (e.g., 0 1 2). If not specified, all episodes will be converted.")
    convert_parser.add_argument("-f", "--converter-functions", default=DEFAULT_CONVERTER_FUNCTIONS, help=f"Path to converter_functions.yaml file (default: {DEFAULT_CONVERTER_FUNCTIONS})")

    args = parser.parse_args()

    if args.command == "download":
        output_dir = Path(args.output_dir) if args.output_dir else Path("./data") / args.dataset_id
        return 0 if download_dataset(args.dataset_id, output_dir, args.episodes) else 1

    if args.command == "convert":
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.input_dir) / "mcap"
        return 0 if convert_dataset(
            Path(args.input_dir),
            output_dir,
            Path(args.converter_functions),
            args.chunks,
            args.episodes
        ) else 1

    parser.print_help()
    return 0
