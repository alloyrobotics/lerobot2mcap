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
    format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Get the package root directory
PACKAGE_ROOT = Path(__file__).parent.parent
DEFAULT_CONVERTER_FUNCTIONS = str(PACKAGE_ROOT / "configs" / "converter_functions.yaml")


def download_dataset(
    dataset_id: str, output_dir: Path, episodes: list[int] | None = None
) -> bool:
    """Download a lerobot dataset from Hugging Face Hub."""
    print(f" Downloading: {dataset_id}")
    if episodes:
        print(f"   Episodes: {episodes}")
    print(f"   Output: {output_dir}")

    try:
        dataset = LeRobotDataset(dataset_id, root=str(output_dir), episodes=episodes)
        print(
            f"✓ Episodes: {dataset.num_episodes}, Frames: {dataset.num_frames}, FPS: {dataset.fps}"
        )
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
    Convert a LeRobot dataset to MCAP format.
    Iterates through each chunk and converts all episodes within that chunk.
    Each episode produces a separate MCAP file in its own directory.

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
        # Initialize the converter
        converter = LeRobotConverter(
            dataset_root=dataset_root, converter_functions_path=converter_functions_path
        )

        # Show conversion plan
        print("\n" + converter.get_conversion_plan(chunks))

        # Perform conversion
        success = converter.convert(
            output_dir=output_dir, chunks=chunks, episodes=episodes
        )

        return success

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        prog="lerobot2mcap", description="Convert LeRobot datasets to MCAP format"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Define download parser arguments
    download_parser = subparsers.add_parser(
        "download", help="Download a LeRobot dataset"
    )
    download_parser.add_argument("dataset_id", help="Dataset ID (e.g., lerobot/pusht)")
    download_parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory (default: dataset_id)",
    )
    download_parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        nargs="+",
        help="Episode IDs to download (e.g., 0 1 2). If not specified, all episodes will be downloaded.",
    )

    # Define
    convert_parser = subparsers.add_parser(
        "convert", help="Convert a LeRobot dataset to MCAP format"
    )
    convert_parser.add_argument(
        "input_dir",
        help="Input directory containing LeRobot dataset (dataset root with meta/info.json)",
    )
    convert_parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory for MCAP files (default: input_dir/mcap)",
    )
    convert_parser.add_argument(
        "-c",
        "--chunks",
        type=int,
        nargs="+",
        help="Chunk IDs to convert (e.g., 0 1 2). If not specified, all chunks will be converted.",
    )
    convert_parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        nargs="+",
        help="Episode IDs to convert (e.g., 0 1 2). If not specified, all episodes will be converted.",
    )
    convert_parser.add_argument(
        "-f",
        "--converter-functions",
        default=DEFAULT_CONVERTER_FUNCTIONS,
        help=f"Path to converter_functions.yaml file (default: {DEFAULT_CONVERTER_FUNCTIONS})",
    )

    args = parser.parse_args()

    # Handle download command
    if args.command == "download":
        download_dir = Path(args.output_dir) if args.output_dir else Path("./data")

        # Download the dataset
        if not download_dataset(args.dataset_id, download_dir, args.episodes):
            return 1  # Download failed

        # Set dataset root for conversion
        dataset_root = download_dir
        mcap_output_dir = dataset_root / "mcap_conversion"
        converter_functions = Path(DEFAULT_CONVERTER_FUNCTIONS)
        chunks = None  # Convert all chunks
        episodes = args.episodes  # Use same episode filter as download

    elif args.command == "convert":
        # Set parameters from convert command arguments
        dataset_root = Path(args.input_dir)
        mcap_output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else dataset_root / "mcap_conversion"
        )
        converter_functions = Path(args.converter_functions)
        chunks = args.chunks
        episodes = args.episodes

    else:
        # No command provided
        parser.print_help()
        return 0

    # Perform conversion (always happens after download, or standalone)
    if convert_dataset(
        dataset_root, mcap_output_dir, converter_functions, chunks, episodes
    ):
        return 0
    else:
        return 1
