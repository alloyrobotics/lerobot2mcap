"""Main converter orchestrator for LeRobot to MCAP conversion."""

import logging
from pathlib import Path

import yaml
from tabular2mcap import McapConverter
from tqdm import tqdm

from .config_generator import ConfigGenerator
from .dataset_info import DatasetInfo

logger = logging.getLogger(__name__)

# Display formatting constants
SEPARATOR_WIDTH = 60


class LeRobotConverter:
    """Main converter orchestrator that manages the conversion process."""

    def __init__(self, dataset_root: Path, converter_functions_path: Path):
        """
        Initialize LeRobotConverter.
        Args:
            dataset_root: Root directory of the LeRobot dataset
            converter_functions_path: Path to converter_functions.yaml
        """
        self.dataset_root = dataset_root
        self.converter_functions_path = converter_functions_path

        # Validate dataset structure
        info_json_path = dataset_root / "meta" / "info.json"
        if not info_json_path.exists():
            raise FileNotFoundError(
                f"Dataset info.json not found at {info_json_path}. "
                f"Is {dataset_root} a valid LeRobot dataset?"
            )

        # Composition: LeRobotConverter HAS-A DatasetInfo
        self.dataset_info = DatasetInfo(info_json_path)

        # Composition: LeRobotConverter HAS-A ConfigGenerator
        self.config_generator = ConfigGenerator(self.dataset_info)

        # Find log file
        self.log_file = self._find_log_file()
        if self.log_file:
            logger.info(f"Found log file: {self.log_file}")
        else:
            logger.warning("No log file found in dataset root")

        logger.info(f"Initialized converter for {self.dataset_info}")

    def _find_log_file(self) -> Path | None:
        """
        Find the .log file in the dataset root directory.
        Returns:
            Path to the log file, or None if not found
        """
        log_files = list(self.dataset_root.glob("*.log"))

        if not log_files:
            return None

        if len(log_files) > 1:
            logger.warning(
                f"Multiple log files found: {[f.name for f in log_files]}. "
                f"Using the first one: {log_files[0].name}"
            )

        return log_files[0]

    def convert(
        self,
        output_dir: Path,
        chunks: list[int] | None = None,
        episodes: list[int] | None = None,
    ) -> bool:
        """
        Convert the LeRobot dataset to MCAP format.
        Iterates through each chunk and converts all episodes within that chunk.
        Each episode produces a separate MCAP file in its own directory.

        Args:
            output_dir: Directory where MCAP files will be saved
            chunks: List of chunk indices to convert (None = all chunks)
            episodes: List of episode indices to convert (None = all episodes)
        Returns:
            True if conversion succeeded, False otherwise
        """
        logger.info("=" * SEPARATOR_WIDTH)
        logger.info("LeRobot to MCAP Conversion")
        logger.info("=" * SEPARATOR_WIDTH)
        logger.info(f"Dataset: {self.dataset_root}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Dataset info: {self.dataset_info}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine which chunks to convert
        if chunks is None:
            total_chunks = self.dataset_info.get_total_chunks()
            chunks = list(range(total_chunks))
            logger.info(f"Converting all {total_chunks} chunks")
        else:
            logger.info(f"Converting chunks: {chunks}")

        # Determine which episodes to convert
        if episodes is None:
            total_episodes = self.dataset_info.get_total_episodes()
            episodes = list(range(total_episodes))
            logger.info(f"Converting all {total_episodes} episodes")
        else:
            logger.info(f"Converting episodes: {episodes}")

        # Check if log file exists (tabular2mcap will handle parsing)
        include_log = False
        if self.log_file is not None and self.log_file.exists():
            include_log = True
            logger.info(f"Log file will be included: {self.log_file.name}")

        # Convert each episode within each chunk
        success_count = 0
        fail_count = 0

        for chunk_idx in chunks:
            for episode_idx in tqdm(episodes, desc=f"Converting episodes in chunk-{chunk_idx:03d}", unit="episode"):
                try:
                    self._convert_episode(episode_idx, chunk_idx, output_dir, include_log)
                    success_count += 1
                except Exception as e:
                    logger.warning(f"Skipping episode {episode_idx} in chunk {chunk_idx}: {e}")
                    fail_count += 1

        # Summary
        logger.info("=" * SEPARATOR_WIDTH)
        logger.info("Conversion Summary")
        logger.info("=" * SEPARATOR_WIDTH)
        logger.info(f"Successfully converted: {success_count} episodes")
        if fail_count > 0:
            logger.warning(f"Failed/Skipped: {fail_count} episodes")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * SEPARATOR_WIDTH)

        return success_count > 0

    def _convert_episode(
        self, episode_idx: int, chunk_idx: int, output_dir: Path, include_log: bool
    ):
        """
        Convert a single episode to MCAP.
        Args:
            episode_idx: The episode index (used as file_index)
            chunk_idx: The chunk index
            output_dir: Output directory for MCAP files
            include_log: Whether to include log file in conversion
        Raises:
            Exception: If episode conversion fails
        """
        # Check if episode files exist
        episode_files = self.dataset_info.get_episode_files(episode_idx, chunk_idx, self.dataset_root)

        if not episode_files["parquet"].exists():
            raise FileNotFoundError(
                f"Parquet file not found: {episode_files['parquet']}"
            )

        # Check video files
        missing_videos = []
        for video_key, video_path in episode_files["videos"].items():
            if not video_path.exists():
                missing_videos.append(video_key)

        if missing_videos:
            logger.warning(
                f"Episode {episode_idx}: Missing videos: {missing_videos}. "
                "These will be skipped."
            )

        # Generate dynamic configuration for this episode
        episode_config = self.config_generator.generate_episode_config(
            episode_idx, chunk_idx, include_log=include_log
        )

        # Create episode directory: mcap_output/episode_000/
        episode_dir = output_dir / f"episode_{episode_idx:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Save config file: episode_000/config_000.yaml
        config_path = episode_dir / f"config_{episode_idx:03d}.yaml"
        with open(config_path, "w") as config_file:
            yaml.dump(episode_config, config_file, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved config: {config_path.relative_to(output_dir)}")

        # Use tabular2mcap's McapConverter
        mcap_converter = McapConverter(config_path, self.converter_functions_path)

        # Output MCAP path: episode_000/episode_000.mcap
        output_mcap = episode_dir / f"episode_{episode_idx:03d}.mcap"

        logger.info(f"Converting episode {episode_idx} -> {output_mcap.relative_to(output_dir)}")

        # Convert
        mcap_converter.convert(self.dataset_root, output_mcap)

    def get_conversion_plan(self, chunks: list[int] | None = None) -> str:
        """
        Generate a human-readable conversion plan.
        Args:
            chunks: List of chunk indices to include in plan (None = all)
        Returns:
            Formatted string describing the conversion plan
        """
        if chunks is None:
            chunks = list(range(self.dataset_info.get_total_chunks()))

        plan = [
            "=" * SEPARATOR_WIDTH,
            "LeRobot to MCAP Conversion Plan",
            "=" * SEPARATOR_WIDTH,
            f"Dataset: {self.dataset_root.name}",
            f"Total episodes: {self.dataset_info.get_total_episodes()}",
            f"Total chunks: {self.dataset_info.get_total_chunks()}",
            f"FPS: {self.dataset_info.get_fps()}",
            f"Video streams: {len(self.dataset_info.video_keys)}",
        ]

        for video_key in self.dataset_info.video_keys:
            codec = self.dataset_info.get_video_codec(video_key)
            plan.append(f"  - {video_key} ({codec})")

        plan.extend([
            f"Log file: {'Yes' if self.log_file else 'No'}",
            "",
            f"Chunks to convert: {len(chunks)}",
            "",
        ])

        # Show config for first chunk as example
        if chunks:
            plan.append(self.config_generator.generate_config_summary(chunks[0]))

        plan.append("=" * SEPARATOR_WIDTH)

        return "\n".join(plan)
