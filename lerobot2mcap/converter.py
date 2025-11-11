"""Main converter orchestrator for LeRobot to MCAP conversion."""

import logging
import tempfile
from pathlib import Path

import yaml
from tabular2mcap import McapConverter
from tqdm import tqdm

from .config_generator import ConfigGenerator
from .dataset_info import DatasetInfo

logger = logging.getLogger(__name__)


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
    ) -> bool:
        """
        Convert the LeRobot dataset to MCAP format.

        Args:
            output_dir: Directory where MCAP files will be saved
            chunks: List of chunk indices to convert (None = all chunks)

        Returns:
            True if conversion succeeded, False otherwise
        """
        logger.info("=" * 60)
        logger.info("LeRobot to MCAP Conversion")
        logger.info("=" * 60)
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

        # Check if log file exists (tabular2mcap will handle parsing)
        include_log = self.log_file is not None and self.log_file.exists()
        if include_log and self.log_file:
            logger.info(f"Log file will be included: {self.log_file.name}")

        # Convert each chunk
        success_count = 0
        fail_count = 0

        for chunk_idx in tqdm(chunks, desc="Converting chunks", unit="chunk"):
            try:
                self._convert_chunk(chunk_idx, output_dir, include_log)
                success_count += 1
            except Exception as e:
                logger.warning(f"Skipping chunk {chunk_idx}: {e}")
                fail_count += 1

        # Summary
        logger.info("=" * 60)
        logger.info("Conversion Summary")
        logger.info("=" * 60)
        logger.info(f"Successfully converted: {success_count}/{len(chunks)} chunks")
        if fail_count > 0:
            logger.warning(f"Failed/Skipped: {fail_count}/{len(chunks)} chunks")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 60)

        return success_count > 0

    def _convert_chunk(
        self, chunk_idx: int, output_dir: Path, include_log: bool
    ):
        """
        Convert a single chunk to MCAP.

        Args:
            chunk_idx: The chunk index to convert
            output_dir: Output directory for MCAP files
            include_log: Whether to include log file in conversion

        Raises:
            Exception: If chunk conversion fails
        """
        # Check if chunk files exist
        chunk_files = self.dataset_info.get_chunk_files(chunk_idx, self.dataset_root)

        if not chunk_files["parquet"].exists():
            raise FileNotFoundError(
                f"Parquet file not found: {chunk_files['parquet']}"
            )

        # Check video files
        missing_videos = []
        for video_key, video_path in chunk_files["videos"].items():
            if not video_path.exists():
                missing_videos.append(video_key)

        if missing_videos:
            logger.warning(
                f"Chunk {chunk_idx}: Missing videos: {missing_videos}. "
                "These will be skipped."
            )

        # Generate dynamic configuration (tabular2mcap will find and parse .log files)
        chunk_config = self.config_generator.generate_chunk_config(
            chunk_idx, include_log=include_log
        )

        # Create temporary config file for tabular2mcap
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as config_file:
            yaml.dump(chunk_config, config_file)
            config_path = Path(config_file.name)

        try:
            # Use tabular2mcap's McapConverter
            mcap_converter = McapConverter(config_path, self.converter_functions_path)

            # Output MCAP path
            output_mcap = output_dir / f"chunk-{chunk_idx:03d}.mcap"

            logger.info(f"Converting chunk {chunk_idx} -> {output_mcap.name}")

            # Convert
            mcap_converter.convert(self.dataset_root, output_mcap)

        finally:
            # Clean up temporary config file
            config_path.unlink(missing_ok=True)

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
            "=" * 60,
            "LeRobot to MCAP Conversion Plan",
            "=" * 60,
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

        plan.append("=" * 60)

        return "\n".join(plan)
