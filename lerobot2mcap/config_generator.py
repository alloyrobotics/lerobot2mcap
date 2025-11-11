"""Configuration generator for tabular2mcap conversion."""

import logging
from pathlib import Path

from .dataset_info import DatasetInfo

logger = logging.getLogger(__name__)


class ConfigGenerator:
    """Generates tabular2mcap configuration dynamically based on dataset info."""

    def __init__(self, dataset_info: DatasetInfo):
        """
        Initialize ConfigGenerator with dataset information.

        Args:
            dataset_info: DatasetInfo instance containing dataset metadata
        """
        self.dataset_info = dataset_info
        logger.info(f"Initialized ConfigGenerator for dataset with {len(dataset_info.video_keys)} video streams")

    def generate_chunk_config(self, chunk_index: int, include_log: bool = False) -> dict:
        """
        Generate configuration dictionary for a specific chunk.

        Args:
            chunk_index: The chunk index to generate config for
            include_log: Whether to include log file mapping

        Returns:
            Configuration dictionary compatible with McapConversionConfig
        """
        # Combine video and log mappings in other_mappings
        other_mappings = self._generate_video_mappings(chunk_index)
        if include_log:
            other_mappings.extend(self._generate_log_mappings())

        config = {
            "writer_format": "ros2",
            "tabular_mappings": self._generate_tabular_mappings(chunk_index),
            "other_mappings": other_mappings,
            "attachments": [],
            "metadata": [],
        }

        logger.debug(f"Generated config for chunk {chunk_index}: "
                    f"{len(config['tabular_mappings'])} tabular, "
                    f"{len(config['other_mappings'])} other mappings")

        return config

    def _generate_tabular_mappings(self, chunk_index: int) -> list[dict]:
        """
        Generate tabular mapping configurations.

        Creates mappings for parquet data files.

        Args:
            chunk_index: The chunk index

        Returns:
            List of tabular mapping configurations
        """
        mappings = []

        # Parquet file mapping
        data_path_template = self.dataset_info.data.get(
            "data_path",
            "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        )
        # Format chunk_index but keep file_index as wildcard
        # Replace {chunk_index:03d} with actual value, and {file_index:03d} with *
        parquet_pattern = data_path_template.replace(
            "{chunk_index:03d}", f"{chunk_index:03d}"
        ).replace(
            "{file_index:03d}", "*"
        )

        mappings.append({
            "file_pattern": f"**/{parquet_pattern}",
            "converter_functions": [
                {
                    "function_name": "row_to_message_with_timestamp",
                    "schema_name": None,  # Will auto-generate schema
                    "topic_suffix": "robot_data",
                    "exclude_columns": ["timestamp"],
                }
            ],
        })

        return mappings

    def _generate_video_mappings(self, chunk_index: int) -> list[dict]:
        """
        Generate video mapping configurations for all video streams.

        Args:
            chunk_index: The chunk index

        Returns:
            List of CompressedVideoMappingConfig dictionaries
        """
        mappings = []
        video_path_template = self.dataset_info.data.get(
            "video_path",
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        )

        for video_key in self.dataset_info.video_keys:
            # Get codec for this video stream
            codec = self.dataset_info.get_video_codec(video_key)

            # Map LeRobot codec names to tabular2mcap format names
            codec_map = {
                "av1": "av1",
                "h264": "h264",
                "h265": "h265",
                "vp9": "vp9",
            }
            video_format = codec_map.get(codec, "h264")

            # Generate file pattern for this video stream
            # Replace placeholders: {video_key}, {chunk_index:03d}, {file_index:03d}
            video_pattern = (
                video_path_template
                .replace("{video_key}", video_key)
                .replace("{chunk_index:03d}", f"{chunk_index:03d}")
                .replace("{file_index:03d}", "*")
            )

            # Generate topic suffix from video_key
            # "observation.images.front" -> "observation/images/front"
            topic_suffix = video_key.replace(".", "/")

            # Get frame_id
            frame_id = self.dataset_info.get_video_frame_id(video_key)

            mappings.append({
                "type": "compressed_video",
                "file_pattern": f"**/{video_pattern}",
                "topic_suffix": topic_suffix,
                "frame_id": frame_id,
                "format": video_format,
            })

        return mappings

    def _generate_log_mappings(self) -> list[dict]:
        """
        Generate log mapping configurations for terminal output logs.

        Uses tabular2mcap's built-in LogConverter to parse raw .log files
        directly into rcl_interfaces/msg/Log messages.

        Returns:
            List of LogMappingConfig dictionaries
        """
        return [{
            "type": "log",
            "file_pattern": "**/*.log",
            "topic_suffix": "terminal_log",
            "format_template": r"^(?P<levelname>INFO|DEBUG|WARNING|ERROR|FATAL)\s(?P<asctime>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\s(?P<filename>\S+):(?P<lineno>\d+)\s(?P<message>.*)$",
            "datetime_format": "%Y-%m-%d %H:%M:%S",
        }]

    def generate_config_summary(self, chunk_index: int) -> str:
        """
        Generate a human-readable summary of the configuration.

        Args:
            chunk_index: The chunk index

        Returns:
            Formatted string describing the configuration
        """
        config = self.generate_chunk_config(chunk_index, include_log=True)

        summary = [
            f"Configuration for chunk-{chunk_index:03d}:",
            f"  Writer format: {config['writer_format']}",
            f"  Tabular mappings: {len(config['tabular_mappings'])}",
        ]

        for i, mapping in enumerate(config['tabular_mappings'], 1):
            summary.append(f"    {i}. {mapping['file_pattern']}")

        summary.append(f"  Video mappings: {len(config['other_mappings'])}")
        for i, mapping in enumerate(config['other_mappings'], 1):
            summary.append(
                f"    {i}. {mapping['topic_suffix']} "
                f"({mapping['format']}, frame_id={mapping['frame_id']})"
            )

        return "\n".join(summary)
