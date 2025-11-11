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

    @staticmethod
    def _format_file_pattern(template: str, chunk_index: int, **kwargs) -> str:
        """
        Format a file path template for a specific chunk, converting file_index to wildcard.
        Args:
            template: Path template with placeholders (e.g., "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
            chunk_index: The chunk index to substitute
            **kwargs: Additional key-value pairs to substitute in the template
        Returns:
            Formatted pattern with chunk_index filled and file_index as wildcard
        """
        # First format the chunk_index and any other kwargs
        formatted = template.format(chunk_index=chunk_index, file_index=0, **kwargs)
        # Then replace the formatted file_index (000) with wildcard
        # This assumes file_index uses :03d format
        formatted = formatted.replace("file-000", "file-*")
        return formatted

    @staticmethod
    def _format_episode_pattern(template: str, chunk_index: int, episode_index: int, **kwargs) -> str:
        """
        Format a file path template for a specific episode (no wildcards).
        Args:
            template: Path template with placeholders (e.g., "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
            chunk_index: The chunk index to substitute
            episode_index: The episode index to use as file_index
            **kwargs: Additional key-value pairs to substitute in the template
        Returns:
            Formatted pattern with specific file path (e.g., "data/chunk-000/file-002.parquet")
        """
        # Format with specific file_index (episode_index)
        formatted = template.format(chunk_index=chunk_index, file_index=episode_index, **kwargs)
        return formatted

    def generate_chunk_config(self, chunk_index: int, include_log: bool = False) -> dict:
        """
        Generate configuration dictionary for a specific chunk.
        Args:
            chunk_index: The chunk index to generate config for
            include_log: Whether to include log file mapping
        Returns:
            Configuration dictionary compatible with McapConversionConfig
        """
        # Video mappings go in other_mappings
        other_mappings = self._generate_video_mappings(chunk_index)

        # Log mappings go in a separate field (if tabular2mcap supports it)
        # For now, we'll exclude logs to avoid validation errors
        config = {
            "writer_format": "ros2",
            "tabular_mappings": self._generate_tabular_mappings(chunk_index),
            "other_mappings": other_mappings,
            "attachments": [],
            "metadata": [],
        }

        # TODO: Add log mappings once we determine the correct format for tabular2mcap
        # if include_log:
        #     config["log_mappings"] = self._generate_log_mappings()

        logger.debug(f"Generated config for chunk {chunk_index}: "
                    f"{len(config['tabular_mappings'])} tabular, "
                    f"{len(config['other_mappings'])} other mappings")

        return config

    def generate_episode_config(self, episode_index: int, chunk_index: int, include_log: bool = False) -> dict:
        """
        Generate configuration dictionary for a specific episode.
        Args:
            episode_index: The episode index (used as file_index in paths)
            chunk_index: The chunk index
            include_log: Whether to include log file mapping
        Returns:
            Configuration dictionary compatible with McapConversionConfig
        """
        # Video mappings go in other_mappings
        other_mappings = self._generate_video_mappings_for_episode(episode_index, chunk_index)

        config = {
            "writer_format": "ros2",
            "tabular_mappings": self._generate_tabular_mappings_for_episode(episode_index, chunk_index),
            "other_mappings": other_mappings,
            "attachments": [],
            "metadata": [],
        }

        logger.debug(f"Generated config for episode {episode_index} in chunk {chunk_index}: "
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

        # Generate parquet file pattern using helper
        parquet_pattern = self._format_file_pattern(
            self.dataset_info.data_path_template,
            chunk_index
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

    def _generate_tabular_mappings_for_episode(self, episode_index: int, chunk_index: int) -> list[dict]:
        """
        Generate tabular mapping configurations for a specific episode.
        Creates mappings for a specific parquet data file (not wildcards).
        Args:
            episode_index: The episode index (used as file_index)
            chunk_index: The chunk index
        Returns:
            List of tabular mapping configurations
        """
        mappings = []

        # Generate parquet file pattern for specific episode (no wildcards)
        parquet_pattern = self._format_episode_pattern(
            self.dataset_info.data_path_template,
            chunk_index,
            episode_index
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

        for video_key in self.dataset_info.video_keys:
            # Get codec for this video stream (already returns valid format or default)
            video_format = self.dataset_info.get_video_codec(video_key)

            # Generate video file pattern using helper
            video_pattern = self._format_file_pattern(
                self.dataset_info.video_path_template,
                chunk_index,
                video_key=video_key
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

    def _generate_video_mappings_for_episode(self, episode_index: int, chunk_index: int) -> list[dict]:
        """
        Generate video mapping configurations for a specific episode.
        Args:
            episode_index: The episode index (used as file_index)
            chunk_index: The chunk index
        Returns:
            List of CompressedVideoMappingConfig dictionaries
        """
        mappings = []

        for video_key in self.dataset_info.video_keys:
            # Get codec for this video stream
            video_format = self.dataset_info.get_video_codec(video_key)

            # Generate video file pattern for specific episode (no wildcards)
            video_pattern = self._format_episode_pattern(
                self.dataset_info.video_path_template,
                chunk_index,
                episode_index,
                video_key=video_key
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

        summary.append(f"  Other mappings: {len(config['other_mappings'])}")
        for i, mapping in enumerate(config['other_mappings'], 1):
            mapping_type = mapping.get('type', 'unknown')
            topic = mapping.get('topic_suffix', 'N/A')
            if mapping_type == 'compressed_video':
                details = f"{topic} ({mapping['format']}, frame_id={mapping['frame_id']})"
            else:
                details = topic
            summary.append(f"    {i}. [{mapping_type}] {details}")

        return "\n".join(summary)
