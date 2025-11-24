"""Configuration generator for tabular2mcap conversion."""

import logging
# from pathlib import Path TO DO: check why currently this is unused

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
        logger.info(
            f"Initialized ConfigGenerator for dataset with {len(dataset_info.video_keys)} video streams"
        )

    @staticmethod
    def _format_file_pattern(template: str, chunk_index: int, **kwargs) -> str:
        """
        Format a file path template for a specific chunk, converting file_index to wildcard.
        Supports both v2.1 and v3 LeRobot formats.
        Args:
            template: Path template with placeholders
            chunk_index: The chunk index to substitute
            **kwargs: Additional key-value pairs to substitute in the template
        Returns:
            Formatted pattern with chunk_index filled and file_index as wildcard
        """
        # Support both v2.1 and v3 naming conventions
        format_params = {
            "episode_chunk": chunk_index,  # v2.1 format
            "episode_index": 0,  # v2.1 format
            "chunk_index": chunk_index,  # v3 format
            "file_index": 0,  # v3 format
            **kwargs,
        }

        # Format the template with all parameters
        formatted = template.format(**format_params)

        # Replace file index with wildcard (supports both formats)
        formatted = formatted.replace(
            "episode_000000", "episode_*"
        )  # v2.1: episode_{episode_index:06d}
        formatted = formatted.replace("file-000", "file-*")  # v3: file-{file_index:03d}

        return formatted

    @staticmethod
    def _format_episode_pattern(
        template: str, chunk_index: int, episode_index: int, **kwargs
    ) -> str:
        """
        Format a file path template for a specific episode (no wildcards).
        Supports both v2.1 and v3 LeRobot formats.
        Args:
            template: Path template with placeholders
            chunk_index: The chunk index to substitute
            episode_index: The episode index to use as file_index
            **kwargs: Additional key-value pairs to substitute in the template
        Returns:
            Formatted pattern with specific file path
        """
        # Support both v2.1 and v3 naming conventions
        format_params = {
            "episode_chunk": chunk_index,  # v2.1 format
            "episode_index": episode_index,  # v2.1 format
            "chunk_index": chunk_index,  # v3 format
            "file_index": episode_index,  # v3 format
            **kwargs,
        }

        # Format with specific file_index (episode_index)
        formatted = template.format(**format_params)
        return formatted

    def generate_chunk_config(self, chunk_index: int) -> dict:
        """
        Generate configuration dictionary for a whole (potentially multi-episode) specific chunk.

        Args:
            chunk_index: The chunk index to generate config for

        Returns:
            Configuration dictionary compatible with McapConversionConfig
        """
        # Video mappings go in other_mappings
        other_mappings = self._generate_video_mappings(chunk_index)

        config = {
            "writer_format": "ros2",
            "tabular_mappings": self._generate_tabular_mappings(chunk_index),
            "other_mappings": other_mappings,
            "attachments": [],
            "metadata": [],
        }

        logger.debug(
            f"Generated config for chunk {chunk_index}: "
            f"{len(config['tabular_mappings'])} tabular, "
            f"{len(config['other_mappings'])} other mappings"
        )

        return config

    def generate_episode_config(
        self, episode_index: int, chunk_index: int, include_log: bool = False
    ) -> dict:
        """
        Generate configuration dictionary for a specific episode.

        Args:
            episode_index: The episode index (used as file_index in paths)
            chunk_index: The chunk index
            include_log: Whether to include log file as MCAP attachment

        Returns:
            Configuration dictionary compatible with McapConversionConfig
        """
        # Video mappings go in other_mappings
        other_mappings = self._generate_video_mappings_for_episode(
            episode_index, chunk_index
        )

        # Generate attachments (log file if requested)
        attachments = []
        if include_log:
            attachments.append(
                {
                    "file_pattern": "**/*.log",
                    "name": "dataset.log",
                    "media_type": "text/plain",
                }
            )

        config = {
            "writer_format": "ros2",
            "tabular_mappings": self._generate_tabular_mappings_for_episode(
                episode_index, chunk_index
            ),
            "other_mappings": other_mappings,
            "attachments": attachments,
            "metadata": [],
        }

        logger.debug(
            f"Generated config for episode {episode_index} in chunk {chunk_index}: "
            f"{len(config['tabular_mappings'])} tabular, "
            f"{len(config['other_mappings'])} other mappings, "
            f"{len(config['attachments'])} attachments"
        )

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
            self.dataset_info.data_path_template, chunk_index
        )

        mappings.append(
            {
                "file_pattern": f"**/{parquet_pattern}",
                "converter_functions": [
                    {
                        "function_name": "row_to_message_with_timestamp",
                        "schema_name": None,  # Will auto-generate schema
                        "topic_suffix": "robot_data",
                        "exclude_columns": ["timestamp"],
                    }
                ],
            }
        )

        return mappings

    def _generate_tabular_mappings_for_episode(
        self, episode_index: int, chunk_index: int
    ) -> list[dict]:
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
            self.dataset_info.data_path_template, chunk_index, episode_index
        )

        mappings.append(
            {
                "file_pattern": f"**/{parquet_pattern}",
                "converter_functions": [
                    {
                        "function_name": "row_to_message_with_timestamp",
                        "schema_name": None,  # Will auto-generate schema
                        "topic_suffix": "robot_data",
                        "exclude_columns": ["timestamp"],
                    }
                ],
            }
        )

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
                self.dataset_info.video_path_template, chunk_index, video_key=video_key
            )

            # Generate topic suffix from video_key
            # "observation.images.front" -> "observation/images/front"
            topic_suffix = video_key.replace(".", "/")

            # Get frame_id
            frame_id = self.dataset_info.get_video_frame_id(video_key)

            mappings.append(
                {
                    "type": "compressed_video",
                    "file_pattern": f"**/{video_pattern}",
                    "topic_suffix": topic_suffix,
                    "frame_id": frame_id,
                    "format": video_format,
                }
            )

        return mappings

    def _generate_video_mappings_for_episode(
        self, episode_index: int, chunk_index: int
    ) -> list[dict]:
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
                video_key=video_key,
            )

            # Generate topic suffix from video_key
            # "observation.images.front" -> "observation/images/front"
            topic_suffix = video_key.replace(".", "/")

            # Get frame_id
            frame_id = self.dataset_info.get_video_frame_id(video_key)

            mappings.append(
                {
                    "type": "compressed_video",
                    "file_pattern": f"**/{video_pattern}",
                    "topic_suffix": topic_suffix,
                    "frame_id": frame_id,
                    "format": video_format,
                }
            )

        return mappings

    def generate_config_summary(self, chunk_index: int) -> str:
        """
        Generate a human-readable summary of the configuration.

        Args:
            chunk_index: The chunk index

        Returns:
            Formatted string describing the configuration
        """
        config = self.generate_chunk_config(chunk_index)

        summary = [
            f"Configuration for chunk-{chunk_index:03d}:",
            f"  Writer format: {config['writer_format']}",
            f"  Tabular mappings: {len(config['tabular_mappings'])}",
        ]

        for i, mapping in enumerate(config["tabular_mappings"], 1):
            summary.append(f"    {i}. {mapping['file_pattern']}")

        summary.append(f"  Other mappings: {len(config['other_mappings'])}")
        for i, mapping in enumerate(config["other_mappings"], 1):
            mapping_type = mapping.get("type", "unknown")
            topic = mapping.get("topic_suffix", "N/A")
            if mapping_type == "compressed_video":
                details = (
                    f"{topic} ({mapping['format']}, frame_id={mapping['frame_id']})"
                )
            else:
                details = topic
            summary.append(f"    {i}. [{mapping_type}] {details}")

        return "\n".join(summary)
