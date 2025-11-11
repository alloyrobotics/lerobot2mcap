"""Dataset information parser for LeRobot datasets.
Parses the info.json file to extract metadata about the dataset,"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetInfo:
    """Parses and holds LeRobot dataset metadata from info.json."""

    def __init__(self, info_json_path: Path):
        """
        Initialize DatasetInfo by parsing info.json.

        Args:
            info_json_path: Path to the info.json file (usually in meta/info.json)
        """
        self.info_json_path = info_json_path
        self.data = self._parse_info_json()
        self.video_keys = self._extract_video_keys()
        logger.info(f"Loaded dataset info from {info_json_path}")
        logger.info(f"Found {len(self.video_keys)} video streams: {self.video_keys}")

    def _parse_info_json(self) -> dict:
        """Parse the info.json file."""
        if not self.info_json_path.exists():
            raise FileNotFoundError(f"info.json not found at {self.info_json_path}")

        with open(self.info_json_path) as f:
            return json.load(f)

    def _extract_video_keys(self) -> list[str]:
        """
        Extract video keys from features.

        Video features have dtype="video" in info.json.
        Example: "observation.images.front", "observation.images.external"
        """
        video_keys = []
        features = self.data.get("features", {})

        for feature_name, feature_info in features.items():
            if isinstance(feature_info, dict) and feature_info.get("dtype") == "video":
                video_keys.append(feature_name)

        return video_keys

    def get_chunk_files(self, chunk_index: int, dataset_root: Path) -> dict:
        """
        Get file paths for a specific chunk.

        Args:
            chunk_index: The chunk index (e.g., 0 for chunk-000)
            dataset_root: Root directory of the dataset

        Returns:
            Dictionary with:
            {
                "parquet": Path to parquet file,
                "videos": {
                    "observation.images.front": Path to video file,
                    "observation.images.external": Path to video file,
                    ...
                }
            }
        """
        chunk_files = {}

        # Get parquet file path
        data_path_template = self.data.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
        # For now, assume file_index is always 000 within a chunk
        parquet_path = data_path_template.format(chunk_index=chunk_index, file_index=0)
        chunk_files["parquet"] = dataset_root / parquet_path

        # Get video file paths
        video_path_template = self.data.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
        chunk_files["videos"] = {}

        for video_key in self.video_keys:
            video_path = video_path_template.format(
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=0
            )
            chunk_files["videos"][video_key] = dataset_root / video_path

        return chunk_files


# TO DO - Not sure if this is doing the correct thing yet 
    def get_total_chunks(self) -> int:
        """
        Calculate total number of chunks in the dataset.

        Uses chunks_size and total_frames to determine chunk count.
        """
        chunks_size = self.data.get("chunks_size", 1000)
        total_frames = self.data.get("total_frames", 0)

        if total_frames == 0:
            return 0

        # Calculate number of chunks (ceiling division)
        return (total_frames + chunks_size - 1) // chunks_size

    def get_fps(self) -> int:
        """Get frames per second from dataset info."""
        return self.data.get("fps", 30)

    def get_video_codec(self, video_key: str) -> str:
        """
        Get video codec for a specific video stream.

        Args:
            video_key: The video key (e.g., "observation.images.front")

        Returns:
            Video codec (e.g., "av1", "h264")
        """
        features = self.data.get("features", {})
        video_info = features.get(video_key, {})

        if isinstance(video_info, dict):
            codec = video_info.get("info", {}).get("video.codec", "h264")
            return codec

        return "h264"  # Default fallback

    def get_video_frame_id(self, video_key: str) -> str:
        """
        Generate frame_id for a video stream.

        Args:
            video_key: The video key (e.g., "observation.images.front")

        Returns:
            Frame ID (e.g., "front_camera")
        """
        # Extract the last part of the video key as frame_id
        # "observation.images.front" -> "front_camera"
        parts = video_key.split(".")
        camera_name = parts[-1] if parts else "camera"
        return f"{camera_name}_camera"

    def get_total_episodes(self) -> int:
        """Get total number of episodes in the dataset."""
        return self.data.get("total_episodes", 0)

    def __repr__(self) -> str:
        return (
            f"DatasetInfo(episodes={self.get_total_episodes()}, "
            f"chunks={self.get_total_chunks()}, "
            f"fps={self.get_fps()}, "
            f"videos={len(self.video_keys)})"
        )
