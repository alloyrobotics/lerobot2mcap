"""Dataset information parser for LeRobot datasets.
Parses the info.json file to extract metadata about the dataset,"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default values for dataset metadata
DEFAULT_FPS = 30
DEFAULT_CODEC = "h264"
DEFAULT_FILE_INDEX = 0  # First file in a chunk
DEFAULT_DATA_PATH_TEMPLATE = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
DEFAULT_VIDEO_PATH_TEMPLATE = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


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

    @property
    def data_path_template(self) -> str:
        """
        Get the data (parquet) path template from dataset metadata.
        Checking data_path value for robustness - falls back to default if not in info.json.
        """
        return self.data.get("data_path", DEFAULT_DATA_PATH_TEMPLATE)

    @property
    def video_path_template(self) -> str:
        """
        Get the video path template from dataset metadata.
        Checking video_path value for robustness - falls back to default if not in info.json.
        """
        return self.data.get("video_path", DEFAULT_VIDEO_PATH_TEMPLATE)

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

        # Get parquet file path using property (checks info.json with fallback)
        parquet_path = self.data_path_template.format(
            chunk_index=chunk_index,
            file_index=DEFAULT_FILE_INDEX
        )
        chunk_files["parquet"] = dataset_root / parquet_path

        # Get video file paths using property (checks info.json with fallback)
        chunk_files["videos"] = {}

        for video_key in self.video_keys:
            video_path = self.video_path_template.format(
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=DEFAULT_FILE_INDEX
            )
            chunk_files["videos"][video_key] = dataset_root / video_path

        return chunk_files

    def get_total_chunks(self) -> int:
        """
        Get total number of chunks in the dataset.
        Assumes single chunk (chunk-000) structure for LeRobot datasets.
        Keeping function here as a placeholder for future multi-chunk support if necessary.
        """
        return 1

    def get_fps(self) -> int:
        """Get frames per second from dataset info."""
        return self.data.get("fps", DEFAULT_FPS)

    def get_video_codec(self, video_key: str) -> str:
        """
        Get video codec for a specific video stream.
        Args:
            video_key: The video key (e.g., "observation.images.front")
        Returns:
            Video codec (e.g., "av1", "h264")
        """
        #### Commented out for testing ### 

        # features = self.data.get("features", {})
        # video_info = features.get(video_key, {})

        # if isinstance(video_info, dict):
        #     codec = video_info.get("info", {}).get("video.codec", DEFAULT_CODEC)
        #     return codec

        return DEFAULT_CODEC

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
        camera_name = video_key.split(".")[-1] if "." in video_key else "camera"
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
