# lerobot2mcap

Convert LeRobot datasets to MCAP format. The config file is automatically generated from your dataset metadata - no manual setup required.

## Features

- **Automatic configuration**: Reads `meta/info.json` and generates all necessary configuration
- **Episode-based conversion**: Converts each episode to a separate MCAP file in its own directory
- **Chunk-aware**: Handles datasets organized in chunks, ready for multi-chunk datasets
- **Multi-video support**: Auto-detects and converts all video streams
- **Terminal log support**: Parses raw `.log` files into `rcl_interfaces/msg/Log` messages with full metadata
- **ROS2 format**: Outputs ROS2-compatible MCAP files
- **Cross-compatible**: Supports LeRobot v2.0, v2.1, and v3 dataset formats automatically

## Architecture

The converter uses an object-oriented architecture:

- **DatasetInfo**: Parses `meta/info.json` to extract dataset structure (FPS, video streams, codecs, episodes, chunks)
- **ConfigGenerator**: Automatically generates episode-specific configurations from dataset metadata
- **LeRobotConverter**: Orchestrates the conversion process, iterating through chunks and episodes
- Uses **tabular2mcap** under the hood for MCAP writing

The converter processes datasets by iterating through each chunk and converting all episodes within that chunk to separate MCAP files.

## Usage

### Download Dataset

If you wish to use an existing dataset offered by Hugging Face, use the following commands.

```bash
# Download from Hugging Face
uv run lerobot2mcap download lerobot/pusht -o ./data

# Download specific episodes
uv run lerobot2mcap download lerobot/pusht -o ./data -e 0 1 2
```

### Record Your Own Dataset With a .log File
If you wish to record a regular dataset without the additional .log file, HuggingFace's tutorials have that covered. To record a dataset while capturing the terminal output, use the following command:

```bash
LOG_FILE=~/.cache/huggingface/lerobot/${HF_USER}/record-test.log
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A680114161 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ 
    front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30},
    external: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 15}     
    }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5A680123701 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/record-test
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=0 \
    --dataset.num_episodes=1 \
    --dataset.single_task="recording edge cases" \
    2>&1 | tee $LOG_FILE

```

### Convert to MCAP

```bash
# Convert all episodes from all chunks (configuration generated automatically)
uv run lerobot2mcap convert ~/.cache/huggingface/lerobot/{repo-id} -o ./mcap_output

# Convert specific episodes
uv run lerobot2mcap convert ~/.cache/huggingface/lerobot/{repo-id} -o ./mcap_output -e 0 1 2

# Advanced: Custom converter functions (optional)
uv run lerobot2mcap convert ~/.cache/huggingface/lerobot/{repo-id} -o ./mcap_output -f ./my_converter_functions.yaml
```

### Expected Dataset Structure

The converter automatically detects and supports both LeRobot v2.0, v2.1, and v3 dataset formats. Example input file structures are given below for v3 and v2.1 (very similar to v2). 

#### LeRobot v3 Format

Multiple episodes will be concatenated into the same files, based on episode and MP4 file size limits defined in the lerobot codebase. The following file structure is an example only; the method and conditions, and recording parameters that data is collected with will dicate how many episodes are merged per file.  
```
dataset_root/
├── meta/
│   ├── info.json           # Dataset metadata (includes total_episodes)
│   ├── episodes.jsonl
│   └── tasks.jsonl
├── data/
│   └── chunk-000/
│       ├── file-000.parquet  # Episode 0-3
│       ├── file-001.parquet  # Episode 4-5
│       └── file-002.parquet  # Episode 6-9
├── videos/
│   ├── observation.images.front/
│   │   └── chunk-000/
│   │       ├── file-000.mp4  # Episode 0-3
│   │       ├── file-001.mp4  # Episode 4-5
│   │       └── file-002.mp4  # Episode 6-9
│   └── observation.images.external/
│       └── chunk-000/
│           ├── file-000.mp4  # Episode 0-3
│           ├── file-001.mp4  # Episode 4-5
│           └── file-002.mp4  # Episode 6-9
└── recording.log           # Optional terminal log
```

#### LeRobot v2.1 Format

Episodes use 6-digit indices (episode_000000, episode_000001, etc.).

```
dataset_root/
├── meta/
│   ├── info.json           # Dataset metadata (includes total_episodes)
│   ├── episodes.jsonl
│   └── tasks.jsonl
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet  # Episode 0
│       ├── episode_000001.parquet  # Episode 1
│       └── episode_000002.parquet  # Episode 2
└── videos/
    └── chunk-000/
        ├── observation.images.phone/
        │   ├── episode_000000.mp4  # Episode 0
        │   ├── episode_000001.mp4  # Episode 1
        │   └── episode_000002.mp4  # Episode 2
        └── observation.images.external/
            ├── episode_000000.mp4  # Episode 0
            ├── episode_000001.mp4  # Episode 1
            └── episode_000002.mp4  # Episode 2
```

**Note**: The converter automatically detects which format your dataset uses from `meta/info.json` and handles it appropriately.

## Output

Each episode produces a separate directory containing an MCAP file and its configuration:

```
mcap_output/
├── episode_000/
│   ├── episode_000.mcap
│   └── config_000.yaml
├── episode_001/
│   ├── episode_001.mcap
│   └── config_001.yaml
└── episode_002/
    ├── episode_002.mcap
    └── config_002.yaml
```

Each MCAP file contains:
- **Robot data** from the episode's parquet file (topic: `robot_data`)
- **Video streams** for each camera (topics: `observation/images/front`, `observation/images/external`, etc.)
- **Terminal logs** as `rcl_interfaces/msg/Log` if `.log` files present (topic: `terminal_log`)
  - Includes full metadata: log level, timestamp, source file, line number, and message
  - Supports multi-line log entries (e.g., stack traces)

## How Configuration Works

**You don't need to create any configuration files** - everything is automatic:

1. **DatasetInfo** reads your `meta/info.json` to discover:
   - Total number of episodes
   - Video streams and their codecs
   - Dataset FPS and structure
   - Data file patterns

2. **ConfigGenerator** creates a config for each episode:
   - Tabular mappings pointing to the specific episode's parquet file
   - Video mappings pointing to the specific episode's video files
   - Log mappings if `.log` files are present
   - Each config is saved alongside its MCAP file for transparency

3. **tabular2mcap** uses each generated config to convert the episode to MCAP

### Advanced: Custom Converter Functions (Optional)

By default, the converter uses [`configs/converter_functions.yaml`](configs/converter_functions.yaml) for data transformation. You can customize this with the `-f` flag:

```yaml
# Custom converter_functions.yaml
functions:
  row_to_message_with_timestamp:
    schema_name: null
    template: |
      {
        "timestamp": {
          "sec": {{ (timestamp) | int }},
          "nsec": {{ ((timestamp % 1) * 1_000_000_000) | int }}
        }
      }
```

**Note**: Log parsing is handled automatically by tabular2mcap's `LogConverter` - no converter function needed.

## Development

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Build package
uv build
```

## Browse Datasets

https://huggingface.co/lerobot
