# lerobot2mcap

Convert LeRobot datasets to MCAP format. The config file is automatically generated from your dataset metadata - no manual setup required.

## Features

- **Automatic configuration**: Reads `meta/info.json` and generates all necessary configuration
- **Chunk-based conversion**: Converts LeRobot datasets organized in chunks
- **Multi-video support**: Auto-detects and converts all video streams
- **Terminal log support**: Parses raw `.log` files into `rcl_interfaces/msg/Log` messages with full metadata
- **ROS2 format**: Outputs ROS2-compatible MCAP files

## Architecture

The converter uses an object-oriented architecture:

- **DatasetInfo**: Parses `meta/info.json` to extract dataset structure (FPS, video streams, codecs, etc.)
- **ConfigGenerator**: Automatically generates configuration from dataset metadata
- **LeRobotConverter**: Orchestrates the conversion process
- Uses **tabular2mcap** under the hood for MCAP writing

## Usage

### Download Dataset

If you wish to use an existing dataset offered by Hugging Face, use the following commands.

```bash
# Download from Hugging Face
uv run lerobot2mcap download lerobot/pusht -o ./data

# Download specific episodes
uv run lerobot2mcap download lerobot/pusht -o ./data -e 0 1 2
```

### Convert to MCAP

```bash
# Convert all chunks (configuration generated automatically)
uv run lerobot2mcap convert ~/.cache/huggingface/lerobot/{repo-id} -o ./mcap_output

# Convert specific chunks
uv run lerobot2mcap convert ~/.cache/huggingface/lerobot/{repo-id} -o ./mcap_output -c 0 1

# Advanced: Custom converter functions (optional)
uv run lerobot2mcap convert ~/.cache/huggingface/lerobot/{repo-id} -o ./mcap_output -f ./my_converter_functions.yaml
```

### Expected Dataset Structure

Video file structure is an example only. Depending on hardware setup subfile names may change

```
dataset_root/
├── meta/
│   └── info.json           # Dataset metadata
├── data/
│   └── chunk-000/
│       └── file-000.parquet
├── videos/
│   ├── observation.images.front/
│   │   └── chunk-000/
│   │       └── file-000.mp4
│   └── observation.images.external/
│       └── chunk-000/
│           └── file-000.mp4
└── recording.log           # Optional terminal log
```

## Output

Each chunk produces one MCAP file with:
- **Robot data** from parquet files (topic: `robot_data`)
- **Video streams** for each camera (topics: `observation/images/front`, `observation/images/external`, etc.)
- **Terminal logs** as `rcl_interfaces/msg/Log` if `.log` files present (topic: `terminal_log`)
  - Includes full metadata: log level, timestamp, source file, line number, and message
  - Supports multi-line log entries (e.g., stack traces)

## How Configuration Works

**You don't need to create any configuration files** - everything is automatic:

1. **DatasetInfo** reads your `meta/info.json` to discover:
   - Video streams and their codecs
   - Dataset FPS and structure
   - Data file patterns

2. **ConfigGenerator** creates a config for each chunk:
   - Tabular mappings for parquet files
   - Video mappings for each camera stream
   - Log mappings if `.log` files are present

3. **tabular2mcap** uses this generated config to convert to MCAP

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
