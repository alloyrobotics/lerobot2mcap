# lerobot2mcap

Convert LeRobot datasets to MCAP format.

## Usage

```bash
# Download dataset
uv run lerobot2mcap download lerobot/pusht -o ./data

# Convert to MCAP
uv run lerobot2mcap convert lerobot/pusht -o ./data

# Specific episodes
uv run lerobot2mcap download lerobot/pusht -o ./data -e 0 1 2
```

Browse datasets: https://huggingface.co/lerobot
