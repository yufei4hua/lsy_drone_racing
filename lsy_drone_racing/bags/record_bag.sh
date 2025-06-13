#!/bin/bash

eval "$(mamba shell hook --shell=bash)"
mamba activate race

BAG_NAME=${1:-my_bag}
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_NAME="${BAG_NAME}_${TIMESTAMP}"

shift
TOPICS="$@"

if [ -z "$TOPICS" ]; then
  echo "No topics specified!"
  exit 1
fi

echo "Recording to: $FULL_NAME"
echo "Topics: $TOPICS"
ros2 bag record -o "$FULL_NAME" $TOPICS