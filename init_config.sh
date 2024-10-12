#!/bin/bash/
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CURRENT_DIR/scheduler/config_init_creator.sh"
tail -f /dev/null