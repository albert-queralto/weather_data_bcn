#!/bin/bash/
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CURRENT_DIR/scheduler/config_init_creator.sh"

exec uvicorn 'backend.app:app' --reload --workers 1 --host 0.0.0.0 --port 8100
tail -f /dev/null

