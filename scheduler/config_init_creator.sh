#!/bin/bash

SCHEDULING_TASKS_DIR="/root/home/scheduler"
mkdir -p "$SCHEDULING_TASKS_DIR"
chmod 777 "$SCHEDULING_TASKS_DIR"

config_file="./config.ini"

mkdir -p "$SCHEDULING_TASKS_DIR"
touch $config_file

cat <<EOF >>"$config_file"

[job-exec "daily_preprocessing"]
schedule = @daily
command = bash /root/home/scheduler/daily_preprocessing_bcn.sh 2>&1 | tee /root/home/logs/daily_preprocessing.log
user = root
tty = false
container = weather_data_bcn
no-overlap = true

EOF
done

chmod +x "$config_file"