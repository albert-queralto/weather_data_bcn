#!/bin/bash


# Function to run the script in the background
run_preprocessor() {
    local file_path="/root/home/src/preprocessing/main.py"
    local start_date=$(date -d "-7 day" +"%Y-%m-%d")
    local end_date=$(date +"%Y-%m-%d")
    local preprocessor_id=1
    local model_id=1

    bash -c "
echo '$(date +"%Y-%m-%d %T") | Executing daily_preprocessing.sh'
echo 'Start date: $start_date | End date: $end_date'
python '$file_path' -lat 41.389 -lon 2.159 -sd '$start_date' -ed '$end_date'
" > /root/home/logs/daily_preprocessing.log 2>&1 &
}

run_preprocessor