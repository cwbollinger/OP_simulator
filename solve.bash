#! /usr/bin/env bash

# https://stackoverflow.com/questions/59895/get-the-source-directory-of-a-bash-script-from-within-the-script-itself
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#source ~/anaconda3/bin/activate
$SCRIPT_DIR/time_window_service_times_solver.py
