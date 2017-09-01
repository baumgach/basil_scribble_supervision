#!/bin/bash

## project settings
VIRTUAL_ENV_PATH=/scratch_net/biwidl102/bmustafa/python_environments/watchdog/

PROJECT_HOME=/scratch_net/biwidl102/bmustafa/acdc_segmenter_internal/
EXP_NAME=$1

## for pyenv
export PATH="/home/bmustafa/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

## activate virtual environment
source $VIRTUAL_ENV_PATH/bin/activate

LOG_DIR=$PROJECT_HOME/acdc_logdir/$EXP_NAME
WATCHING=1

while [ $WATCHING -eq 1 ]; do
    ## get time elapsed since most recent edit
    MOST_RECENT_FILE="$(find $LOG_DIR -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")"
    TIME_ELAPSED=$(($(date +%s) - $(date +%s -r $MOST_RECENT_FILE)))
    TIME_ELAPSED_MINS=$(($TIME_ELAPSED / 60))
    echo $TIME_ELAPSED_MINS
    SLEEP_TIME=1m
    TERMINATE=0

        if [ $TIME_ELAPSED_MINS -gt 10 ];
            then
                SLEEP_TIME=10m
                if [ $TIME_ELAPSED_MINS -gt 60 ];
                    then
                        TERMINATE=1;
                        WATCHING=0
                        SLEEP_TIME=0
                fi;
                python $PROJECT_HOME/watchdog.py $TIME_ELAPSED_MINS $EXP_NAME $TERMINATE
        fi;
    sleep $SLEEP_TIME
done

echo "Watchdog terminated"