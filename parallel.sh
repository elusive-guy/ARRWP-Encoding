CONFIG_DIR=$1
REPEAT=$2
MAX_JOBS=${3:-2}
MAIN=${4:-main}


(
  trap 'kill 0' SIGINT
  for CONFIG in "$CONFIG_DIR"/*.yaml; do
    if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
      while true; do
        CUR_JOBS=$(jobs | grep Running | wc -l)
        if [ $CUR_JOBS -ge $MAX_JOBS ]; then
          sleep 1
        else
          break
        fi
      done
      echo "Job launched: $CONFIG"
      python $MAIN.py --cfg $CONFIG --repeat $REPEAT --mark_done &
      CUR_JOBS=$(jobs | grep Running | wc -l)
      ((CUR_JOBS < MAX_JOBS)) && sleep 60
    fi
  done

  wait
)
