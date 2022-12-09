#!/usr/bin/env bash

if [[ $# -ne 1 ]]; then
    echo Usage: bash $0 WORK_DIR
    exit 1
fi

set -e

# CLI args
WORK_DIR=$1
LAST_ROUND=$WORK_DIR/last_round

if [ ! -d "$LAST_ROUND" ];then
  mkdir "$LAST_ROUND"
fi

[ -d "$WORK_DIR/predictions" ] && mv $WORK_DIR/predictions/* $LAST_ROUND
[ -f "$WORK_DIR/test*.txt" ] && mv $WORK_DIR/test*.txt $LAST_ROUND

