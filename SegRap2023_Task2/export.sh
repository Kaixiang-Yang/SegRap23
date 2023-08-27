#!/usr/bin/env bash

# chmod 777 ./build.sh
# ./build.sh

docker save segrap2023_gtv:v1 | gzip -c > GTV_TT_CD.tar.gz
