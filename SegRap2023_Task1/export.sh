#!/usr/bin/env bash

# chmod 777 ./build.sh
# ./build.sh

docker save segrap2023_oar_hust:v1 | gzip -c > OAR_TT_CD_Pp.tar.gz
