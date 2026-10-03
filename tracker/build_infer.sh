#!/usr/bin/env bash
# Build libanti_uav_infer.so on the RK3588 board.
# RKNN_INCLUDE: directory containing rknn_api.h (from rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include)
# RKNN_LIB:     librknnrt.so (default: system install)
set -euo pipefail
cd "$(dirname "$0")"
RKNN_INCLUDE=${RKNN_INCLUDE:-/usr/include}
RKNN_LIB=${RKNN_LIB:-/usr/lib/librknnrt.so}
g++ -O3 -std=c++17 -Wall -Wextra -Werror -fPIC -shared \
  -march=armv8.2-a+fp16 -I"$RKNN_INCLUDE" native_infer.cpp \
  "$RKNN_LIB" -Wl,-rpath,"$(dirname "$RKNN_LIB")" -o libanti_uav_infer.so.new
mv -f libanti_uav_infer.so.new libanti_uav_infer.so
