#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
g++ -O3 -std=c++17 -fPIC -shared native_io.cpp -o libnative_io.tmp.so \
  $(pkg-config --cflags --libs gstreamer-app-1.0 gstreamer-allocators-1.0 gstreamer-video-1.0 librga)
mv libnative_io.tmp.so libnative_io.so
