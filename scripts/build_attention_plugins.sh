#!/usr/bin/env bash
# Build the swappable attention algorithms into plugins/.
# The console lists whatever is present there; a missing library is reported as
# a load error rather than being silently replaced.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build" --target attention_av_memory_language_v1 attention_av_memory_language_v2 attention_plugin_v1_test attention_plugin_v2_test attention_surround_test -j "$(getconf _NPROCESSORS_ONLN)"
for check in attention_plugin_v1_test attention_plugin_v2_test attention_surround_test; do
  "$ROOT/build/$check"
done
ls -l "$ROOT/plugins"/libattention_*.so
