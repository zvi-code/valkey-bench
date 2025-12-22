#!/bin/bash
PID=$1
PERF="perf"
echo "=== Cache performance ==="
$PERF stat -e stalled-cycles-frontend,stalled-cycles-backend,cycles,context-switches,cpu-migrations,cache-misses,cache-references,cycles,instructions \
  -t $PID -- sleep 10

echo -e "\n=== Pipeline stalls ==="
$PERF stat -e stalled-cycles-frontend,stalled-cycles-backend,cycles \
  -t $PID -- sleep 10

echo -e "\n=== Context switches ==="
$PERF stat -e context-switches,cpu-migrations \
  -t $PID -- sleep 10
echo -e "\n=== Cache misses ==="
$PERF stat -e cache-misses,cache-references \
  -t $PID -- sleep 10