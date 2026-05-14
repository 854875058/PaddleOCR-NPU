#!/usr/bin/env bash
# Re-exec under bash when invoked via `sh` / `dash` (avoids "Illegal option -o pipefail").
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

# Locate the project root by finding start_server.py near the script.
# Works whether the script is placed at <root>/scripts/ocr_service.sh or directly at <root>/ocr_service.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/start_server.py" ]; then
  ROOT_DIR="$SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/../start_server.py" ]; then
  ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  echo "ERROR: cannot locate start_server.py near $SCRIPT_DIR" >&2
  exit 1
fi
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

CMD="${1:-start}"
if [ $# -gt 0 ]; then
  shift
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6663}"
PHYSICAL_NPU_DEVICE_IDS="${PHYSICAL_NPU_DEVICE_IDS:-0,1,2,3}"
SERVICE_LOCAL_NPU_DEVICE_IDS="${SERVICE_LOCAL_NPU_DEVICE_IDS:-1,2,3}"
MIN_INSTANCES="${MIN_INSTANCES:-1}"
MAX_INSTANCES="${MAX_INSTANCES:-32}"
PER_CARD_MAX="${PER_CARD_MAX:-0}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:-120}"
SCALE_COOLDOWN="${SCALE_COOLDOWN:-20}"
BATCH_ACQUIRE_WAIT="${BATCH_ACQUIRE_WAIT:-6}"
INSTANCE_HBM_MB="${INSTANCE_HBM_MB:-8000}"
HBM_SAFETY_MARGIN_MB="${HBM_SAFETY_MARGIN_MB:-6144}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/ocr_service.log}"
PID_FILE="${PID_FILE:-$LOG_DIR/ocr_service.pid}"

mkdir -p "$LOG_DIR"

is_pid_running() {
  local pid="$1"
  if [ -z "$pid" ]; then
    return 1
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    return 1
  fi
  local cmdline
  cmdline="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  [[ "$cmdline" == *"python start_server.py"* ]]
}

read_pid() {
  if [ -f "$PID_FILE" ]; then
    cat "$PID_FILE" 2>/dev/null || true
  fi
}

cleanup_stale_pid() {
  local pid
  pid="$(read_pid)"
  if [ -n "$pid" ] && ! is_pid_running "$pid"; then
    rm -f "$PID_FILE"
  fi
}

port_in_use() {
  python - "$PORT" <<'PY'
import socket, sys
port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("0.0.0.0", port))
except OSError:
    sys.exit(0)
else:
    sys.exit(1)
finally:
    s.close()
PY
}

port_pids() {
  local pids=""
  if command -v ss >/dev/null 2>&1; then
    pids="$(ss -ltnp 2>/dev/null | awk -v port=":$PORT" '
      index($4, port) {
        while (match($0, /pid=[0-9]+/)) {
          pid = substr($0, RSTART + 4, RLENGTH - 4)
          print pid
          $0 = substr($0, RSTART + RLENGTH)
        }
      }' | sort -u)"
  elif command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | sort -u)"
  elif command -v fuser >/dev/null 2>&1; then
    pids="$(fuser "${PORT}/tcp" 2>/dev/null | tr ' ' '\n' | sed '/^$/d' | sort -u)"
  fi

  if [ -n "$pids" ]; then
    printf '%s\n' "$pids"
  fi
}

wait_for_port_release() {
  local retries="${1:-20}"
  local delay="${2:-0.5}"
  local i
  for ((i=0; i<retries; i++)); do
    if ! port_in_use; then
      return 0
    fi
    sleep "$delay"
  done
  return 1
}

stop_port_processes() {
  local pids
  pids="$(port_pids || true)"
  if [ -n "$pids" ]; then
    echo "Port $PORT is occupied by PID(s): $(echo "$pids" | tr '\n' ' ' | xargs)"

    local pid
    while IFS= read -r pid; do
      [ -n "$pid" ] || continue
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
      fi
    done <<< "$pids"

    if wait_for_port_release 10 0.5; then
      return 0
    fi

    echo "Port $PORT still busy after graceful stop, sending SIGKILL"
    while IFS= read -r pid; do
      [ -n "$pid" ] || continue
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
      fi
    done <<< "$pids"

    if wait_for_port_release 10 0.5; then
      return 0
    fi
  fi

  echo "Could not resolve listener PID from ss/lsof/fuser, fallback to pkill"
  pkill -f "python start_server.py.*--port $PORT" 2>/dev/null || true
  pkill -f "uvicorn.*$PORT" 2>/dev/null || true
  sleep 1

  if wait_for_port_release 10 0.5; then
    return 0
  fi

  return 1
}

show_logs() {
  if [ -f "$LOG_FILE" ]; then
    exec tail -f "$LOG_FILE"
  else
    echo "Log file not found: $LOG_FILE"
    exit 1
  fi
}

start_service() {
  cleanup_stale_pid

  local old_pid
  old_pid="$(read_pid)"
  if [ -n "$old_pid" ] && is_pid_running "$old_pid"; then
    echo "OCR service is already running with PID=$old_pid"
    echo "Log file: $LOG_FILE"
    return 0
  fi

  if port_in_use; then
    echo "Port $PORT is already in use. Trying to clean stale listener(s)..."
    stop_port_processes || true
    if port_in_use; then
      echo "Port $PORT is still in use after cleanup. Refusing to start."
      local occupied_pids
      occupied_pids="$(port_pids || true)"
      if [ -n "$occupied_pids" ]; then
        echo "Remaining PID(s): $(echo "$occupied_pids" | tr '\n' ' ' | xargs)"
      fi
      exit 1
    fi
  fi

  export ASCEND_RT_VISIBLE_DEVICES="$PHYSICAL_NPU_DEVICE_IDS"
  unset ASCEND_VISIBLE_DEVICES || true

  IFS=',' read -r -a _SERVICE_LOCAL_NPU_ARRAY <<< "$SERVICE_LOCAL_NPU_DEVICE_IDS"
  if [ "${#_SERVICE_LOCAL_NPU_ARRAY[@]}" -eq 0 ]; then
    echo "SERVICE_LOCAL_NPU_DEVICE_IDS is empty"
    exit 1
  fi
  local primary_id="${_SERVICE_LOCAL_NPU_ARRAY[0]}"

  echo "Starting OCR service with:"
  echo "  host=$HOST"
  echo "  port=$PORT"
  echo "  physical_npu_device_ids=$PHYSICAL_NPU_DEVICE_IDS"
  echo "  service_local_npu_device_ids=$SERVICE_LOCAL_NPU_DEVICE_IDS"
  echo "  local_primary_npu_device_id=$primary_id"
  echo "  ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
  echo "  min_instances=$MIN_INSTANCES"
  echo "  max_instances=$MAX_INSTANCES"
  echo "  per_card_max=$PER_CARD_MAX  (0 = unlimited; rely on HBM only)"
  echo "  idle_timeout=$IDLE_TIMEOUT"
  echo "  scale_cooldown=$SCALE_COOLDOWN"
  echo "  batch_acquire_wait=$BATCH_ACQUIRE_WAIT"
  echo "  instance_hbm_mb=$INSTANCE_HBM_MB"
  echo "  hbm_safety_margin_mb=$HBM_SAFETY_MARGIN_MB"
  echo "  log_file=$LOG_FILE"
  echo "  pid_file=$PID_FILE"

  # setsid 让子进程脱离当前 session，避免 SSH 关闭/Ctrl+C 杀掉服务
  setsid nohup python start_server.py \
    --host "$HOST" \
    --port "$PORT" \
    --npu_device_ids "$SERVICE_LOCAL_NPU_DEVICE_IDS" \
    --npu_device_id "$primary_id" \
    --min_instances "$MIN_INSTANCES" \
    --max_instances "$MAX_INSTANCES" \
    --per_card_max "$PER_CARD_MAX" \
    --idle_timeout "$IDLE_TIMEOUT" \
    --scale_cooldown "$SCALE_COOLDOWN" \
    --batch_acquire_wait "$BATCH_ACQUIRE_WAIT" \
    --instance_hbm_mb "$INSTANCE_HBM_MB" \
    --hbm_safety_margin_mb "$HBM_SAFETY_MARGIN_MB" \
    > "$LOG_FILE" 2>&1 < /dev/null &

  local new_pid=$!
  disown $new_pid 2>/dev/null || true
  echo "$new_pid" > "$PID_FILE"
  echo "OCR service started in background with PID=$new_pid"
  echo "Log file: $LOG_FILE"
  echo "Tail logs:  bash scripts/ocr_service.sh logs"
  echo "Status:     bash scripts/ocr_service.sh status"
}

stop_service() {
  cleanup_stale_pid
  local pid
  pid="$(read_pid)"

  if [ -n "$pid" ]; then
    if is_pid_running "$pid"; then
      kill "$pid"
      echo "Stopped OCR service PID=$pid"
    else
      echo "Process PID=$pid is not running"
    fi
  else
    echo "OCR service is not running"
  fi

  if port_in_use; then
    echo "Port $PORT is still occupied, stopping listener(s) by port"
    stop_port_processes || true
  fi
  rm -f "$PID_FILE"
}

status_service() {
  cleanup_stale_pid
  local pid
  pid="$(read_pid)"
  if [ -n "$pid" ] && is_pid_running "$pid"; then
    echo "OCR service is running with PID=$pid"
    echo "Log file: $LOG_FILE"
    return 0
  fi
  echo "OCR service is not running"
  return 1
}

restart_service() {
  stop_service || true
  sleep 1
  start_service
}

case "$CMD" in
  start)
    start_service
    ;;
  stop)
    stop_service
    ;;
  restart)
    restart_service
    ;;
  status)
    status_service
    ;;
  logs)
    show_logs
    ;;
  *)
    echo "Usage: bash scripts/ocr_service.sh {start|stop|restart|status|logs}"
    exit 1
    ;;
esac
