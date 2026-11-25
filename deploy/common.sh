# shellcheck shell=bash
#
# common.sh - functions shared by deploy/server-setup.sh and deploy/deploy.sh (sourced, never run directly).
# Ported from the uniform project's deploy scripts (YK_CCTV_U_D_M) and adapted to the litter app:
# LM_* data paths, the litter_monitor / auth SQLite databases, the v3_store_cctv model checksum, the
# monitored-region (ROI) segmenter and the API's local-only guard.
#
# The calling script sets the configuration variables (APP_DIR, APP_PORT, WEB_PORT, ...) before sourcing.

say()  { echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }

STORAGE_DIR=""; DB_PATH=""; AUTH_DB_PATH=""; MODELS_DIR=""; LAST_BACKUP=""

pyexec() {   # the app's Python when the venv exists, else the system python3 (snippets use the stdlib only)
  if [[ -x "$APP_DIR/.venv/bin/python" ]]; then "$APP_DIR/.venv/bin/python" "$@"; else python3 "$@"; fi
}

require_tools() {
  local t
  for t in "$@"; do
    command -v "$t" >/dev/null 2>&1 || die "'$t' is not installed. Install it: sudo apt-get install -y $t"
  done
}

check_settings() {
  [[ "$BACKUP_KEEP" =~ ^[1-9][0-9]*$ ]] || die "BACKUP_KEEP must be a whole number >= 1 (got '$BACKUP_KEEP')"
  [[ "$TORCH_VARIANT" == "cpu" || "$TORCH_VARIANT" == "cuda" ]] || die "TORCH_VARIANT must be cpu or cuda"
  [[ "$APP_PORT" =~ ^[0-9]+$ && "$WEB_PORT" =~ ^[0-9]+$ ]] || die "APP_PORT and WEB_PORT must be numbers"
  [[ "$APP_PORT" != "$WEB_PORT" ]] || die "APP_PORT and WEB_PORT must differ"
  case "$BACKUP_DIR/" in
    "$APP_DIR"/*) die "BACKUP_DIR ($BACKUP_DIR) must be outside APP_DIR ($APP_DIR)" ;;
  esac
}

repo_id() {   # git URL -> host/owner/repo (https and ssh forms compare equal)
  local u="$1"
  u="${u%.git}"; u="${u%/}"; u="${u#*://}"; u="${u#*@}"; u="${u/:/\/}"
  echo "${u,,}"
}

python_ok() {   # $1 = interpreter; the pinned packages (deploy/constraints.txt) need Python 3.12 - 3.14
  "$1" -c 'import sys; sys.exit(0 if (3, 12) <= sys.version_info[:2] <= (3, 14) else 1)' 2>/dev/null
}

load_node() {   # nvm is not loaded in non-interactive shells: load it when present
  if [[ -s "$NVM_DIR/nvm.sh" ]]; then
    set +u; # shellcheck disable=SC1091
    . "$NVM_DIR/nvm.sh"; nvm use --silent default >/dev/null 2>&1 || true; set -u
  fi
}

node_ok() {   # Next.js 16 needs Node >= 20.9
  command -v node >/dev/null 2>&1 || return 1
  node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>20||(a===20&&b>=9)?0:1)'
}

require_pm2() {
  command -v pm2 >/dev/null 2>&1 || die "pm2 is not installed. Install it for this user, then re-run:
    npm install -g pm2
  (with nvm, no sudo is needed). Then: pm2 startup   (prints the command that starts pm2 on boot)"
  if [[ "$(dirname "$(command -v pm2)")" != "$(dirname "$(command -v node)")" ]]; then
    warn "pm2 ($(command -v pm2)) belongs to a different Node.js than the active one ($(command -v node)).
  After changing the Node version: npm install -g pm2 && pm2 update, then run 'pm2 startup' again."
  fi
}

# ------------------------------------------------------------------------------------- .env reading
env_value() {   # last value of KEY ($2) in FILE ($1), quotes and trailing comments removed; empty when unset
  [[ -f "$1" ]] || return 0
  sed -n "s/^[[:space:]]*\(export[[:space:]]\{1,\}\)\{0,1\}$2[[:space:]]*=[[:space:]]*//p" "$1" | tail -n 1 \
    | sed -e 's/[[:space:]]\{1,\}#.*$//' -e 's/[[:space:]]*$//' -e "s/^[\"']//" -e "s/[\"']\$//"
}

app_env() {   # the value the API sees for KEY ($1): process environment first, then .env (python-dotenv rule)
  local v="${!1:-}"
  if [[ -n "$v" ]]; then echo "$v"; else env_value "$APP_DIR/.env" "$1"; fi
}

resolve_paths() {   # where the app really keeps its data - same rules as backend/core/config.py
  local line
  STORAGE_DIR=""; DB_PATH=""; AUTH_DB_PATH=""; MODELS_DIR=""
  while IFS= read -r line; do
    line="${line%$'\r'}"
    case "$line" in
      STORAGE_DIR=*)  STORAGE_DIR="${line#*=}" ;;
      DB_PATH=*)      DB_PATH="${line#*=}" ;;
      AUTH_DB_PATH=*) AUTH_DB_PATH="${line#*=}" ;;
      MODELS_DIR=*)   MODELS_DIR="${line#*=}" ;;
    esac
  done < <(pyexec - "$APP_DIR" <<'PY'
import os, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
env = {}
dotenv = root / ".env"
if dotenv.exists():                               # same precedence as python-dotenv (override=False)
    for raw in dotenv.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k, v = k.strip().removeprefix("export ").strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        else:
            v = v.split(" #", 1)[0].strip()
        env[k] = v
get = lambda k: os.environ.get(k) or env.get(k) or ""
path = lambda v, default: (lambda p: p if p.is_absolute() else root / p)(Path(v) if v else default).resolve()
# backend/core/config.py: every default is relative to the project root (not to LM_STORAGE_DIR)
print(f"STORAGE_DIR={path(get('LM_STORAGE_DIR'), root / 'storage').as_posix()}")
print(f"DB_PATH={path(get('LM_DB_PATH'), root / 'storage' / 'litter_monitor.db').as_posix()}")
print(f"AUTH_DB_PATH={path(get('LM_AUTH_DB_PATH'), root / 'storage' / 'auth.db').as_posix()}")
print(f"MODELS_DIR={path(get('LM_MODELS_DIR'), root / 'models').as_posix()}")
PY
)
  [[ -n "$STORAGE_DIR" && -n "$DB_PATH" && -n "$MODELS_DIR" ]] \
    || die "Could not work out the storage / database / models paths (check $APP_DIR/.env)."
  case "$BACKUP_DIR/" in
    "$STORAGE_DIR"/*) die "BACKUP_DIR ($BACKUP_DIR) must be outside the storage folder ($STORAGE_DIR)" ;;
  esac
}

ensure_runtime_dirs() {
  # mkdir -p only: existing folders and their contents are never deleted, emptied or recreated
  local d
  for d in "$STORAGE_DIR" "$STORAGE_DIR/alerts" "$STORAGE_DIR/videos" "$STORAGE_DIR/uploads" "$STORAGE_DIR/media" \
           "$STORAGE_DIR/sampled" "$STORAGE_DIR/hard_examples" "$STORAGE_DIR/exports" \
           "$(dirname "$DB_PATH")" "$(dirname "$AUTH_DB_PATH")" "$MODELS_DIR/litter_yolo"; do
    mkdir -p "$d"
  done
}

# ------------------------------------------------------------------------------- Python / Node build
install_python_deps() {
  local py="$APP_DIR/.venv/bin/python" cons="$APP_DIR/deploy/constraints.txt" req pins
  [[ -f "$cons" ]] || die "deploy/constraints.txt is missing: it pins the package versions the tests passed with."
  req="$(mktemp)"; pins="$(mktemp)"
  if [[ "$INSTALL_DEV_DEPS" == "1" ]]; then
    cp "$APP_DIR/requirements.txt" "$req"
  else  # the running backend never imports these: tests / legacy training engine only
    grep -Ev '^[[:space:]]*(pytest|httpx|scikit-learn|joblib)([^A-Za-z0-9_-]|$)' "$APP_DIR/requirements.txt" > "$req"
  fi
  "$py" -m pip install --quiet --upgrade pip
  if [[ "$TORCH_VARIANT" == "cpu" ]]; then
    say "Installing CPU-only PyTorch (no CUDA libraries) at the pinned version"
    "$py" -m pip install --quiet --index-url "$TORCH_CPU_INDEX" -c "$cons" torch torchvision
    # pin the exact +cpu builds so the requirements below cannot swap them for the CUDA builds on PyPI
    "$py" -m pip freeze | grep -E '^(torch|torchvision)==' > "$pins"
    "$py" -m pip install --quiet -r "$req" -c "$cons" -c "$pins" --extra-index-url "$TORCH_CPU_INDEX"
  else
    "$py" -m pip install --quiet -r "$req" -c "$cons"
  fi
  rm -f "$req" "$pins"
}

verify_model() {   # read-only: the production weights must match the checksum recorded in git
  local file="$MODELS_DIR/litter_yolo/$MODEL_VERSION/best.pt" sums="$APP_DIR/$MODEL_SUMS" want have
  [[ -f "$file" ]] || die "Model weights missing: $file. Restore the approved $MODEL_VERSION/best.pt
  (it is tracked in git; with LM_MODELS_DIR set, copy it there). These scripts never replace a model."
  [[ -f "$sums" ]] || die "Checksum file missing: $sums (set MODEL_SUMS)."
  want="$(awk '$2 ~ /best\.pt$/ {print $1; exit}' "$sums")"
  have="$(sha256sum "$file" | cut -d' ' -f1)"
  [[ -n "$want" && "$have" == "$want" ]] || die "$file does not match $MODEL_SUMS
    expected sha256 $want
    found    sha256 $have
  Restore the approved model file - these scripts never replace or re-register a model."
  say "Model $MODEL_VERSION verified (sha256 ${have:0:12}...)"
}

verify_python_stack() {
  say "Checking that the backend packages import"
  EXPECT_CPU="$([[ "$TORCH_VARIANT" == "cpu" ]] && echo 1 || echo 0)" "$APP_DIR/.venv/bin/python" - <<'PY'
import os, sys
import boto3, cv2, fastapi, torch, torchvision, transformers, ultralytics, uvicorn
print(f"    python {sys.version.split()[0]} | torch {torch.__version__} (CUDA build: {torch.version.cuda or 'no'})"
      f" | ultralytics {ultralytics.__version__} | transformers {transformers.__version__}"
      f" | opencv {cv2.__version__} | boto3 {boto3.__version__}")
if os.environ.get("EXPECT_CPU") == "1" and torch.version.cuda:
    sys.exit("torch is a CUDA build although TORCH_VARIANT=cpu")
PY
  verify_model
}

ensure_roi_model() {   # monitored-region segmenter weights (models/roi_segmenter, ~105 MB, git-ignored)
  if [[ "$(app_env LDM_ROI)" == "off" ]]; then say "LDM_ROI=off - monitored-region model not needed"; return 0; fi
  if (cd "$APP_DIR" && .venv/bin/python -m backend.services.roi_service check >/dev/null 2>&1); then
    say "Monitored-region model present (models/roi_segmenter)"; return 0
  fi
  say "Monitored-region model missing: downloading it once (~105 MB, needs internet)"
  (cd "$APP_DIR" && .venv/bin/python -m backend.services.roi_service fetch) \
    || warn "Could not download the monitored-region model. Detection runs with roi.on_failure (full frame + a
  visible warning) until it is installed:  cd $APP_DIR && .venv/bin/python -m backend.services.roi_service fetch"
}

build_frontend() {
  say "Installing dashboard dependencies (npm ci, from package-lock.json)"
  (cd "$APP_DIR" && npm ci --include=dev --no-audit --no-fund)
  say "Building the dashboard (npm run build -> .next)"
  (cd "$APP_DIR" && NEXT_TELEMETRY_DISABLED=1 NEXT_DIST_DIR=.next npm run build)
}

# ---------------------------------------------------------------------------------------- env checks
check_env() {   # $1 = strict (stop on problems) | warn ; prints key NAMES and paths, never values
  local mode="$1" url cors cams s3 problems=()
  if [[ ! -f "$APP_DIR/.env" ]]; then
    problems+=("$APP_DIR/.env is missing. Create it from .env.example (see deploy/SERVER_DEPLOY.md, step 3).")
  else
    [[ -n "$(env_value "$APP_DIR/.env" ADMIN_API_TOKEN)" ]] \
      || problems+=("ADMIN_API_TOKEN is empty in .env: admin pages (cameras, settings, models) will refuse every change.")
    [[ "$(stat -c %a "$APP_DIR/.env")" =~ ^[67]00$ ]] || warn ".env is readable by other users: chmod 600 $APP_DIR/.env"
  fi

  cams="$(app_env LM_START_CAMERAS)"
  if [[ "${cams,,}" != "false" ]]; then
    warn "LM_START_CAMERAS is not 'false': the API will try to open the configured RTSP cameras. A cloud server
  normally cannot reach the store NVR - set LM_START_CAMERAS=false in .env unless it is on the store network / VPN."
  fi

  s3="$(app_env LDM_S3_UPLOAD)"
  if [[ "${s3,,}" == "on" ]]; then
    if [[ -z "$(app_env AWS_ACCESS_KEY_ID)" || -z "$(app_env AWS_SECRET_ACCESS_KEY)" ]]; then
      problems+=("LDM_S3_UPLOAD=on but AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are not set in .env.")
    fi
  fi

  # The dashboard bakes NEXT_PUBLIC_API_URL in at BUILD time; unset = http://localhost:8000.
  url="$(env_value "$APP_DIR/.env.local" NEXT_PUBLIC_API_URL)"
  [[ -n "$url" ]] || url="$(env_value "$APP_DIR/.env" NEXT_PUBLIC_API_URL)"
  if [[ -z "$url" ]]; then
    [[ "$APP_PORT" == "8000" ]] || problems+=("NEXT_PUBLIC_API_URL is unset (dashboard calls http://localhost:8000) but APP_PORT=$APP_PORT.
  Set NEXT_PUBLIC_API_URL=http://localhost:$APP_PORT in .env.local.")
  elif [[ ! "$url" =~ ^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?/?$ ]]; then
    warn "NEXT_PUBLIC_API_URL ($url) is not localhost. The API is LOCAL-ONLY (backend/api/local_only.py): it answers
  403 to every request that does not come from this server itself, including requests forwarded by Nginx.
  Browsers will show 'Backend offline' until login protection is enabled and that guard is lifted.
  Supported today: SSH tunnel access (deploy/SERVER_DEPLOY.md, step 5)."
  fi

  cors="$(app_env CORS_ORIGINS)"
  [[ -n "$cors" ]] || cors="http://localhost:3000,http://127.0.0.1:3000"     # backend/core/config.py default
  [[ ",$cors," == *",http://localhost:$WEB_PORT,"* ]] \
    || problems+=("CORS_ORIGINS does not contain http://localhost:$WEB_PORT - the browser will block every API call
  from the dashboard. Add it to CORS_ORIGINS in .env.")

  if (( ${#problems[@]} )); then
    local p msg=""
    for p in "${problems[@]}"; do msg+=$'\n'"  - $p"; done
    if [[ "$mode" == "strict" ]]; then die "Configuration problems:$msg"; else warn "Configuration problems:$msg"; fi
  fi
  say "Data locations: storage $STORAGE_DIR | database $DB_PATH | auth db $AUTH_DB_PATH | models $MODELS_DIR"
}

# ------------------------------------------------------------------------------------- job / health
active_jobs() {   # "<count> <details>" of queued / running work, read from the database (read-only)
  if [[ ! -f "$DB_PATH" ]]; then echo "0"; return 0; fi
  pyexec - "$DB_PATH" <<'PY'
import sqlite3, sys
con = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True, timeout=30)
active = {"jobs": "status IN ('queued','running')",            # image / video jobs
          "sampled_jobs": "state IN ('queued','running')",      # four-screenshot video analysis
          "analyses": "status = 'RUNNING'",
          "captures": "status = 'running'"}
tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
total, parts = 0, []
for table, where in active.items():
    if table in tables:
        n = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()[0]
        total += n
        if n:
            parts.append(f"{n} {table}")
print(total, ", ".join(parts))
PY
}

check_no_active_jobs() {
  local out n
  out="$(active_jobs)" || die "Could not read the job states from $DB_PATH - not restarting blind."
  out="${out%$'\r'}"; n="${out%% *}"
  [[ "$n" =~ ^[0-9]+$ ]] || die "Could not read the job states from $DB_PATH - not restarting blind."
  if (( n == 0 )); then say "No queued or running jobs"; return 0; fi
  if curl -fsS -m 5 -o /dev/null "http://127.0.0.1:$APP_PORT/api/health" 2>/dev/null; then
    [[ "$FORCE_RESTART" == "1" ]] || die "Jobs are queued or running: ${out#* }. A restart interrupts them (and any S3
  upload in progress; results saved so far are kept). Wait until they finish (pm2 logs $API_NAME), or re-run
  with FORCE_RESTART=1."
    warn "Restarting with active jobs (${out#* }) because FORCE_RESTART=1"
  else
    warn "The database lists ${out#* } as active, but the API is not running: they are left over from an earlier
  stop and the API marks them interrupted when it starts."
  fi
}

check_running_health() {   # before changing anything: do the apps that pm2 runs answer?
  local bad=()
  if [[ "$(pm2_status "$API_NAME")" == "online" ]] \
     && ! curl -fsS -m 10 -o /dev/null "http://127.0.0.1:$APP_PORT/api/health"; then
    bad+=("$API_NAME (http://127.0.0.1:$APP_PORT/api/health)")
  fi
  if [[ "$(pm2_status "$WEB_NAME")" == "online" ]] \
     && ! curl -fsS -m 10 -o /dev/null "http://127.0.0.1:$WEB_PORT/"; then
    bad+=("$WEB_NAME (http://127.0.0.1:$WEB_PORT/)")
  fi
  if (( ${#bad[@]} )); then
    [[ "$ALLOW_UNHEALTHY" == "1" ]] || die "Running but not answering: ${bad[*]}. Look at 'pm2 logs' first;
  ALLOW_UNHEALTHY=1 continues anyway (e.g. to ship the fix)."
    warn "Continuing although ${bad[*]} did not answer (ALLOW_UNHEALTHY=1)"
  fi
}

# --------------------------------------------------------------------------------------------- pm2
pm2_clean() {   # pm2 records the caller's environment: keep these scripts' own settings out of the apps
  env -u FORCE_RESET -u FORCE_RESTART -u ALLOW_MODEL_CHANGE -u ALLOW_UNHEALTHY -u ALLOW_ROOT -u REF \
      -u BACKUP_KEEP -u BACKUP_DIR -u NEXT_DIST_DIR pm2 "$@"
}

api_spec() { echo "$APP_DIR/.venv/bin/python|-m uvicorn backend.main:app --host 127.0.0.1 --port $APP_PORT|$APP_DIR|none"; }
web_spec() { echo "$APP_DIR/node_modules/next/dist/bin/next|start --hostname 127.0.0.1 --port $WEB_PORT|$APP_DIR|$(command -v node)"; }

pm2_start_api() {   # ONE process, fork mode, loopback only - never -i / cluster mode (in-process job queue + SQLite)
  pm2_clean start "$APP_DIR/.venv/bin/python" --name "$API_NAME" --cwd "$APP_DIR" --interpreter none \
    --kill-timeout 30000 --exp-backoff-restart-delay 1000 --time \
    -- -m uvicorn backend.main:app --host 127.0.0.1 --port "$APP_PORT"
}

pm2_start_web() {
  pm2_clean start "$APP_DIR/node_modules/next/dist/bin/next" --name "$WEB_NAME" --cwd "$APP_DIR" \
    --interpreter "$(command -v node)" --kill-timeout 10000 --exp-backoff-restart-delay 1000 --time \
    -- start --hostname 127.0.0.1 --port "$WEB_PORT"
}

pm2_field() {   # $1 = name, $2 = status | spec ; "missing" when the app is not registered
  pm2 jlist 2>/dev/null | node -e '
    let s = ""; process.stdin.on("data", d => s += d).on("end", () => {
      let list = []; try { list = JSON.parse(s.slice(s.indexOf("["))) } catch (e) {}
      const p = list.find(x => x.name === process.argv[1]);
      if (!p) return console.log("missing");
      const e = p.pm2_env, args = Array.isArray(e.args) ? e.args.join(" ") : (e.args || "");
      console.log(process.argv[2] === "status" ? e.status
        : [e.pm_exec_path, args, e.pm_cwd, e.exec_interpreter].join("|")) })' "$1" "$2"
}

pm2_status() { pm2_field "$1" status; }

ensure_app() {   # $1 = name, $2 = start function, $3 = wanted spec. Restart = a short interruption.
  local status
  status="$(pm2_status "$1")"
  if [[ "$status" == "missing" ]]; then
    say "Starting $1"; "$2"
  elif [[ "$(pm2_field "$1" spec)" != "$3" ]]; then
    say "$1: start command changed (path, port or Node version) - registering it again"
    pm2 delete "$1" >/dev/null; "$2"
  elif [[ "$status" == "online" ]]; then
    say "Restarting $1 (one process: a few seconds without it, not a zero-downtime reload)"
    pm2_clean restart "$1" >/dev/null
  else
    say "$1 is '$status' - starting it again"
    pm2 delete "$1" >/dev/null; "$2"
  fi
}

wait_healthy() {   # the API loads YOLO + the ROI segmenter at start-up: allow up to 4 minutes
  local i
  for i in $(seq 1 120); do
    if curl -fsS -o /dev/null "http://127.0.0.1:$APP_PORT/api/health" 2>/dev/null \
       && curl -fsS -o /dev/null "http://127.0.0.1:$WEB_PORT/" 2>/dev/null; then
      say "Healthy: API http://127.0.0.1:$APP_PORT/api/health and dashboard http://127.0.0.1:$WEB_PORT/"
      curl -fsS -m 10 "http://127.0.0.1:$APP_PORT/api/health" | pyexec -c '
import json, sys
h = json.load(sys.stdin); m = h.get("model") or {}
print("    model", m.get("version"), "on", m.get("device"), "| model_error", h.get("model_error"),
      "| cameras", h.get("cameras"))
' 2>/dev/null || true
      return 0
    fi
    sleep 2
  done
  return 1
}

# ------------------------------------------------------------------------------------------ backups
backup_list() {   # complete backups made by these scripts, newest first (names start with a timestamp)
  local d
  shopt -s nullglob
  for d in "$BACKUP_DIR"/*/; do
    if [[ -f "$d/.yk-backup" && -f "$d/COMPLETE" ]]; then echo "${d%/}"; fi
  done | sort -r
  shopt -u nullglob
}

prune_backups() {   # keep the newest BACKUP_KEEP complete backups; only folders marked by these scripts
  local d i=0 all
  all="$(backup_list)"
  while IFS= read -r d; do
    [[ -n "$d" ]] || continue
    i=$((i + 1))
    if (( i > BACKUP_KEEP )) && [[ "$d" == "$BACKUP_DIR"/* && -f "$d/.yk-backup" ]]; then
      say "Removing old backup $(basename "$d") (BACKUP_KEEP=$BACKUP_KEEP)"
      rm -rf -- "$d"
    fi
  done <<< "$all"
  shopt -s nullglob
  for d in "$BACKUP_DIR"/*/; do
    if [[ -f "$d/.yk-backup" && ! -f "$d/COMPLETE" ]]; then
      warn "Incomplete backup kept for inspection (not counted, not removed): ${d%/}"
    fi
  done
  shopt -u nullglob
}

backup_data() {
  # Everything that is not in git: the storage folder (rsync; files unchanged since the previous backup are
  # hard links to it, so they take no extra space), every SQLite database (litter_monitor.db, auth.db, the
  # export ledger, ...: SQLite backup API = consistent copy while the API writes, then integrity_check),
  # the runtime config files and the env files. Reads application data only - never changes or deletes it.
  local stamp dest prev need=0 free rc=0 files=0 env_names="" f db rel db_lines="" cfg_names=""
  local -a dbs=()
  require_tools rsync
  stamp="$(date +%Y%m%d-%H%M%S)-$(git -C "$APP_DIR" rev-parse --short HEAD 2>/dev/null || echo nogit)"
  dest="$BACKUP_DIR/$stamp"
  mkdir -p "$BACKUP_DIR"; chmod 700 "$BACKUP_DIR"
  [[ ! -e "$dest" ]] || die "Backup folder $dest already exists"
  prev="$(backup_list)"; prev="${prev%%$'\n'*}"

  # SQLite databases: the configured ones + any *.db inside storage
  [[ -f "$DB_PATH" ]] && dbs+=("$DB_PATH")
  [[ -f "$AUTH_DB_PATH" && "$AUTH_DB_PATH" != "$DB_PATH" ]] && dbs+=("$AUTH_DB_PATH")
  if [[ -d "$STORAGE_DIR" ]]; then
    while IFS= read -r -d '' db; do
      [[ "$db" == "$DB_PATH" || "$db" == "$AUTH_DB_PATH" ]] || dbs+=("$db")
    done < <(find "$STORAGE_DIR" -type f -name '*.db' -print0)
  fi

  local -a rs=(-a --numeric-ids)
  if [[ -n "$prev" && -d "$prev/storage" ]]; then rs+=("--link-dest=$prev/storage"); fi
  for db in "${dbs[@]}"; do                            # live databases are copied with the backup API below
    if [[ "$db" == "$STORAGE_DIR"/* ]]; then
      rel="${db#"$STORAGE_DIR"/}"
      rs+=("--exclude=/$rel" "--exclude=/$rel-wal" "--exclude=/$rel-shm" "--exclude=/$rel-journal")
    fi
  done
  if [[ -d "$STORAGE_DIR" ]]; then                     # only new / changed files need space
    need=$(rsync "${rs[@]}" --dry-run --stats "$STORAGE_DIR/" "$dest/storage/" \
           | awk -F': ' '/^Total transferred file size/ {gsub(/[^0-9]/, "", $2); print $2}')
    need=$(( ${need:-0} / 1024 ))
  fi
  for db in "${dbs[@]}"; do need=$(( need + $(du -sk "$db" | cut -f1) )); done
  free=$(df -Pk "$BACKUP_DIR" | awk 'NR==2 {print $4}')
  (( free > need + 1048576 )) || die "Not enough disk for the backup: ~$((need / 1024)) MB needed + 1 GB margin, \
$((free / 1024)) MB free in $BACKUP_DIR. Free space, lower BACKUP_KEEP or set BACKUP_DIR; nothing was changed."

  mkdir -p "$dest"
  echo "Backup made by deploy/server-setup.sh or deploy/deploy.sh" > "$dest/.yk-backup"
  if [[ -d "$STORAGE_DIR" ]]; then
    say "Backing up $STORAGE_DIR (~$((need / 1024)) MB new or changed${prev:+ since $(basename "$prev")})"
    rsync "${rs[@]}" "$STORAGE_DIR/" "$dest/storage/" || rc=$?
    if (( rc == 24 )); then
      warn "Some files vanished while being copied (deleted by the app meanwhile) - they are not in the backup"
    elif (( rc != 0 )); then
      die "rsync failed (exit $rc): the backup $dest is INCOMPLETE. Nothing was changed; fix the cause and re-run."
    fi
    files=$(find "$dest/storage" -type f | wc -l)
  fi
  for db in "${dbs[@]}"; do
    if [[ "$db" == "$STORAGE_DIR"/* ]]; then rel="storage/${db#"$STORAGE_DIR"/}"; else rel="external/$(basename "$db")"; fi
    mkdir -p "$dest/database/$(dirname "$rel")"
    pyexec - "$db" "$dest/database/$rel" <<'PY' || die "The backup of database $db failed - nothing was changed."
import sqlite3, sys
src = sqlite3.connect(sys.argv[1], timeout=30)
dst = sqlite3.connect(sys.argv[2])
src.backup(dst)                          # consistent snapshot, including the WAL, while the API writes
result = dst.execute("PRAGMA integrity_check").fetchone()[0]
dst.close(); src.close()
if result != "ok":
    sys.exit(f"integrity_check of the copy: {result}")
PY
    db_lines+="                $db -> database/$rel (integrity_check ok)"$'\n'
  done
  mkdir -p "$dest/config"
  for f in config/camera_roi.json config/store_config.json; do     # runtime / per-server configuration
    if [[ -f "$APP_DIR/$f" ]]; then cp -p "$APP_DIR/$f" "$dest/config/"; cfg_names+="$f "; fi
  done
  for f in .env .env.local; do
    if [[ -f "$APP_DIR/$f" ]]; then cp -p "$APP_DIR/$f" "$dest/config/$f"; env_names+="$f "; fi
  done
  {
    echo "created       $(date -Is)"
    echo "app commit    $(git -C "$APP_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "storage       $STORAGE_DIR -> storage/ ($files files; unchanged files hard-linked to ${prev:-nothing})"
    echo "databases"
    if [[ -n "$db_lines" ]]; then printf '%s' "$db_lines"; else echo "                none (no database yet)"; fi
    echo "config        ${cfg_names:-none }-> config/"
    echo "env files     ${env_names:-none }-> config/"
    echo "restore       pm2 stop both apps; rsync -a storage/ <storage>/ ; copy database/storage/* back into"
    echo "              <storage>/ (database/external/* to its LM_* path); copy config/* back; pm2 start both apps."
    echo "              Nothing is restored automatically."
  } > "$dest/MANIFEST.txt"
  touch "$dest/COMPLETE"
  chmod -R go-rwx "$dest"
  LAST_BACKUP="$dest"
  say "Backup complete: $dest"
  sed 's/^/    /' "$dest/MANIFEST.txt"
  prune_backups
}
