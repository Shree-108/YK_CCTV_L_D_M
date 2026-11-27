#!/usr/bin/env bash
#
# server-setup.sh - ONE-TIME setup of the Yatrikart CCTV Litter Detection app (LDM) on an Ubuntu server
# (EC2, Ubuntu 24.04 recommended). Safe to re-run: every step checks what is already there first, and a re-run
# restarts the apps so they serve what it rebuilt. Releases after the first one: deploy/deploy.sh.
#
# What it does
#   1. Checks the machine: CPU, RAM, free disk, GPU (CPU-only unless TORCH_VARIANT=cuda).
#   2. Installs the system packages (apt) and Node.js >= 20.9 via nvm (Ubuntu's nodejs is too old for Next.js 16).
#   3. Installs pm2 for this user (INSTALL_PM2=0: only check and print how to).
#   4. Uses the checkout this script is in (must point at REPO_URL); fast-forwards BRANCH. Before pulling an
#      update it checks for active jobs and backs up the data.
#   5. Creates .venv (rebuilt if made with another Python) and installs the backend packages at the versions pinned
#      in deploy/constraints.txt, PyTorch from the CPU-only index; `npm ci` + `npm run build`.
#   6. Verifies the packages import, torch is a CPU build, and models/litter_yolo/v3_store_cctv/best.pt matches the
#      SHA-256 recorded in git (read-only). Downloads the monitored-region segmenter once (unless LDM_ROI=off).
#   7. Creates the runtime folders where the app keeps them (LM_STORAGE_DIR / LM_DB_PATH), never deleting any.
#   8. Checks .env / .env.local (key names only, values are never printed).
#   9. Registers both pm2 apps or restarts them, `pm2 save`, health check.
#  10. Makes pm2 start on boot (systemd service pm2-<user>, via `pm2 startup`; needs sudo).
#
# Never touches application data, never overwrites model weights, never opens ports, never configures Nginx.
# Both apps listen on 127.0.0.1 only; the public site is the Nginx reverse proxy in deploy/nginx/yk-cctv.conf
# (http://<server>/ldm and /ldm-api), protected by the app's login (LM_AUTH_MODE=on).
#
# Usage (as the normal deploy user, NOT root - sudo is used only for apt and the boot service):
#   git clone https://github.com/Shree-108/YK_CCTV_L_D_M.git ~/yk-cctv-ldm && cd ~/yk-cctv-ldm
#   bash deploy/server-setup.sh
#
# Configuration (environment variables, defaults in brackets)
#   APP_DIR            project checkout [the parent of this deploy/ folder]
#   REPO_URL           repository the checkout's origin must be [https://github.com/Shree-108/YK_CCTV_L_D_M.git]
#                      Private repository: SSH deploy key (git@github.com:...). Never put a token in the URL.
#   BRANCH             branch to deploy [main]
#   PM2_APP_NAME       pm2 name prefix: <name>-api and <name>-web [yk-cctv-ldm]
#   APP_PORT           FastAPI port, bound to 127.0.0.1 [8000]
#   WEB_PORT           Next.js dashboard port, bound to 127.0.0.1 [3000] (the uniform app uses 3010 / 8010)
#   PYTHON_BIN         Python interpreter for the venv, 3.12 - 3.14 [python3]
#   NODE_VERSION       Node.js version installed with nvm when the system one is too old [22]
#   NVM_VERSION        nvm release used to install Node [v0.40.3]
#   TORCH_VARIANT      cpu = CPU-only PyTorch wheels | cuda = default PyPI wheels (GPU servers) [cpu]
#   INSTALL_DEV_DEPS   1 = also install pytest, httpx, scikit-learn, joblib [0]
#   MODEL_VERSION      production model folder under models/litter_yolo [v3_store_cctv]
#   MODEL_SUMS         checksum file for its best.pt [models/backups/v3_store_cctv_20261012/SHA256SUMS]
#   MIN_FREE_DISK_GB   stop if less free disk than this on APP_DIR's filesystem [15]
#   BACKUP_DIR         where backups go - outside APP_DIR and storage [$HOME/yk-cctv-ldm-backups]
#   BACKUP_KEEP        complete backups to keep; older ones made by these scripts are removed [5]
#   ALLOW_MODEL_CHANGE 1 = accept an update that changes models/ [0]
#   FORCE_RESTART      1 = restart the apps even while jobs are queued or running [0]
#   ALLOW_UNHEALTHY    1 = continue although a running app does not answer its health check [0]
#   ALLOW_ROOT         1 = allow running as root (not recommended: pm2/nvm become root's) [0]
#   INSTALL_PM2        1 = install pm2 when missing | 0 = fail with instructions [1]
#   PM2_VERSION        pm2 version installed when missing [latest]
#   PM2_STARTUP        1 = register pm2 as a systemd boot service (sudo) | 0 = only print how to [1]
#
set -euo pipefail
set -o errtrace

# ------------------------------------------------------------------------------------------- config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(realpath -m "${APP_DIR:-$(dirname "$SCRIPT_DIR")}")"
REPO_URL="${REPO_URL:-https://github.com/Shree-108/YK_CCTV_L_D_M.git}"
BRANCH="${BRANCH:-main}"
PM2_APP_NAME="${PM2_APP_NAME:-yk-cctv-ldm}"
APP_PORT="${APP_PORT:-8000}"
WEB_PORT="${WEB_PORT:-3000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NODE_VERSION="${NODE_VERSION:-22}"
NVM_VERSION="${NVM_VERSION:-v0.40.3}"
TORCH_VARIANT="${TORCH_VARIANT:-cpu}"
INSTALL_DEV_DEPS="${INSTALL_DEV_DEPS:-0}"
MODEL_VERSION="${MODEL_VERSION:-v3_store_cctv}"
MODEL_SUMS="${MODEL_SUMS:-models/backups/v3_store_cctv_20261012/SHA256SUMS}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-15}"
BACKUP_DIR="$(realpath -m "${BACKUP_DIR:-$HOME/yk-cctv-ldm-backups}")"
BACKUP_KEEP="${BACKUP_KEEP:-5}"
ALLOW_MODEL_CHANGE="${ALLOW_MODEL_CHANGE:-0}"
FORCE_RESTART="${FORCE_RESTART:-0}"
ALLOW_UNHEALTHY="${ALLOW_UNHEALTHY:-0}"
ALLOW_ROOT="${ALLOW_ROOT:-0}"
INSTALL_PM2="${INSTALL_PM2:-1}"
PM2_VERSION="${PM2_VERSION:-latest}"
PM2_STARTUP="${PM2_STARTUP:-1}"

API_NAME="$PM2_APP_NAME-api"
WEB_NAME="$PM2_APP_NAME-web"
NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"
# libgl1 / libglib2.0-0: OpenCV; ffmpeg: video decoding of uploaded / NVR clips
APT_PACKAGES=(git curl ca-certificates rsync build-essential python3 python3-venv python3-dev libgl1 libglib2.0-0 ffmpeg)

die() { echo "ERROR: $*" >&2; exit 1; }
trap 'echo "ERROR: \"${BASH_COMMAND}\" failed (line ${LINENO}). Fix the cause above and re-run - every step is safe to repeat." >&2' ERR
[[ -f "$SCRIPT_DIR/common.sh" ]] || die "$SCRIPT_DIR/common.sh is missing. Run this script from a checkout of the project."
# shellcheck source=deploy/common.sh
. "$SCRIPT_DIR/common.sh"

# ------------------------------------------------------------------------------------ setup-only steps
check_machine() {
  say "Checking the machine"
  local cpus mem_mb free_gb itype="" token=""
  cpus=$(nproc); mem_mb=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
  free_gb=$(df -Pk "$APP_DIR" | awk 'NR==2 {print int($4/1048576)}')
  token=$(curl -fsS -m 2 -X PUT "http://169.254.169.254/latest/api/token" \
            -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null || true)
  if [[ -n "$token" ]]; then
    itype=$(curl -fsS -m 2 -H "X-aws-ec2-metadata-token: $token" \
            "http://169.254.169.254/latest/meta-data/instance-type" 2>/dev/null || true)
  fi
  echo "    ${itype:+EC2 $itype | }$cpus CPU | ${mem_mb} MB RAM | ${free_gb} GB free on $APP_DIR | $(. /etc/os-release && echo "$PRETTY_NAME")"
  if [[ -e /proc/driver/nvidia/version ]] || command -v nvidia-smi >/dev/null 2>&1; then
    echo "    NVIDIA GPU driver found. TORCH_VARIANT=$TORCH_VARIANT$([[ $TORCH_VARIANT == cpu ]] && echo ' (the GPU will NOT be used; set TORCH_VARIANT=cuda to use it)')"
  else
    echo "    No NVIDIA GPU driver found - CPU-only inference."
    [[ "$TORCH_VARIANT" == "cpu" ]] || die "TORCH_VARIANT=$TORCH_VARIANT but this machine has no NVIDIA driver. Use TORCH_VARIANT=cpu."
  fi
  (( free_gb >= MIN_FREE_DISK_GB )) || die "Only ${free_gb} GB free; at least ${MIN_FREE_DISK_GB} GB are needed \
(Python packages ~4 GB incl. transformers, node_modules + build ~1 GB, ROI model, plus videos, screenshots and backups)."
  (( mem_mb >= 3500 )) || warn "${mem_mb} MB RAM: 'npm run build', YOLO and the ROI segmenter may run out of memory. Add swap \
(sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile) or use a larger instance."
}

install_system_packages() {
  local missing=() p
  for p in "${APT_PACKAGES[@]}"; do
    dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
  done
  if (( ${#missing[@]} )); then
    say "Installing system packages: ${missing[*]}"
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "${missing[@]}"
  else
    say "System packages already installed"
  fi
}

install_node() {
  load_node
  if node_ok; then say "Node.js $(node --version) is new enough (>= 20.9)"; return 0; fi
  say "Node.js >= 20.9 is needed: installing Node $NODE_VERSION with nvm $NVM_VERSION"
  if [[ ! -s "$NVM_DIR/nvm.sh" ]]; then
    curl -fsSL "https://raw.githubusercontent.com/nvm-sh/nvm/$NVM_VERSION/install.sh" | PROFILE=/dev/null bash
  fi
  set +u; # shellcheck disable=SC1091
  . "$NVM_DIR/nvm.sh"; nvm install "$NODE_VERSION"; nvm alias default "$NODE_VERSION"; nvm use --silent default; set -u
  node_ok || die "Node.js installation failed (node --version: $(node --version 2>/dev/null || echo none))"
  if ! grep -qs 'NVM_DIR' "$HOME/.bashrc"; then
    say "Adding nvm to ~/.bashrc so new SSH sessions find node and pm2"
    printf '\nexport NVM_DIR="%s"\n[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"\n' "$NVM_DIR" >> "$HOME/.bashrc"
  fi
}

install_pm2() {   # into the active (nvm) Node, so no sudo
  if command -v pm2 >/dev/null 2>&1; then say "pm2 $(pm2 --version 2>/dev/null) is installed"; return 0; fi
  [[ "$INSTALL_PM2" == "1" ]] || return 0
  say "Installing pm2@$PM2_VERSION for $(id -un) (npm install -g, into Node $(node --version))"
  npm install -g --no-fund --no-audit "pm2@$PM2_VERSION"
  hash -r
}

pm2_boot_service() {   # systemd unit pm2-<user> that resurrects the `pm2 save`d apps after a reboot
  local user unit node_dir
  user="$(id -un)"; unit="/etc/systemd/system/pm2-$user.service"; node_dir="$(dirname "$(command -v node)")"
  if systemctl is-enabled --quiet "pm2-$user" 2>/dev/null && grep -qF "$node_dir" "$unit" 2>/dev/null; then
    say "pm2 boot service pm2-$user is already enabled"; return 0
  fi
  if [[ "$PM2_STARTUP" != "1" ]]; then
    warn "pm2 does not start on boot yet. Run:  pm2 startup   and execute the sudo command it prints, then  pm2 save"
    return 0
  fi
  say "Registering pm2 as the systemd boot service pm2-$user (sudo)"
  sudo env PATH="$PATH:$node_dir" "$(command -v pm2)" startup systemd -u "$user" --hp "$HOME" >/dev/null \
    || die "pm2 startup failed. Run it by hand: pm2 startup   (then execute the sudo command it prints)"
  systemctl is-enabled --quiet "pm2-$user" || die "pm2-$user.service was not enabled. Check: systemctl status pm2-$user"
}

check_python() {
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "$PYTHON_BIN not found. Install it (sudo apt-get install python3 python3-venv)."
  python_ok "$PYTHON_BIN" || die "$PYTHON_BIN is $("$PYTHON_BIN" --version 2>&1). The pinned packages (deploy/constraints.txt)
  need Python 3.12 - 3.14. Use Ubuntu 24.04 (python3 = 3.12) or set PYTHON_BIN=python3.12."
  say "Python: $("$PYTHON_BIN" --version 2>&1)"
}

checkout_code() {
  [[ -d "$APP_DIR/.git" ]] || die "$APP_DIR is not a git checkout. Clone first:
    git clone --branch $BRANCH $REPO_URL <folder> && cd <folder> && bash deploy/server-setup.sh"
  if [[ "$REPO_URL" =~ ^https?://[^/]*@ ]]; then
    die "REPO_URL contains credentials. Use an SSH deploy key (git@github.com:...) instead of a token in the URL."
  fi
  cd "$APP_DIR"
  say "Using the checkout in $APP_DIR"
  [[ "$(repo_id "$(git remote get-url origin)")" == "$(repo_id "$REPO_URL")" ]] \
    || die "The checkout's origin is $(git remote get-url origin), not $REPO_URL. Not changing remotes - fix it by hand
  (git remote set-url origin <url>) or set REPO_URL."
  local changed
  changed="$(git status --porcelain --untracked-files=no)"
  [[ -z "$changed" ]] || die "Tracked files were changed on this server - not pulling over them:
$changed
  Review them (git diff). Keep a copy of anything needed, then: git stash   (or git checkout -- <file>)."
  git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null \
    || die "Branch $BRANCH does not exist on origin ($(git remote get-url origin)) or origin is not reachable."
  git fetch origin "$BRANCH"
  if git show-ref --verify --quiet "refs/heads/$BRANCH"; then git checkout -q "$BRANCH"
  else git checkout -q -b "$BRANCH" --track "origin/$BRANCH"; fi
  git merge-base --is-ancestor HEAD "origin/$BRANCH" \
    || die "Local $BRANCH has commits that are not on origin. Resolve manually (git log origin/$BRANCH..$BRANCH)."
  if [[ "$(git rev-parse HEAD)" != "$(git rev-parse "origin/$BRANCH")" ]]; then
    if git diff --name-only HEAD "origin/$BRANCH" -- models | grep -q .; then
      warn "This update changes model files:"
      git diff --stat HEAD "origin/$BRANCH" -- models >&2
      [[ "$ALLOW_MODEL_CHANGE" == "1" ]] || die "Not updating. Review the change; re-run with ALLOW_MODEL_CHANGE=1 to accept it."
    fi
    resolve_paths
    check_no_active_jobs
    backup_data
    say "Updating $(git rev-parse --short HEAD) -> $(git rev-parse --short "origin/$BRANCH")"
    git pull --ff-only origin "$BRANCH"
  fi
}

setup_venv() {
  local want have=""
  want="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  if [[ -x "$APP_DIR/.venv/bin/python" ]]; then
    have="$("$APP_DIR/.venv/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
  fi
  if [[ -n "$have" && "$have" != "$want" ]] || [[ -d "$APP_DIR/.venv" && -z "$have" ]]; then
    say "Rebuilding .venv (built with Python ${have:-unknown}, now $want)"
    rm -rf -- "$APP_DIR/.venv"
  fi
  if [[ ! -x "$APP_DIR/.venv/bin/python" ]]; then
    say "Creating .venv with Python $want"
    "$PYTHON_BIN" -m venv "$APP_DIR/.venv"
  fi
  say "Installing backend dependencies into .venv (pinned: deploy/constraints.txt, TORCH_VARIANT=$TORCH_VARIANT)"
  install_python_deps
}

# ---------------------------------------------------------------------------------------------- main
[[ "$EUID" -ne 0 || "$ALLOW_ROOT" == "1" ]] || die "Run as the normal deploy user (e.g. ubuntu), not root; sudo is used where needed. (ALLOW_ROOT=1 overrides.)"
check_settings

check_machine
install_system_packages
check_python
install_node
install_pm2
require_pm2
checkout_code
cd "$APP_DIR"
resolve_paths
ensure_runtime_dirs
setup_venv
verify_python_stack
ensure_roi_model
check_env warn

registered=0
if [[ "$(pm2_status "$API_NAME")" != "missing" || "$(pm2_status "$WEB_NAME")" != "missing" ]]; then
  registered=1
  check_no_active_jobs          # a re-run restarts the apps after rebuilding them
  check_running_health
fi
build_frontend

ensure_app "$API_NAME" pm2_start_api "$(api_spec)"
ensure_app "$WEB_NAME" pm2_start_web "$(web_spec)"
pm2 save
if ! wait_healthy; then
  pm2 logs "$API_NAME" --lines 30 --nostream || true
  pm2 logs "$WEB_NAME" --lines 30 --nostream || true
  die "The apps did not become healthy within 4 minutes - see the logs above (pm2 logs $API_NAME)."
fi
pm2_boot_service

say "Setup complete$([[ $registered == 1 ]] && echo ' (apps restarted with the rebuilt code)')"
cat <<EOF
    App dir      : $APP_DIR  (branch $BRANCH @ $(git -C "$APP_DIR" rev-parse --short HEAD))
    API          : $API_NAME  ->  http://127.0.0.1:$APP_PORT  (one process; public at /ldm-api via Nginx)
    Dashboard    : $WEB_NAME  ->  http://127.0.0.1:$WEB_PORT/ldm  (public at /ldm via Nginx)
    Data         : $STORAGE_DIR  (database $DB_PATH)
    Runtimes     : $("$APP_DIR/.venv/bin/python" --version 2>&1), Node $(node --version), pm2 $(pm2 --version)
    Backups      : $BACKUP_DIR  (newest $BACKUP_KEEP kept)
    Boot service : pm2-$(id -un)  ($(systemctl is-enabled --quiet "pm2-$(id -un)" && echo enabled || echo 'not enabled - run pm2 startup'))
    Status/logs  : pm2 list   |   pm2 logs $API_NAME   |   pm2 logs $WEB_NAME

Public access: install the Nginx site once (deploy/nginx/yk-cctv.conf, steps in deploy/SERVER_DEPLOY.md), create the
admin login once:  .venv/bin/python -m backend.auth add-user admin --role admin
then browse to http://<server-ip>/ldm
Releases: bash deploy/deploy.sh
EOF
say "pm2 apps"
pm2 list
