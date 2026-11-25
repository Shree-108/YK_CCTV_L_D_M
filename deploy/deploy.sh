#!/usr/bin/env bash
#
# deploy.sh - release a commit of BRANCH on a server prepared by deploy/server-setup.sh.
# Run it by hand for every release; nothing here runs automatically.
#
# Safety checks - the deploy STOPS (nothing changed) when one fails:
#   - the environment from setup is missing (.venv, Node >= 20.9, pm2, rsync) or .venv has another Python
#   - the checkout's origin is not REPO_URL, the checkout is not on BRANCH, or REF is not a commit of origin/BRANCH
#   - tracked files were edited on this server (FORCE_RESET=1 discards them; a patch is saved first)
#   - the release changes models/ (ALLOW_MODEL_CHANGE=1 accepts)
#   - the production model file does not match its SHA-256 (models are never changed by this script)
#   - an untracked file would be replaced, or runtime data (storage, .env, .env.local) is tracked by git
#   - .env problems (missing, no ADMIN_API_TOKEN, CORS_ORIGINS without the dashboard, S3 on without keys)
#   - jobs are queued or running (FORCE_RESTART=1 accepts: they are interrupted)
#   - a running app does not answer its health check (ALLOW_UNHEALTHY=1 accepts)
#   - the backup cannot be made completely
# Then: backup -> git reset --hard REF -> pinned dependencies -> import / CPU / model checks -> ROI model
#   -> npm ci + build -> restart both apps -> pm2 save -> health check -> last logs -> pm2 list.
#
# Restart, not zero-downtime: the API is ONE process (in-process job queue + SQLite + background S3 uploads),
# so it is restarted - a few seconds without API (plus model loading). `npm run build` replaces the files the
# running dashboard serves, so the dashboard can show errors until it restarts. Deploy outside store hours.
#
# Usage (as the deploy user):
#   bash deploy/deploy.sh
#   REF=<commit> bash deploy/deploy.sh      # roll the CODE back to an earlier commit of BRANCH
#
# Configuration (environment variables, defaults in brackets) - same names and defaults as server-setup.sh:
#   APP_DIR REPO_URL BRANCH [main] PM2_APP_NAME [yk-cctv-ldm] APP_PORT [8000] WEB_PORT [3000] PYTHON_BIN
#   TORCH_VARIANT [cpu] INSTALL_DEV_DEPS [0] MODEL_VERSION MODEL_SUMS BACKUP_DIR BACKUP_KEEP [5]
#   REF                 commit to deploy; must be on origin/BRANCH [origin/$BRANCH]
#   FORCE_RESET         1 = discard tracked-file edits made on this server (saved as a patch first) [0]
#   ALLOW_MODEL_CHANGE  1 = accept a release that changes models/ [0]
#   FORCE_RESTART       1 = restart even while jobs are queued or running [0]
#   ALLOW_UNHEALTHY     1 = deploy although a running app does not answer its health check [0]
#   ALLOW_ROOT          1 = allow running as root [0]
#
set -euo pipefail
set -o errtrace

# ------------------------------------------------------------------------------------------- config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(realpath -m "${APP_DIR:-$(dirname "$SCRIPT_DIR")}")"
REPO_URL="${REPO_URL:-https://github.com/Shree-108/YK_CCTV_L_D_M.git}"
BRANCH="${BRANCH:-main}"
REF="${REF:-origin/$BRANCH}"
PM2_APP_NAME="${PM2_APP_NAME:-yk-cctv-ldm}"
APP_PORT="${APP_PORT:-8000}"
WEB_PORT="${WEB_PORT:-3000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_VARIANT="${TORCH_VARIANT:-cpu}"
INSTALL_DEV_DEPS="${INSTALL_DEV_DEPS:-0}"
MODEL_VERSION="${MODEL_VERSION:-v3_store_cctv}"
MODEL_SUMS="${MODEL_SUMS:-models/backups/v3_store_cctv_20261012/SHA256SUMS}"
BACKUP_DIR="$(realpath -m "${BACKUP_DIR:-$HOME/yk-cctv-ldm-backups}")"
BACKUP_KEEP="${BACKUP_KEEP:-5}"
FORCE_RESET="${FORCE_RESET:-0}"
ALLOW_MODEL_CHANGE="${ALLOW_MODEL_CHANGE:-0}"
FORCE_RESTART="${FORCE_RESTART:-0}"
ALLOW_UNHEALTHY="${ALLOW_UNHEALTHY:-0}"
ALLOW_ROOT="${ALLOW_ROOT:-0}"

API_NAME="$PM2_APP_NAME-api"
WEB_NAME="$PM2_APP_NAME-web"
NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"
PREV_COMMIT=""

rollback_hint() {
  echo "    The code is now at $(git -C "$APP_DIR" rev-parse --short HEAD); before this deploy it was ${PREV_COMMIT:0:7}." >&2
  echo "    Roll the code back:  REF=$PREV_COMMIT bash deploy/deploy.sh" >&2
  echo "    Data backup of this run: ${LAST_BACKUP:-none} (restore steps in its MANIFEST.txt)" >&2
}
die() { echo "ERROR: $*" >&2; if [[ -n "$PREV_COMMIT" ]]; then rollback_hint; fi; exit 1; }
trap 'echo "ERROR: \"${BASH_COMMAND}\" failed (line ${LINENO})." >&2; if [[ -n "$PREV_COMMIT" ]]; then rollback_hint; fi' ERR
[[ -f "$SCRIPT_DIR/common.sh" ]] || die "$SCRIPT_DIR/common.sh is missing."
# shellcheck source=deploy/common.sh
. "$SCRIPT_DIR/common.sh"

no_tracked_runtime_data() {
  local tracked rel_storage=""
  if [[ "$STORAGE_DIR" == "$APP_DIR"/* ]]; then rel_storage="${STORAGE_DIR#"$APP_DIR"/}"; fi
  tracked="$(git ls-files -- ${rel_storage:+"$rel_storage"} .env .env.local config/camera_roi.json)"
  [[ -z "$tracked" ]] || die "Runtime data is tracked by git and would be overwritten by the reset:
$tracked
  Remove it from git in the repository (git rm --cached) before deploying."
}

review_reset() {
  local edited added f collide=()
  edited="$(git status --porcelain --untracked-files=no)"
  if [[ -n "$edited" ]]; then
    warn "Tracked files were edited on this server; git reset --hard would discard these changes:"
    echo "$edited" >&2
    [[ "$FORCE_RESET" == "1" ]] || die "Not deploying. Review them (git diff), keep what is needed in the repository,
  then re-run - or re-run with FORCE_RESET=1 to discard them (a patch is saved in the backup first)."
  fi
  if git diff --name-only HEAD "$REF" -- models | grep -q .; then
    warn "This release changes model files:"
    git diff --stat HEAD "$REF" -- models >&2
    [[ "$ALLOW_MODEL_CHANGE" == "1" ]] || die "Not deploying. Review the change (git diff HEAD $REF -- models);
  re-run with ALLOW_MODEL_CHANGE=1 to accept it."
  fi
  if git diff --name-only HEAD "$REF" -- config/store_config.json | grep -q .; then
    warn "This release changes config/store_config.json (store / cart mapping used for S3 folders and CSV exports)."
  fi
  added="$(git diff --name-only --diff-filter=A HEAD "$REF")"
  while IFS= read -r f; do
    if [[ -n "$f" && -e "$f" ]] && ! git ls-files --error-unmatch -- "$f" >/dev/null 2>&1; then collide+=("$f"); fi
  done <<< "$added"
  (( ${#collide[@]} == 0 )) || die "These untracked files would be replaced by files of the release:
$(printf '    %s\n' "${collide[@]}")
  Move them elsewhere first."
}

# ---------------------------------------------------------------------------------------------- main
[[ "$EUID" -ne 0 || "$ALLOW_ROOT" == "1" ]] || die "Run as the deploy user, not root (ALLOW_ROOT=1 overrides)."
check_settings
[[ -d "$APP_DIR/.git" ]] || die "$APP_DIR is not a git checkout. Run deploy/server-setup.sh first."
cd "$APP_DIR"
[[ -x .venv/bin/python ]] || die "$APP_DIR/.venv is missing. Run deploy/server-setup.sh first."
python_ok .venv/bin/python || die ".venv uses $(.venv/bin/python --version 2>&1); Python 3.12 - 3.14 is needed. Run deploy/server-setup.sh."
have="$(.venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
want="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "$have")"
[[ "$have" == "$want" ]] || die ".venv was built with Python $have but $PYTHON_BIN is now $want. Run deploy/server-setup.sh (it rebuilds .venv)."
load_node
node_ok || die "Node.js >= 20.9 not found (nvm in $NVM_DIR?). Run deploy/server-setup.sh first."
require_pm2
require_tools rsync curl git sha256sum

say "Checking repository, branch and commit"
[[ "$(repo_id "$(git remote get-url origin)")" == "$(repo_id "$REPO_URL")" ]] \
  || die "The checkout's origin is $(git remote get-url origin), not $REPO_URL. Check APP_DIR / REPO_URL."
current="$(git rev-parse --abbrev-ref HEAD)"
[[ "$current" == "$BRANCH" ]] || die "The checkout is on '$current', not '$BRANCH'. Switch with: git checkout $BRANCH (or set BRANCH)."
git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null \
  || die "Branch $BRANCH does not exist on origin ($(git remote get-url origin)) or origin is not reachable."
git fetch origin "$BRANCH"
git rev-parse --verify --quiet "$REF^{commit}" >/dev/null || die "REF=$REF is not a known commit."
git merge-base --is-ancestor "$REF^{commit}" "origin/$BRANCH" \
  || die "REF=$REF is not a commit of origin/$BRANCH - only commits of the deployment branch are deployed."
target="$(git rev-parse "$REF^{commit}")"
echo "    origin   $(git remote get-url origin)"
echo "    branch   $BRANCH"
echo "    current  $(git log -1 --format='%h %s' HEAD)"
echo "    target   $(git log -1 --format='%h %s' "$target")"
if [[ "$(git rev-parse HEAD)" == "$target" ]]; then
  say "Already at the target commit - rebuilding and restarting anyway"
elif git merge-base --is-ancestor HEAD "$target"; then
  echo "    new commits:"; git log --format='      %h %s' "HEAD..$target" | head -n 30
else
  warn "The target is not ahead of the current commit (roll-back or rewritten history): these commits go away:"
  git log --format='      %h %s' "$target..HEAD" | head -n 30 >&2
fi

say "Running the safety checks"
resolve_paths
no_tracked_runtime_data
review_reset
verify_model                       # the model in use now matches its checksum (nothing changed it unnoticed)
check_env strict
check_no_active_jobs
check_running_health

backup_data
if [[ "$FORCE_RESET" == "1" && -n "$(git status --porcelain --untracked-files=no)" ]]; then
  git diff HEAD > "$LAST_BACKUP/local-changes.patch"
  chmod go-rwx "$LAST_BACKUP/local-changes.patch"
  say "Saved the discarded server edits as local-changes.patch in the backup"
fi

PREV_COMMIT="$(git rev-parse HEAD)"
say "Resetting the code to $(git rev-parse --short "$target") (git reset --hard; untracked and ignored files stay)"
git reset --hard "$target"

ensure_runtime_dirs
say "Updating backend dependencies (pinned: deploy/constraints.txt, TORCH_VARIANT=$TORCH_VARIANT)"
install_python_deps
verify_python_stack
ensure_roi_model
build_frontend

ensure_app "$API_NAME" pm2_start_api "$(api_spec)"
ensure_app "$WEB_NAME" pm2_start_web "$(web_spec)"
pm2 save

if ! wait_healthy; then
  pm2 logs "$API_NAME" --lines 40 --nostream || true
  pm2 logs "$WEB_NAME" --lines 40 --nostream || true
  die "The apps did not become healthy within 4 minutes - see the logs above."
fi

say "Last 20 log lines"
pm2 logs "$API_NAME" --lines 20 --nostream || true
pm2 logs "$WEB_NAME" --lines 20 --nostream || true
say "Deployed $(git rev-parse --short HEAD) of $BRANCH (previous: ${PREV_COMMIT:0:7}; backup: $LAST_BACKUP)"
echo "    Roll the code back: REF=$PREV_COMMIT bash deploy/deploy.sh"
say "pm2 apps"
pm2 list
