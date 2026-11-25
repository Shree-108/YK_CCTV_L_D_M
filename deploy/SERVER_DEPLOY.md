# Deploying LDM on a server (Ubuntu 24.04 + pm2)

These steps install the Litter Detection app on a Linux server (EC2). They do not cover the Windows demo on a
laptop; that is in `deploy/README.md` (`start-ldm.ps1`).

There are two scripts. Run both **on the server** as the `ubuntu` user, not as root:

| Script | When to run it | What it does |
|---|---|---|
| `deploy/server-setup.sh` | Once (it is safe to run again) | Installs system packages, Node 22 (nvm) and pm2. Creates `.venv` with pinned packages and CPU-only PyTorch. Checks the model checksum, downloads the ROI model, builds the dashboard, starts two pm2 apps and sets up the boot service. |
| `deploy/deploy.sh` | For every release | Runs safety checks and a backup, then `git reset --hard` to the new commit, rebuilds, restarts, prints the logs and runs `pm2 list`. |

Both scripts load their shared functions from `deploy/common.sh` and pin package versions with `deploy/constraints.txt`.

They create these pm2 apps:

| pm2 name | What it runs | Listens on |
|---|---|---|
| `yk-cctv-ldm-api` | FastAPI, as a single process | `127.0.0.1:8000` |
| `yk-cctv-ldm-web` | Next.js dashboard | `127.0.0.1:3000` |

The ports differ from the uniform app (8010 / 3010), so both apps can run on the same server.

> **Access: SSH tunnel only (for now).** The API is local-only by design (`backend/api/local_only.py`). It returns
> 403 to every request that does not come from the server itself, including requests forwarded by Nginx. This
> stays in place until login protection is enabled and that guard is lifted, which needs a code change. Until
> then, open the dashboard through an SSH tunnel (step 5). Do **not** open ports 8000 or 3000 in the security group.

## 1. Launch the instance

- **AMI:** Ubuntu Server 24.04 LTS. It ships Python 3.12, which the scripts require; 22.04 is refused.
- **Size:** at least 2 vCPU and **8 GB RAM** (e.g. `t3.large`). YOLO, the SegFormer ROI model and `next build`
  together need more than 4 GB. Disk: **40 GB+** gp3.
- **Region:** ap-south-1, the same region as the `yk-litter-detection-data` bucket.
- **Security group:** SSH (22) from your office IP only. Nothing else is needed.

## 2. Get the code onto the server

```bash
ssh -i <key.pem> ubuntu@<server-ip>
sudo apt-get update && sudo apt-get install -y git
git clone --branch main https://github.com/Shree-108/YK_CCTV_L_D_M.git ~/yk-cctv-ldm
cd ~/yk-cctv-ldm
```

If the repository is private, add an SSH deploy key on GitHub and clone `git@github.com:Shree-108/YK_CCTV_L_D_M.git`.
Then run the scripts with `REPO_URL=git@github.com:Shree-108/YK_CCTV_L_D_M.git`. Never put a token in the URL.

The server only gets what has been **committed and pushed** to `main`. Work that is still uncommitted on the
development PC is not deployed.

## 3. Create the settings file (before setup)

```bash
cp .env.example .env
nano .env          # fill in the values below
chmod 600 .env
```

Minimum for a server:

```ini
ADMIN_API_TOKEN=<long random value>        # e.g. from: openssl rand -hex 32
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
LM_START_CAMERAS=false                     # a cloud server cannot reach the store NVR
# S3 upload of results (optional)
LDM_S3_UPLOAD=on
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=ap-south-1
LDM_S3_BUCKET=yk-litter-detection-data
```

`NEXT_PUBLIC_API_URL` can stay unset. The dashboard then calls `http://localhost:8000`, which the SSH tunnel
forwards to the server. `config/store_config.json` comes from git. Fill in the real store and cart values in git,
not on the server; otherwise S3 uploads go to `cart_unassigned/`.

## 4. Run the setup

```bash
bash deploy/server-setup.sh
```

The first run takes 15–25 minutes: CPU PyTorch, transformers, npm and the Next.js build. It asks for your sudo
password for apt and the pm2 boot service. At the end it prints a summary, the loaded model and `pm2 list`, which
should show both apps `online`.

After that, use these commands in a **new** SSH session, so nvm puts `node` and `pm2` on the PATH:

```bash
pm2 list                          # both apps
pm2 logs yk-cctv-ldm-api          # live logs (Ctrl+C to leave)
pm2 restart yk-cctv-ldm-api       # manual restart
systemctl status pm2-ubuntu       # the boot service that restarts the apps after a reboot
curl -s http://127.0.0.1:8000/api/health
```

To change a setting, put it in front of the command, e.g. `APP_PORT=8001 WEB_PORT=3001 bash deploy/server-setup.sh`.
Other options include `INSTALL_PM2=0`, `PM2_STARTUP=0` and `TORCH_VARIANT=cuda` (GPU instance). The top of each
script lists every option.

## 5. Open the dashboard (SSH tunnel)

On your PC:

```bash
ssh -i <key.pem> -N -L 3000:127.0.0.1:3000 -L 8000:127.0.0.1:8000 ubuntu@<server-ip>
```

Leave that window open, then go to **http://localhost:3000**. The sidebar should show "Connected to backend".
Both ports are needed: the browser loads the dashboard from 3000 and calls the API on 8000.

If something already uses port 3000 or 8000 on your PC, choose other local ports, for example
`-L 3100:127.0.0.1:3000 -L 8100:127.0.0.1:8000`. In that case, before setup or deploy, add both of these to the
server's config:
- `http://localhost:3100` to `CORS_ORIGINS` in `.env`
- `NEXT_PUBLIC_API_URL=http://localhost:8100` in `.env.local`

## 6. Every release

1. On your PC, commit and push to `main`.
2. On the server, run:

```bash
cd ~/yk-cctv-ldm && bash deploy/deploy.sh
```

The deploy stops without changing anything if a check fails, and it says what to do. It checks for:
- jobs that are still running
- files edited on the server
- a release that changes `models/`
- a model whose checksum doesn't match
- `.env` problems
- an app that is unhealthy
- a backup that failed

A restart interrupts S3 uploads that are in progress. They run in memory and are not resumed. Deploy when no
video is being analysed.

To roll the code back, run the `REF=<commit> bash deploy/deploy.sh` line that the deploy prints. Backups are in
`~/yk-cctv-ldm-backups`, the newest 5 are kept. Each backup contains:
- `storage/`
- consistent copies of every SQLite database
- `config/camera_roi.json` and `store_config.json`
- `.env`
- a `MANIFEST.txt` with the restore steps

Restoring data is a manual step.

## What these scripts never do

- They don't open ports, configure Nginx or HTTPS, or expose the app beyond `127.0.0.1`.
- They don't replace, download or re-register the YOLO model. The only download is the ROI segmenter, when it is missing.
- They don't delete application data. Pruning only removes old backups that these scripts created themselves.
