# deploy

Runs AVP as a systemd service that keeps watching for new experiment folders.

[avp.service](avp.service) starts, as user `roboprocessing`:

```bash
conda activate torch3.13
python AVP.py --daemon --poll-interval 60 --stability-seconds 60
```

in `/home/roboprocessing/Desktop/AutomaticVideoProcessor`, after `/media/roboprocessing/Data` is mounted. It restarts on failure after 10 s.

## Install

```bash
sudo cp deploy/avp.service /etc/systemd/system/avp.service
sudo systemctl daemon-reload
sudo systemctl enable --now avp.service
```

## Manage

```bash
systemctl status avp.service
journalctl -u avp.service -f          # Live output
sudo systemctl restart avp.service    # After code or config changes
sudo systemctl stop avp.service
```

On stop, AVP finishes the current folder before exiting (`SIGTERM`). systemd kills it after `TimeoutStopSec=30`, so a long folder can be interrupted; it is reprocessed on the next start because `.done` is missing.

Edit the user, paths or conda environment in `avp.service` for another machine, then copy it again and run `daemon-reload`.