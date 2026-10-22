sudo cp deploy/avp.service /etc/systemd/system/avp.service
sudo systemctl daemon-reload && sudo systemctl enable --now avp.service
sudo systemctl restart avp.service
journalctl -u avp.service -f











# deploy

Runs AVP as a systemd service that watches `Root_Address/Video_Addresses` (RobotScientistClips on the Samba share) for uploaded videos. Processing happens in `Temporary_Processing.address` (outside the share); only CSVs, logs and `.done` markers are mirrored to `Root_Address/Output_Address`. A folder is queued once its video has not been written for `Daemon.stability_seconds` (config.yaml, default 60 s); queued folders are processed one at a time, oldest upload first. The input folder is rescanned every `Daemon.poll_interval` seconds.

[avp.service](avp.service) starts, as user `roboprocessing`:

```bash
conda activate torch3.11
python main.py --daemon
```

in `/home/roboprocessing/Desktop/AutomaticVideoProcessor`, after `/run/media/roboprocessing/Data` is mounted. It restarts on failure after 10 s.

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

On stop, AVP finishes the current folder before exiting (`SIGTERM`). systemd kills it after `TimeoutStopSec=30`, so a long folder can be interrupted. The interrupted view has no `.done`, so on the next start its partial outputs are deleted and it is processed again.

Edit the user, paths or conda environment in `avp.service` for another machine, then copy it again and run `daemon-reload`.