#!/usr/bin/env bash
# Install or update Twinkly Bridge on Debian 12/13 (or Ubuntu 22.04+).
#
#   From a checkout:  sudo ./deploy/install.sh
#   From anywhere:    curl -fsSL https://raw.githubusercontent.com/danieltrillo/twinkly-pixel-control/main/deploy/install.sh | sudo bash
#
# What it does: installs python3 + python3-yaml + python3-aiohttp (plus numpy,
# OpenCV and scipy for camera pixel mapping), copies the code to
# /opt/twinkly-bridge, creates /etc/twinkly-bridge/config.yaml from the example
# if there is none, and enables the systemd service. Mapping clips and results
# live in /var/lib/twinkly-bridge/mapping.

set -euo pipefail

REPO_URL="${TWINKLY_BRIDGE_REPO:-https://github.com/danieltrillo/twinkly-pixel-control.git}"
BRANCH="${TWINKLY_BRIDGE_BRANCH:-main}"
INSTALL_DIR=/opt/twinkly-bridge
CONFIG_DIR=/etc/twinkly-bridge
SERVICE=twinkly-bridge

if [[ $EUID -ne 0 ]]; then
  echo "please run as root (sudo)" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends python3 python3-yaml python3-aiohttp git ca-certificates >/dev/null
# Camera pixel mapping (the /map page) decodes videos with numpy, OpenCV and scipy. Skip with TWINKLY_BRIDGE_MAPPING=0.
if [[ "${TWINKLY_BRIDGE_MAPPING:-1}" != "0" ]]; then
  apt-get install -y -qq --no-install-recommends python3-numpy python3-opencv python3-scipy >/dev/null
fi

# Where is the source? A checkout next to this script, or a fresh clone.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
SRC=""
if [[ -n "$SCRIPT_DIR" && -d "$SCRIPT_DIR/../twinklybridge" ]]; then
  SRC="$(cd "$SCRIPT_DIR/.." && pwd)"
  echo "installing from checkout $SRC"
else
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  echo "cloning $REPO_URL ($BRANCH)"
  git clone --quiet --depth 1 --branch "$BRANCH" "$REPO_URL" "$TMP/src"
  SRC="$TMP/src"
fi

id -u $SERVICE >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin $SERVICE

mkdir -p "$INSTALL_DIR" "$CONFIG_DIR"
rm -rf "$INSTALL_DIR/twinklybridge"
cp -r "$SRC/twinklybridge" "$INSTALL_DIR/twinklybridge"
cp "$SRC/README.md" "$INSTALL_DIR/README.md" 2>/dev/null || true
find "$INSTALL_DIR" -name '__pycache__' -type d -prune -exec rm -rf {} +

if [[ ! -f "$CONFIG_DIR/config.yaml" ]]; then
  cp "$SRC/twinklybridge/example-config.yaml" "$CONFIG_DIR/config.yaml"
  echo "created $CONFIG_DIR/config.yaml from the example; edit it then: sudo systemctl reload twinkly-bridge"
fi
chown -R $SERVICE:$SERVICE "$CONFIG_DIR"
chmod 750 "$CONFIG_DIR"
chmod 640 "$CONFIG_DIR/config.yaml"

cat > /usr/local/bin/twinkly-bridge <<'EOF'
#!/bin/sh
PYTHONPATH=/opt/twinkly-bridge exec /usr/bin/python3 -m twinklybridge "$@"
EOF
chmod 755 /usr/local/bin/twinkly-bridge

cp "$SRC/deploy/twinkly-bridge.service" /etc/systemd/system/twinkly-bridge.service
systemctl daemon-reload
systemctl enable twinkly-bridge >/dev/null 2>&1
systemctl restart twinkly-bridge

sleep 1
if systemctl is-active --quiet twinkly-bridge; then
  PORT="$(python3 - <<'EOF'
import yaml
try:
    print(yaml.safe_load(open("/etc/twinkly-bridge/config.yaml")).get("web", {}).get("port", 8080))
except Exception:
    print(8080)
EOF
)"
  echo
  echo "Twinkly Bridge is running."
  echo "  config:  $CONFIG_DIR/config.yaml   (twinkly-bridge check -c $CONFIG_DIR/config.yaml)"
  echo "  logs:    journalctl -u twinkly-bridge -f"
  echo "  status:  http://<this-host>:$PORT"
  echo "  mapping: http://<this-host>:$PORT/map   (clips + results in /var/lib/twinkly-bridge/mapping)"
  echo "  this host's addresses:"
  ip -4 -brief addr | grep -v "^lo" | sed "s/^/    /"
else
  echo "Twinkly Bridge did not start. See: journalctl -u twinkly-bridge -n 50" >&2
  exit 1
fi
