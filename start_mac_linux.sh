#!/usr/bin/env bash
# Einmal ausfuehrbar machen:  chmod +x start_mac_linux.sh
set -e
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
  echo "== Lege virtuelle Umgebung an =="
  python3 -m venv venv
fi

source venv/bin/activate
python -m pip install --upgrade pip >/dev/null
pip install -r requirements.txt

python tools/check_setup.py || true
python main.py "$@"
