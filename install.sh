# /bin/sh
python3 -m venv .venv
source .venv/bin/activate
if [[ $(uname -m) == 'arm64' ]]; then
  pip3 install -r requirements.txt
else
  ARCHFLAGS="-arch x86_64" pip3 install -r requirements.txt
fi