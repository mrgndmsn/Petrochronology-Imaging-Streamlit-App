#!/bin/bash
cd "$(dirname "$0")" || exit 1
python_bin=""
for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(not ((3, 11) <= sys.version_info[:2] <= (3, 13)))' 2>/dev/null; then
        python_bin="$candidate"
        break
    fi
done
if [ -n "$python_bin" ]; then
    "$python_bin" launch.py
else
    echo "Install Python 3.13 from https://www.python.org/downloads/ and try again."
fi
read -r -p "Press Return to close this window. "
