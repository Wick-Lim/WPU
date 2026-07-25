#!/usr/bin/env sh
# Fetch the real GLM-5.2 tokenizer.json so the server uses the GLM BPE tokenizer
# (else it falls back to the byte tokenizer). ~20 MB, public repo. Placed next to
# this script so make_tokenizer() auto-detects it.
set -e
DEST="$(cd "$(dirname "$0")" && pwd)/tokenizer.json"
URL="https://huggingface.co/zai-org/GLM-5.2-FP8/resolve/main/tokenizer.json"
echo "fetching $URL -> $DEST"
curl -sL "$URL" -o "$DEST"
echo "done ($(wc -c < "$DEST") bytes). server now uses the GLM tokenizer; run: python3 host/wpu_server.py"
