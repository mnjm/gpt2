#!/usr/bin/env bash

OUT_DIR="dataset/tinyshakespeare"
OUT_FILE="$OUT_DIR/raw.txt"
URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

mkdir -p "$OUT_DIR"

if command -v wget >/dev/null 2>&1; then
    wget -O "$OUT_FILE" "$URL"
elif command -v curl >/dev/null 2>&1; then
    curl -o "$OUT_FILE" "$URL"
else
    echo "Error: Neither wget nor curl is found."
    exit 1
fi

if [ -f "$OUT_FILE" ]; then
    echo "Download complete: $OUT_FILE"
else
    echo "Download failed."
    exit 1
fi