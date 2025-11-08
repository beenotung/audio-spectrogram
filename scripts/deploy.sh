#!/bin/bash
set -euo pipefail

npm run build

rm -rf public
mkdir public
cp index.html bundle.js public/

npx surge public https://audio-spectrogram.surge.sh

rm -rf public
