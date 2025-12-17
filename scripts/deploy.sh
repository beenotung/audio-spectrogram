#!/bin/bash
set -euo pipefail

npm run build
npx surge web
