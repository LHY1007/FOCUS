#!/bin/sh
set -eu
cd "$(git rev-parse --show-toplevel)"
chmod +x hooks/pre-commit
git config --local core.hooksPath hooks
printf '%s\n' 'Enabled the repository whitespace check in hooks/pre-commit.'
