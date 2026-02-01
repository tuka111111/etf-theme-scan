#!/bin/bash
set -e

# main 直書き禁止
[ "$GITHUB_REF_NAME" = "main" ] && exit 1

# decision_id / schema 保護
git diff | grep -E 'decision_id|schema_version' && exit 1

# 差分量制限
LINES=$(git diff --stat | awk '{s+=$1} END {print s}')
[ "$LINES" -gt 300 ] && exit 1
