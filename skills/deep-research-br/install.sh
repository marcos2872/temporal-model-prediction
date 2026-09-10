#!/usr/bin/env bash
# Instalador dual OpenCode + Claude Code (PT-BR). Uso: bash skills/deep-research-br/install.sh [--global]
set -euo pipefail
DEST_SKILL="deep-research-br"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${1:-}"
copy_to() { mkdir -p "$1/$DEST_SKILL"; cp -r "$SRC/SKILL.md" "$SRC/scripts" "$SRC/references" "$SRC/evals" "$SRC/THIRD_PARTY.md" "$1/$DEST_SKILL/"; echo "  -> $1/$DEST_SKILL"; }
echo "Instalando skill $DEST_SKILL a partir de $SRC"
copy_to "./.opencode/skills"
copy_to "./.claude/skills"
if [ "$PREFIX" = "--global" ]; then
  copy_to "$HOME/.config/opencode/skills"
  copy_to "$HOME/.claude/skills"
fi
echo "OK. Teste: python3 skills/deep-research-br/evals/smoke.py"
