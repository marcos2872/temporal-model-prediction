#!/usr/bin/env bash
# Baixa os checkpoints treinados do GitHub Release `modelos-v1` e extrai na raiz do repo.
# Uso: bash scripts/baixar_modelos.sh [--dir /tmp/modelos] [--tag modelos-v1]
# Requer: gh (preferido) ou curl + tar + sha256sum.
set -euo pipefail

REPO="marcos2872/temporal-model-prediction"
TAG="modelos-v1"
DEST="/tmp/modelos"
while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DEST="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    *) echo "uso: $0 [--dir DIR] [--tag TAG]" >&2; exit 1;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$DEST"
cd "$ROOT"

ARQS="00-baseline-ph-modelos.tar.gz 01-baseline-od-modelos.tar.gz 02-lstnet-ph-modelos.tar.gz 03-lstnet-od-modelos.tar.gz 04-patchtst-ph-modelos.tar.gz 05-patchtst-od-modelos.tar.gz 06-ensemble-ph-modelos.tar.gz 07-ensemble-od-modelos.tar.gz SHA256SUMS.txt"

if command -v gh >/dev/null 2>&1; then
  gh release download "$TAG" --repo "$REPO" --pattern '*modelos.tar.gz' --pattern 'SHA256SUMS.txt' --dir "$DEST" --clobber
else
  for f in $ARQS; do
    curl -fL -o "$DEST/$f" "https://github.com/$REPO/releases/download/$TAG/$f"
  done
fi

(cd "$DEST" && sha256sum -c SHA256SUMS.txt)
for f in "$DEST"/*-modelos.tar.gz; do
  tar -xzf "$f" -C "$ROOT"
done
echo "OK: checkpoints extraídos em resultados/*/modelos/ (confira com: ls resultados/*/modelos/)"
