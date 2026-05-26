#!/bin/bash
# ── Lancer RVM — double-cliquer pour démarrer l'interface ──────────────────

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="/tmp/rvm_app.pid"
LOG_FILE="/tmp/rvm_app.log"
PORT=7860

# Couleurs terminal
GRN='\033[0;32m'; YLW='\033[0;33m'; RED='\033[0;31m'; RST='\033[0m'

echo ""
echo "╔══════════════════════════════════════╗"
echo "║    RVM — Détourage Vidéo             ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── Détection Python ────────────────────────────────────────────────────────
# Ordre : pyenv 3.9.13 → autre version pyenv active → python3 système
_find_python() {
    local p="$HOME/.pyenv/versions/3.9.13/bin/python3"
    [ -x "$p" ] && echo "$p" && return
    if command -v pyenv &>/dev/null; then
        p="$(pyenv which python3 2>/dev/null)"
        [ -x "$p" ] && echo "$p" && return
    fi
    command -v python3
}
PYTHON="$(_find_python)"

# Déjà lancé ?
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo -e "${YLW}⚠️  RVM est déjà en cours d'exécution (PID $(cat "$PID_FILE")).${RST}"
    echo "   Ouverture du navigateur..."
    open "http://localhost:$PORT"
    exit 0
fi

# Vérifier python
if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
    echo -e "${RED}❌ Python introuvable.${RST}"
    echo "   Installez Python 3.9+ via pyenv, conda ou Homebrew,"
    echo "   puis relancez ce script."
    read -p "Appuyez sur Entrée pour fermer..."
    exit 1
fi

echo "   Python : $PYTHON"
echo ""

# Démarrer le serveur en arrière-plan
cd "$PROJECT_DIR"
echo "🚀 Démarrage du serveur RVM..."
nohup "$PYTHON" app.py > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "   PID : $(cat "$PID_FILE")"
echo ""

# Attendre que le port soit disponible (max 30s)
echo "⏳ Attente du serveur..."
for i in $(seq 1 30); do
    if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/" 2>/dev/null | grep -q "200"; then
        break
    fi
    printf "   %ds\r" "$i"
    sleep 1
done

# Vérifier que le serveur répond
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/" 2>/dev/null)
if [ "$STATUS" = "200" ]; then
    echo -e "${GRN}✅ Serveur prêt ! Ouverture du navigateur...${RST}"
    echo ""
    open "http://localhost:$PORT"
    osascript -e 'display notification "Interface ouverte sur http://localhost:7860" with title "RVM — Démarré"' 2>/dev/null
    echo "────────────────────────────────────────"
    echo "  Interface : http://localhost:$PORT"
    echo "  Logs      : $LOG_FILE"
    echo "  PID       : $(cat "$PID_FILE")"
    echo "────────────────────────────────────────"
    echo ""
    echo "  Double-cliquez sur 'Arrêter RVM.command' pour stopper."
    echo "  (Vous pouvez fermer ce Terminal — le serveur continue de tourner.)"
    echo ""
else
    echo -e "${RED}❌ Le serveur n'a pas démarré.${RST}"
    echo "   Consultez les logs :"
    tail -20 "$LOG_FILE"
    rm -f "$PID_FILE"
    read -p "Appuyez sur Entrée pour fermer..."
    exit 1
fi
