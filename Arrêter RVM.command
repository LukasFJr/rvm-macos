#!/bin/bash
# ── Arrêter RVM — double-cliquer pour stopper l'interface ──────────────────

PID_FILE="/tmp/rvm_app.pid"

GRN='\033[0;32m'; YLW='\033[0;33m'; RED='\033[0;31m'; RST='\033[0m'

echo ""
echo "╔══════════════════════════════════════╗"
echo "║    RVM — Arrêt du serveur            ║"
echo "╚══════════════════════════════════════╝"
echo ""

STOPPED=0

# Méthode 1 : via PID file
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "   Arrêt du processus PID $PID..."
        kill "$PID"
        sleep 1
        # Force kill si toujours vivant
        kill -0 "$PID" 2>/dev/null && kill -9 "$PID" 2>/dev/null
        STOPPED=1
    fi
    rm -f "$PID_FILE"
fi

# Méthode 2 : chercher par nom si PID file absent/périmé
if [ "$STOPPED" -eq 0 ]; then
    PIDS=$(pgrep -f "python.*app\.py" 2>/dev/null)
    if [ -n "$PIDS" ]; then
        echo "   Processus trouvés : $PIDS"
        echo "$PIDS" | xargs kill 2>/dev/null
        sleep 1
        echo "$PIDS" | xargs kill -9 2>/dev/null
        STOPPED=1
    fi
fi

if [ "$STOPPED" -eq 1 ]; then
    echo -e "${GRN}✅ RVM arrêté.${RST}"
    osascript -e 'display notification "Serveur RVM arrêté." with title "RVM — Arrêté"' 2>/dev/null
else
    echo -e "${YLW}⚠️  RVM n'était pas en cours d'exécution.${RST}"
fi

echo ""
sleep 2
