#!/bin/bash

# Configuration
SESSION_NAME="init6_session"
SCRIPT_FILE="init6.py"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/init6_$(date +%Y%m%d_%H%M%S).log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Check if init6.py exists
if [ ! -f "$SCRIPT_FILE" ]; then
    echo "Error: $SCRIPT_FILE not found in current directory"
    exit 1
fi

# Kill existing session if it exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

# Create new tmux session and run the script with logging
tmux new-session -d -s "$SESSION_NAME" -c "$(pwd)" \
    "python3 $SCRIPT_FILE 2>&1 | tee $LOG_FILE"

# Optional: Set up logging in tmux itself
tmux pipe-pane -t "$SESSION_NAME" "cat >> $LOG_FILE"

echo "tmux session '$SESSION_NAME' created successfully"
echo "Running: python3 $SCRIPT_FILE"
echo "Logs saved to: $LOG_FILE"
echo ""
echo "To attach to session: tmux attach-session -t $SESSION_NAME"
echo "To detach from session: Ctrl+b then d"
echo "To kill session: tmux kill-session -t $SESSION_NAME"

# Optional: Attach to the session immediately
# Uncomment the line below if you want to attach automatically
# tmux attach-session -t "$SESSION_NAME"