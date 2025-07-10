#!/bin/bash
# MLFlow Local Server Startup Script
# Ensures privacy-compliant experiment tracking with no external data transmission

set -e

# Configuration
MLFLOW_HOST="${MLFLOW_HOST:-localhost}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_BACKEND="${MLFLOW_BACKEND:-file:./mlflow_experiments}"
MLFLOW_ARTIFACTS="${MLFLOW_ARTIFACTS:-file:./mlflow_artifacts}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting MLFlow Local Server (Privacy-Compliant)${NC}"
echo "=================================================="

# Check if MLFlow is installed
if ! command -v mlflow &> /dev/null; then
    echo -e "${RED}Error: MLFlow is not installed${NC}"
    echo "Please install MLFlow: pip install mlflow>=2.8.0"
    exit 1
fi

# Create directories if they don't exist
echo -e "${YELLOW}Creating MLFlow directories...${NC}"
mkdir -p mlflow_experiments
mkdir -p mlflow_artifacts
mkdir -p logs/tensorboard

# Check if port is already in use
if lsof -Pi :$MLFLOW_PORT -sTCP:LISTEN -t >/dev/null ; then
    echo -e "${YELLOW}Warning: Port $MLFLOW_PORT is already in use${NC}"
    echo "You can access the existing MLFlow UI at: http://$MLFLOW_HOST:$MLFLOW_PORT"
    echo "Or kill the existing process and restart this script"
    exit 0
fi

# Start MLFlow server
echo -e "${GREEN}Starting MLFlow tracking server...${NC}"
echo "Host: $MLFLOW_HOST"
echo "Port: $MLFLOW_PORT"
echo "Backend: $MLFLOW_BACKEND"
echo "Artifacts: $MLFLOW_ARTIFACTS"
echo ""

echo -e "${BLUE}Privacy Notice:${NC}"
echo "✅ All data stays on your local machine"
echo "✅ No external connections or data transmission"
echo "✅ Fully offline capable"
echo "✅ Complete data sovereignty"
echo ""

# Start the server
echo -e "${GREEN}MLFlow server starting...${NC}"
echo "Access the UI at: http://$MLFLOW_HOST:$MLFLOW_PORT"
echo "Press Ctrl+C to stop the server"
echo ""

mlflow server \
    --host "$MLFLOW_HOST" \
    --port "$MLFLOW_PORT" \
    --backend-store-uri "$MLFLOW_BACKEND" \
    --default-artifact-root "$MLFLOW_ARTIFACTS" \
    --serve-artifacts 