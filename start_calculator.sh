#!/bin/bash

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    echo -e "${2}${1}${NC}"
}

# Print error message and exit
error_exit() {
    print_msg "ERROR: $1" "${RED}"
    exit 1
}

# Check if a command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        error_exit "$1 is not installed. Please install it first."
    fi
}

# Check required commands
check_command python
check_command pip3

# Install Python requirements if needed
if ! python -c "import serial" &> /dev/null; then
    print_msg "Installing pyserial..." "${BLUE}"
    pip3 install pyserial numpy || error_exit "Failed to install Python requirements"
fi

# Build and flash STM32 program
print_msg "Building and flashing STM32 program..." "${BLUE}"
./embedded/run.sh || error_exit "Failed to build and flash STM32 program"

# Find serial port
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Linux detected"
    # Linux: try to find USB-Serial device
    PORT=$(ls /dev/ttyUSB* 2>/dev/null | head -n1)
    if [ -z "$PORT" ]; then
        PORT=$(ls /dev/ttyACM* 2>/dev/null | head -n1)
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS: try to find USB-Serial device
    PORT=$(ls /dev/tty.usbserial* 2>/dev/null | head -n1)
    if [ -z "$PORT" ]; then
        PORT=$(ls /dev/tty.usbmodem* 2>/dev/null | head -n1)
    fi
else
    # Windows: assume COM4 (user should modify this)
    echo "Windows detected"
    PORT=$(yq eval '.hardware.serial_port' ./config.yaml)
    echo "PORT: $PORT"
    if [ -z "$PORT" ]; then
        PORT="COM4"
    fi
fi

if [ -z "$PORT" ]; then
    error_exit "No serial port found"
fi

print_msg "Using serial port: $PORT" "${BLUE}"

# Start Python script
print_msg "Starting Matrix Calculator..." "${GREEN}"
python ./python/matrix_calculator.py -p "$PORT" -s 3 || error_exit "Failed to run Python script"