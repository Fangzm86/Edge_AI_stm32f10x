#!/bin/bash
# run.sh - Build and flash script for STM32F1 project
# Updated for new project structure with embedded/ directory

# Exit on error
set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root and embedded directory
EMBEDDED_DIR=$(dirname "$(readlink -f "$0")")
echo "embedded_dir: $EMBEDDED_DIR"

# Print colored message
print_msg() {
    echo -e "${2}${1}${NC}"
}

# Print error message and exit
error_exit() {
    print_msg "ERROR: $1" "${RED}"
    exit 1
}

# Check if command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        error_exit "$1 is not installed or not in PATH. Please install it first."
    fi
}

# Check required commands
check_command cmake
check_command ninja
check_command openocd
check_command arm-none-eabi-gcc
check_command arm-none-eabi-objcopy

# Parse command line arguments
BUILD_ONLY=0
FLASH_ONLY=0
CLEAN_BUILD=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            BUILD_ONLY=1
            shift
            ;;
        --flash-only)
            FLASH_ONLY=1
            shift
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --build-only    Only build the project, don't flash"
            echo "  --flash-only    Only flash the project, don't build"
            echo "  --clean         Clean build directory before building"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            error_exit "Unknown option: $1"
            ;;
    esac
done

# Navigate to embedded directory
cd "${EMBEDDED_DIR}" || error_exit "Failed to enter embedded directory"

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    print_msg "Creating build directory..." "${BLUE}"
    mkdir -p build
fi

# Clean build if requested
if [ $CLEAN_BUILD -eq 1 ] && [ $FLASH_ONLY -eq 0 ]; then
    print_msg "Cleaning build directory..." "${BLUE}"
    rm -rf build/*
fi

# Build the project
if [ $FLASH_ONLY -eq 0 ]; then
    print_msg "Configuring project with CMake..." "${BLUE}"
    cmake -B build -G Ninja || error_exit "CMake configuration failed"
    
    print_msg "Building project..." "${BLUE}"
    cmake --build build || error_exit "Build failed"
    
    # Get project name from CMakeLists.txt
    PROJECT_NAME=$(grep "project(" CMakeLists.txt | cut -d'(' -f2 | cut -d' ' -f1 | tr -d ')')
    
    # Check if the ELF file exists
    if [ ! -f "build/${PROJECT_NAME}.elf" ]; then
        # Try to find any .elf file
        ELF_FILE=$(find build -name "*.elf" | head -n 1)
        if [ -z "$ELF_FILE" ]; then
            error_exit "Could not find .elf file in build directory"
        else
            PROJECT_NAME=$(basename "$ELF_FILE" .elf)
            print_msg "Found ELF file: $ELF_FILE" "${YELLOW}"
        fi
    fi
    
    # Generate .bin and .hex files
    print_msg "Generating binary files..." "${BLUE}"
    arm-none-eabi-objcopy -O binary "build/${PROJECT_NAME}.elf" "build/${PROJECT_NAME}.bin" || error_exit "Failed to generate .bin file"
    arm-none-eabi-objcopy -O ihex "build/${PROJECT_NAME}.elf" "build/${PROJECT_NAME}.hex" || error_exit "Failed to generate .hex file"
    
    print_msg "Build successful!" "${GREEN}"
    print_msg "Generated files:" "${BLUE}"
    print_msg "  - build/${PROJECT_NAME}.elf" "${BLUE}"
    print_msg "  - build/${PROJECT_NAME}.bin" "${BLUE}"
    print_msg "  - build/${PROJECT_NAME}.hex" "${BLUE}"
fi

# Flash the program
if [ $BUILD_ONLY -eq 0 ]; then
    if [ ! -f "flash.cfg" ]; then
        error_exit "flash.cfg not found in embedded directory"
    fi
    
    print_msg "Flashing program to device..." "${BLUE}"
    openocd -f flash.cfg || error_exit "Flashing failed"
    
    print_msg "Flashing successful!" "${GREEN}"
fi

print_msg "All operations completed successfully!" "${GREEN}"
exit 0