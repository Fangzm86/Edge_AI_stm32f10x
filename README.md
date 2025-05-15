# STM32F103 Demo Project

This repository contains a demo project for the STM32F103 microcontroller using the STM32 HAL libraries. The project is configured with CMake and optimized for development with Visual Studio Code.

## Overview

This project demonstrates a basic setup for STM32F1 series microcontrollers with the following features:

- CMake-based build system
- STM32 HAL driver integration
- OpenOCD configuration for flashing and debugging
- VSCode integration with IntelliSense support

## Directory Structure

```
├── .vscode/               # VSCode configuration files
├── Core/                  # Application code
│   ├── Inc/               # Header files
│   └── Src/               # Source files
├── Drivers/               # STM32 HAL and CMSIS drivers
│   ├── CMSIS/             # CMSIS core files
│   └── STM32F1xx_HAL_Driver/ # STM32F1 HAL drivers
├── cmake/                 # CMake configuration files
├── build/                 # Build output directory
├── CMakeLists.txt         # Main CMake configuration
├── flash.cfg              # OpenOCD flash configuration
└── README.md              # This file
```

## Prerequisites

To build and run this project, you'll need:

- ARM GCC Toolchain (arm-none-eabi-gcc)
- CMake (version 3.14 or higher)
- Ninja build system
- OpenOCD
- CMSIS-DAP compatible debugger
- Visual Studio Code with the following extensions:
  - C/C++ Extension
  - CMake Tools
  - Cortex-Debug

## Setting Up the Development Environment

### Installing the Toolchain

#### Windows

##### Recommended Method: Using MSYS2

Using MSYS2 is recommended for Windows as it simplifies installation and automatically handles PATH configuration:

1. Download and install [MSYS2](https://www.msys2.org/)
2. Open the MSYS2 MinGW 64-bit terminal
3. Install the required tools:

```bash
# Update package database
pacman -Syu

# Install tools
pacman -S mingw-w64-x86_64-cmake
pacman -S mingw-w64-x86_64-ninja
pacman -S mingw-w64-x86_64-openocd
```

4. Download and install the [ARM GCC Toolchain](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads) separately
5. Add the ARM GCC Toolchain bin directory to your PATH environment variable

##### Alternative Method: Manual Installation

If you prefer manual installation:

1. Download and install the [ARM GCC Toolchain](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads)
2. Download and install [CMake](https://cmake.org/download/)
3. Download and install [Ninja](https://github.com/ninja-build/ninja/releases)
4. Download and install [OpenOCD](https://openocd.org/pages/getting-openocd.html)
5. Add all tools to your PATH environment variable manually

#### Linux

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install gcc-arm-none-eabi cmake ninja-build openocd

# Fedora
sudo dnf install arm-none-eabi-gcc-cs cmake ninja-build openocd
```

#### macOS

```bash
# Using Homebrew
brew install --cask gcc-arm-embedded
brew install cmake ninja openocd
```

### Setting Up VSCode

1. Install Visual Studio Code
2. Install the recommended extensions:
   - C/C++ Extension (ms-vscode.cpptools)
   - CMake Tools (ms-vscode.cmake-tools)
   - Cortex-Debug (marus25.cortex-debug)

## Building the Project

### Using VSCode

1. Open the project folder in VSCode
2. Select the "CMake: Configure" command from the command palette (Ctrl+Shift+P)
3. Select the "Build" task from the command palette or press Ctrl+Shift+B

### Using Command Line

```bash
# Configure CMake
cmake -B build -G Ninja

# Build the project
cmake --build build
```

## Flashing the Firmware

### Using OpenOCD

The project includes a pre-configured `flash.cfg` file for OpenOCD:

```bash
# Flash using OpenOCD
openocd -f flash.cfg
```

### Using VSCode

1. Open the "Run and Debug" view (Ctrl+Shift+D)
2. Select "Flash & Debug" from the dropdown menu
3. Press F5 to flash and start debugging

## Debugging

### Using VSCode

1. Open the "Run and Debug" view (Ctrl+Shift+D)
2. Select "Cortex Debug" from the dropdown menu
3. Press F5 to start debugging
4. Use the debug toolbar to control execution

## Troubleshooting

### Common Issues

#### Header File Recognition Issues

If VSCode shows errors about unrecognized types or headers:

1. Reload VSCode
2. Reset the IntelliSense database (C/C++: Reset IntelliSense Database)
3. Check the `.vscode/c_cpp_properties.json` file for correct include paths

#### OpenOCD Connection Issues

If OpenOCD fails to connect to your device:

1. Check that your debugger is properly connected
2. Verify that the correct interface is selected in `flash.cfg`
3. Try running OpenOCD with verbose output: `openocd -f flash.cfg -d3`

## License

[Include your license information here]

## Acknowledgments

- STM32 HAL libraries by STMicroelectronics
- CMSIS by ARM

## Contributing

[Include contribution guidelines if applicable]