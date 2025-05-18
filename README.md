# STM32F103 Demo Project

This repository contains a demo project for the STM32F103 microcontroller using the STM32 HAL libraries. The project is configured with CMake and optimized for development with Visual Studio Code.

## Overview

This project demonstrates a basic setup for STM32F1 series microcontrollers with the following features:

- CMake-based build system
- STM32 HAL driver integration
- OpenOCD configuration for flashing and debugging
- VSCode integration with IntelliSense support
- UART communication for matrix operations
- Python client for sending matrices and receiving results

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
├── python/                # Python client for matrix operations
│   ├── matrix_calculator.py  # Python client script
│   └── README.md          # Python client documentation
├── CMakeLists.txt         # Main CMake configuration
├── flash.cfg              # OpenOCD flash configuration
├── run.sh                 # Build and flash script
├── start_calculator.sh    # One-click script to start all components
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

This project includes comprehensive debugging support through VSCode and the Cortex-Debug extension. The debugging configuration automatically builds your project and loads it to the microcontroller.

### Quick Start Debugging

1. Open the "Run and Debug" view (Ctrl+Shift+D)
2. Select a debug configuration from the dropdown menu:
   - **Cortex Debug**: Standard debugging session
   - **Flash & Debug**: Debugging with clean restart
3. Press F5 to start debugging
4. Use the debug toolbar to control execution

### Available Debug Configurations

#### Cortex Debug
- Standard debugging configuration
- Builds the project before launching
- Automatically loads the program to the device
- Halts at the main() function
- Enables semihosting support

#### Flash & Debug
- Similar to Cortex Debug but with additional reset commands
- Useful when you need a clean restart
- Ensures all breakpoints are enabled after loading

### Understanding the Debug Process

When you press F5 to start debugging:

1. **Build Phase**
   - The `Run Script Build` task executes `./run.sh --build-only`
   - This builds the project and generates .elf, .bin, and .hex files
   - Build errors will prevent the debug session from starting

2. **Debug Session Initialization**
   - OpenOCD starts as a GDB server
   - GDB connects to the OpenOCD server
   - The debug session establishes connection with the device

3. **Program Loading**
   - The `load` command in `postLaunchCommands` loads the program
   - This happens through GDB, not through a separate flash operation
   - The program is loaded directly into the device's memory

4. **Execution Control**
   - Execution halts at the main() function (due to `runToEntryPoint: "main"`)
   - Breakpoints become active
   - You can now step through code, inspect variables, etc.

### Advanced Debugging Features

#### Live Watch
The Cortex Debug configuration includes live watch support:
```json
"liveWatch": {
  "enabled": true,
  "samplesPerSecond": 4
}
```
This allows you to monitor variables in real-time while the program is running.

#### Semihosting
Semihosting is enabled by default, allowing you to use printf for debug output:
```c
// In your code
printf("Debug value: %d\n", value);
```
The output will appear in the Debug Console.

#### Memory View
- Open the Memory view in VSCode to inspect memory regions
- Useful for checking register values and memory contents
- Access via Debug sidebar or Command Palette

### Debugging vs. Production Flashing

#### For Development (Debugging)
- Use VSCode's debug button (F5)
- Program is loaded via GDB's `load` command
- Maintains debug context and symbol information

#### For Production
- Use the "Production Flash" task
- Or run `./run.sh` directly
- Writes the program permanently to flash

### Debugging Tips

1. **Clean Build When Needed**
   - If you encounter strange behavior, use "Run Script Clean Build" task
   - This performs a clean rebuild of the project

2. **SVD File for Peripheral Registers**
   - The project includes an SVD file for register inspection
   - View peripheral registers in the Cortex-Debug sidebar

3. **Breakpoint Types**
   - Normal breakpoints: Click in the gutter
   - Conditional breakpoints: Right-click > Add Conditional Breakpoint
   - Data breakpoints: Break when a memory location changes

4. **Optimize for Debugging**
   - Use `-Og` optimization level for best debugging experience
   - Add `volatile` to variables you want to inspect reliably

5. **Debug Console Commands**
   - You can enter GDB commands directly in the Debug Console
   - Example: `p/x variable` to print a variable in hex format

## Troubleshooting

### Troubleshooting Debug Issues

#### Build-Related Problems

1. **Build Fails Before Debug**
   - Check the build output in the terminal
   - Ensure all required tools are in PATH
   - Try "Run Script Clean Build" task
   - Verify CMake configuration is correct

2. **Missing Binary Files**
   - Check if .elf file exists in build directory
   - Ensure build process completed successfully
   - Verify file paths in launch.json are correct

#### Connection Issues

1. **OpenOCD Connection Fails**
   - Check physical connections:
     * Debugger is properly connected
     * Power supply is stable
     * All pins are correctly connected
   - Verify interface configuration:
     * Check interface type in flash.cfg
     * Try different interface speeds
   - Run OpenOCD with verbose output:
     ```bash
     openocd -f flash.cfg -d3
     ```

2. **GDB Connection Problems**
   - Ensure OpenOCD is running properly
   - Check if another debug session is active
   - Verify GDB port settings in launch.json
   - Try killing and restarting OpenOCD

#### Program Loading Issues

1. **Program Won't Load**
   - Check if the .elf file is valid:
     ```bash
     arm-none-eabi-readelf -h build/your_project.elf
     ```
   - Verify memory settings in linker script
   - Ensure device is properly erased

2. **Program Loads But Won't Run**
   - Check if entry point is correct
   - Verify reset configuration
   - Look for hardfault handlers being triggered

#### Debugging Problems

1. **Breakpoints Not Working**
   - Ensure code is built with debug information
   - Check if optimization level is too aggressive
   - Verify breakpoint is set at a valid location
   - Try setting breakpoint after loading program

2. **Can't See Variable Values**
   - Check if variable is optimized out
   - Use volatile for important debug variables
   - Ensure debug symbols are present
   - Try viewing memory location directly

3. **Program Behavior Different When Debugging**
   - Check for timing-sensitive code
   - Look for watchdog timer issues
   - Verify interrupt handling
   - Consider effects of debug clock settings

#### SVD File Issues

1. **Peripheral Registers Not Visible**
   - Verify SVD file is correctly referenced in launch.json
   - Check if SVD file matches your device
   - Try reloading the debug window

2. **Wrong Register Values**
   - Ensure you're using the correct SVD file version
   - Check if peripheral is properly initialized
   - Verify clock settings

#### Common Solutions

1. **General Debug Session Problems**
   - Close VSCode and reopen
   - Kill all OpenOCD processes
   - Reconnect the debugger
   - Power cycle the target device

2. **Build Environment Issues**
   - Verify environment variables
   - Check tool versions compatibility
   - Ensure all dependencies are installed

3. **Code Execution Issues**
   - Check stack size configuration
   - Verify interrupt vector table
   - Look for initialization problems
   - Monitor system clock configuration

## Matrix Determinant Calculator

This project includes a matrix determinant calculator feature that demonstrates UART communication between the STM32 microcontroller and a Python client.

### Features

- Receive matrices via UART
- Calculate matrix determinants (up to 10x10)
- Send results back via UART
- Python client for easy interaction
- Verification using NumPy

### How It Works

1. The STM32 firmware initializes UART1 (115200 baud, 8N1)
2. Python client sends a matrix in the format: `size,a11,a12,...,ann\n`
3. STM32 parses the matrix and calculates its determinant
4. Result is sent back as: `Determinant: X.XXXXXX\r\n`
5. Python client verifies the result using NumPy

### Using the Calculator

#### One-Click Start

The easiest way to use the calculator is with the provided start script:

```bash
chmod +x start_calculator.sh
./start_calculator.sh
```

This will:
1. Build and flash the STM32 firmware
2. Install required Python packages
3. Start the Python client with a random 3x3 matrix

#### Manual Usage

You can also use the Python client directly:

```bash
# Generate and send a random 3x3 matrix
python python/matrix_calculator.py -p <PORT> -s 3

# Send a specific 2x2 matrix
python python/matrix_calculator.py -p <PORT> -s 2 -m "1,2,3,4"
```

See `python/README.md` for more details on the Python client.

## License

[Include your license information here]

## Acknowledgments

- STM32 HAL libraries by STMicroelectronics
- CMSIS by ARM
- NumPy for matrix operations in Python

## Contributing

[Include contribution guidelines if applicable]