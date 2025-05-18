# Matrix Calculator Python Client

This Python client communicates with the STM32 microcontroller to calculate matrix determinants and continuously monitor serial data.

## Features

- Send matrices to STM32 via UART
- Receive calculated determinants
- Verify results using NumPy
- Automatically monitor additional serial data
- Support for random matrix generation
- Command-line interface

## Requirements

- Python 3.6 or higher
- Required packages:
  - pyserial
  - numpy

## Installation

Install the required packages:

```bash
pip install pyserial numpy
```

## Usage

### Basic Usage

```bash
python matrix_calculator.py -p <PORT> -s <SIZE>
```

This will:
1. Generate and send a random matrix
2. Receive and display the calculated determinant
3. Continue monitoring the serial port for additional data
4. Press Ctrl+C to exit

### Command-line Options

- `-p, --port`: Serial port (e.g., COM3, /dev/ttyUSB0) [required]
- `-b, --baudrate`: Baud rate (default: 115200)
- `-s, --size`: Matrix size (default: 3)
- `-m, --matrix`: Matrix as comma-separated values (row-major order)

### Examples

1. Generate and send a random 3x3 matrix:
   ```bash
   python matrix_calculator.py -p COM3 -s 3
   ```

2. Send a specific 2x2 matrix:
   ```bash
   python matrix_calculator.py -p COM3 -s 2 -m "1,2,3,4"
   ```

## Communication Protocol

### Send Format
```
size,a11,a12,...,ann\n
```

Example for 2x2 matrix:
```
2,1,2,3,4\n
```

### Receive Format
For matrix calculations:
```
Determinant: X.XXXXXX\n
```

For additional data:
```
[Any data format sent by STM32]\n
```

## Verification

The client automatically verifies the STM32's calculation by:
1. Computing the determinant using NumPy
2. Comparing the results
3. Displaying the difference

## Troubleshooting

1. **Serial Port Issues**
   - Check that the port exists and is available
   - Verify you have permission to access the port
   - Try a different port if available

2. **Communication Problems**
   - Ensure the STM32 is properly programmed
   - Check that the baud rate matches (default: 115200)
   - Verify the matrix size is within limits (max 10x10)

3. **Data Reception Issues**
   - If no data appears, check if STM32 is sending data
   - Verify the baud rate matches
   - Check if data ends with newline characters
   - Press Ctrl+C to exit monitoring mode

## Program Flow

1. **Initialization**
   - Open serial port
   - Wait for device initialization
   - Read welcome message

2. **Matrix Operation**
   - Send matrix data
   - Receive determinant result
   - Verify with NumPy calculation

3. **Continuous Monitoring**
   - Automatically switch to monitoring mode
   - Display any received data
   - Exit with Ctrl+C

## Error Handling

- Serial port connection errors
- Data format errors
- Invalid matrix size
- Communication timeouts
- Invalid data reception