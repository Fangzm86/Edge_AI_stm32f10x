#!/usr/bin/env python3
"""
Matrix Calculator - Python Client for STM32 Matrix Determinant Calculator

This script sends a matrix to an STM32 microcontroller via UART,
which calculates the determinant and returns the result.
"""

import serial
import numpy as np
import time
import argparse
import sys
import os

def send_matrix(port, baudrate, matrix):
    """
    Send a matrix to the STM32 and receive the calculated determinant.
    
    Args:
        port (str): Serial port name
        baudrate (int): Baud rate
        matrix (numpy.ndarray): Matrix to send
    
    Returns:
        float: Calculated determinant
    """
    try:
        # Open serial port
        ser = serial.Serial(port, baudrate, timeout=5)
        print(f"Connected to {port} at {baudrate} baud")
        
        # Wait for device to initialize
        time.sleep(2)
        
        # Read welcome message
        welcome = ser.read_until(b'\n').decode('utf-8').strip()
        print(f"Device says: {welcome}")
        
        # Format matrix as string: size,a11,a12,...,ann
        size = matrix.shape[0]
        matrix_str = f"{size}"
        
        for i in range(size):
            for j in range(size):
                matrix_str += f",{matrix[i, j]}"
        
        matrix_str += "\n"
        
        print(f"Sending matrix: {matrix}")
        print(f"Formatted as: {matrix_str.strip()}")
        
        # Send matrix
        ser.write(matrix_str.encode('utf-8'))
        
        # Wait for response
        response = ser.read_until(b'\n').decode('utf-8').strip()
        print(f"Response: {response}")
        
        # Parse determinant from response
        if "Determinant:" in response:
            determinant = float(response.split(":")[1].strip())
            print(f"Calculated determinant: {determinant}")
            
            # Verify with numpy
            np_det = np.linalg.det(matrix)
            print(f"NumPy determinant: {np_det}")
            print(f"Difference: {abs(determinant - np_det)}")
            
            return determinant
        else:
            print(f"Error: {response}")
            return None
    
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        return None
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed")

def generate_random_matrix(size):
    """Generate a random matrix of given size"""
    return np.random.randint(-10, 10, (size, size))

def main():
    """
    Main function for Matrix Determinant Calculator Client.
    
    Parses command line arguments to either:
    - Accept a user-provided matrix as comma-separated values
    - Generate a random matrix of specified size
    
    Validates matrix dimensions and sends the matrix via serial port.
    
    Args:
        --port/-p (str): Required serial port name
        --baudrate/-b (int): Baud rate (default: 115200)
        --size/-s (int): Matrix size (default: 3)
        --matrix/-m (str): Optional comma-separated matrix values
    
    Exits with error code 1 if:
    - Matrix dimensions don't match specified size
    - Required arguments are missing
    """
    parser = argparse.ArgumentParser(description='Matrix Determinant Calculator Client')
    parser.add_argument('--port', '-p', required=True, help='Serial port')
    parser.add_argument('--baudrate', '-b', type=int, default=115200, help='Baud rate (default: 115200)')
    parser.add_argument('--size', '-s', type=int, default=3, help='Matrix size (default: 3)')
    parser.add_argument('--matrix', '-m', help='Matrix as comma-separated values (row-major order)')
    
    args = parser.parse_args()
    
    if args.matrix:
        # Parse matrix from command line
        values = [float(x) for x in args.matrix.split(',')]
        if len(values) != args.size * args.size:
            print(f"Error: Matrix size ({args.size}x{args.size}) doesn't match number of values ({len(values)})")
            sys.exit(1)
        
        matrix = np.array(values).reshape((args.size, args.size))
    else:
        # Generate random matrix
        matrix = generate_random_matrix(args.size)
    
    send_matrix(args.port, args.baudrate, matrix)

if __name__ == "__main__":
    main()