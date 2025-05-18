#!/usr/bin/env python3
"""
Matrix Calculator - Python Client for STM32 Matrix Determinant Calculator

This script sends a matrix to an STM32 microcontroller via UART,
which calculates the determinant and returns the result.
After receiving the result, it continues to listen for any additional data.
"""

import serial
import numpy as np
import time
import argparse
import sys
import os

def send_matrix_and_receive(port, baudrate, matrix):
    """
    Send a matrix to the STM32, receive the calculated determinant,
    and then continue listening for additional data.
    
    Args:
        port (str): Serial port name
        baudrate (int): Baud rate
        matrix (numpy.ndarray): Matrix to send
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
        else:
            print(f"Error: {response}")
        
        # Continue listening for additional data
        print("\nNow listening for additional data... (Press Ctrl+C to exit)")
        ser.timeout = 1  # Set shorter timeout for continuous reading
        
        while True:
            try:
                if ser.in_waiting > 0:
                    data = ser.readline().decode('utf-8').strip()
                    if data:
                        print(f"Received: {data}")
                time.sleep(0.1)  # Small delay to reduce CPU usage
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except UnicodeDecodeError:
                print("Warning: Received invalid data")
                continue
    
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed")

def generate_random_matrix(size):
    """Generate a random matrix of given size"""
    return np.random.randint(-10, 10, (size, size))

def main():
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
    
    send_matrix_and_receive(args.port, args.baudrate, matrix)

if __name__ == "__main__":
    main()