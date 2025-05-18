import serial
import time

def test_communication(port, baudrate):
    try:
        with serial.Serial(port, baudrate, timeout=5) as ser:
            print(f"Connected to {port}")
            
            # 发送测试命令
            # ser.write(b"TEST\n")
            
            # 等待响应
            start_time = time.time()
            while time.time() - start_time < 5:  # 等待5秒
                if ser.in_waiting > 0:
                    response = ser.readline().decode().strip()
                    print(f"Received: {response}")
                    return True
                time.sleep(0.1)
            
            print("No response received within 5 seconds")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_communication("COM4", 115200)