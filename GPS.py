import serial
ser = serial.Serial('/dev/ttyTHS1', 9600, timeout=1)

# For testing connection between Jetson Orin Nano and Microcontroller through UART
while True:
    line = ser.readline().decode('utf-8', errors='ignore').strip()
    if line.startswith('$'):
        print("GPS Data:", line)