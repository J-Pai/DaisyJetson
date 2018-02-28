import serial

ser = serial.Serial('/dev/cu.usbmodem1421', 115200)

def passByte(b):
    print("Passing byte " + str(b))
    ser.write(bytes([int(b)]))

def forward():
    print("Forward")
    passByte(0)

def backward():
    print("Backward")
    passByte(1)

def halt():
    print("Stopping")
    passByte(2)

def turn(d):
    print("Turning: " + str(d))

    if d == 0:
        passByte(3)
    elif d == 1:
        passByte(4)

