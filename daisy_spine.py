import serial
from sys import argv

class DaisySpine:
    ser = None
    def __init__(self, com_port = "/dev/ttyACM0", baud_rate = 9600, time_out = 1):
        self.ser = serial.Serial(com_port, baud_rate, timeout = time_out)

    def read_line(self):
        print(self.ser.readline())

    def read_all_lines(self):
        while self.ser.inWaiting() > 0:
            print(self.ser.readline())
        self.ser.flushInput()

    def pass_byte_basic(self, b):
        self.ser.write(bytes([int(b)]))

    def pass_byte(self, b):
        self.ser.flushInput()
        print("Passing byte " + str(b))
        self.pass_byte_basic(b)
        print(self.ser.readline())

    def forward(self):
        print("Forward")
        self.pass_byte(1)

    def backward(self):
        print("Backward")
        self.pass_byte(4)

    def halt(self):
        print("Stopping")
        self.pass_byte(0)

    def turn(self, d):
        print("Turning: " + str(d))

        if d == 0:
            self.pass_byte(2)
        elif d == 1:
            self.pass_byte(3)

if __name__ == "__main__":
    spine = None
    if len(argv) != 1:
        spine = DaisySpine(com_port = argv[1])
    else:
        spine = DaisySpine()

    spine.read_all_lines()
    spine.halt();
    while True:
        # spine.forward()
        input_str = input()
        if (len(input_str) > 0):
            spine.pass_byte(input_str)
