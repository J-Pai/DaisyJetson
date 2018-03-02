import serial
from sys import argv

class DaisySpine:
    ser = None
    def __init__(self, com_port = "/dev/ttyACM1", baud_rate = 28800, time_out = 1):
        self.ser = serial.Serial(com_port, baud_rate, timeout = time_out)

    def read_line(self):
        print(self.ser.readline())

    def read_all_lines_debug(self, chunk_size = 200):
        print("=== READING ALL LINES ===")

        read_buffer = b''

        while True:
            byte_chunk = self.ser.read(size = chunk_size)
            read_buffer += byte_chunk
            if not len(byte_chunk) == chunk_size:
                break

        print(read_buffer.decode("utf-8"))
        self.ser.reset_input_buffer()

        return read_buffer;

        print("=== DONE ===")

    def read_all_lines(self, chunk_size = 50):
        read_buffer = b''

        while True:
            byte_chunk = self.ser.read(size = chunk_size)
            read_buffer += byte_chunk
            if not len(byte_chunk) == chunk_size:
                break

        self.ser.reset_input_buffer()

        return read_buffer;

    def pass_byte_basic(self, b):
        self.ser.write(bytes([int(b)]))

    def pass_byte_debug(self, b):
        print("+++ PASSING BYTE +++")

        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        if (int(b) > 255 or int(b) < 0):
            print("Byte out of range: " + b)
            print("+++ FAIL +++")
            return

        print("Passing byte " + str(b))
        self.pass_byte_basic(b)

        ret = self.read_all_lines()

        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        print("+++ DONE +++")
        return ret

    def pass_byte(self, b):
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        if (int(b) > 255 or int(b) < 0):
            print("Byte out of range: " + b)
            return

        self.pass_byte_basic(b)
        ret = self.read_all_lines()

        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        return ret

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

    print(spine.read_all_lines())
    spine.halt();
    while True:
        # spine.forward()
        input_str = input()
        if (len(input_str) > 0):
            print(spine.pass_byte(input_str))
