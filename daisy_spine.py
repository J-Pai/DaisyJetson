#!/usr/bin/env python3

import serial
import struct
from sys import argv
from enum import Enum

class Dir(Enum):
    CW = 0
    CCW = 1

class DaisySpine:
    ser = None
    def __init__(self, com_port = "/dev/ttyACM0", baud_rate = 28800, time_out = 1):
        self.ser = serial.Serial(com_port, baud_rate, timeout = time_out)

    def read_line(self):
        return self.ser.readline()

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

        return read_buffer;

    def pass_byte_basic(self, b):
        self.ser.reset_output_buffer()
        self.ser.write(bytes([int(b)]))

    def pass_byte_debug(self, b):
        print("+++ PASSING BYTE +++")

        if (int(b) > 255 or int(b) < 0):
            print("Byte out of range: " + b)
            print("+++ FAIL +++")
            return

        print("Passing byte " + str(b))
        self.pass_byte_basic(b)
        ret = self.read_line()

        print("+++ DONE +++")
        return ret

    def pass_byte(self, b):
        if (int(b) > 255 or int(b) < 0):
            print("Byte out of range: " + b)
            return

        self.pass_byte_basic(b)
        ret = self.read_line()
        return ret

    def forward(self):
        return self.pass_byte(1)

    def backward(self):
        return self.pass_byte(4)

    def halt(self):
        return self.pass_byte(0)

    def turn(self, d):
        if d == Dir.CW:
            return self.pass_byte(2)
        elif d == Dir.CCW:
            return self.pass_byte(3)

    def move(self, leftSpeed, rightSpeed):
        leftByte = struct.pack(">h", leftSpeed)
        rightByte = struct.pack(">h", rightSpeed)

        cmdStr = [5]
        cmdStr.extend(bytearray(leftByte))
        cmdStr.extend(bytearray(rightByte))

        self.ser.reset_input_buffer()
        self.ser.write(cmdStr)
        return self.read_line()

if __name__ == "__main__":
    spine = None
    if len(argv) != 1:
        spine = DaisySpine(com_port = argv[1])
    else:
        spine = DaisySpine()

    print(spine.read_all_lines())
    spine.halt();
    while True:
        input_str = input()
        code = int(input_str)
        if code == 0:
            print(spine.halt())
        if code == 1:
            print(spine.forward())
        if code == 2:
            print(spine.turn(Dir.CW))
        if code == 3:
            print(spine.turn(Dir.CCW))
        if code == 4:
            print(spine.backward())
