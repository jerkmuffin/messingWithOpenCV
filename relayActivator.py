from serial import Serial
import os

s = Serial('/dev/ttyUSB0')

def on():
  s.write(b'\xa0\x01\x01\xa2')
def off():
  s.write(b'\xa0\x01\x00\xa1')

