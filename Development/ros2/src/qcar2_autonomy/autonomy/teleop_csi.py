
import sys
import tty
import termios
import os
import time

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2

SPEED = 0.15   # forward/back speed
TURN  = 0.4    # turn rate

def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

def main():
    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs...")
    if not qlabs.open("localhost"):
        print("Unable to connect — is workspace_setup.py running?")
        sys.exit()
    print("Connected\n")

    car = QLabsQCar2(qlabs)
    car.actorNumber = 0   # must match actorNumber in workspace_setup.py

    print("Controls: W=forward  S=back  A=left  D=right  SPACE=stop  Q=quit")
    print("─────────────────────────────────────────────────────────────────")

    speed = 0.0
    turn  = 0.0

    while True:
        key = get_key().lower()

        if key == 'q':
            car.set_velocity_and_request_state(
                forward=0, turn=0,
                headlights=False, leftTurnSignal=False,
                rightTurnSignal=False, brakeSignal=False, reverseSignal=False
            )
            print("\nStopped. Bye.")
            break
        elif key == 'w':
            speed =  SPEED; turn = 0.0
        elif key == 's':
            speed = -SPEED; turn = 0.0
        elif key == 'a':
            speed =  SPEED; turn =  TURN
        elif key == 'd':
            speed =  SPEED; turn = -TURN
        elif key == ' ':
            speed = 0.0;   turn = 0.0

        car.set_velocity_and_request_state(
            forward=speed, turn=turn,
            headlights=False, leftTurnSignal=False,
            rightTurnSignal=False, brakeSignal=False, reverseSignal=False
        )
        print(f"\r  speed={speed:+.2f}  turn={turn:+.2f}   ", end='', flush=True)

if __name__ == '__main__':
    main()