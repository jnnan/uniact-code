import struct
import math

class Button:
    def __init__(self):
        self.pressed = False
        self.on_press = False
        self.on_release = False

    def update(self, state):
        self.on_press = state and not self.pressed
        self.on_release = not state and self.pressed
        self.pressed = state


class Gamepad:
    def __init__(self, smooth=0.03, dead_zone=0.01):
        self.lx = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.l2 = 0.0
        self.ly = 0.0

        self.smooth = smooth
        self.dead_zone = dead_zone

        self.R1 = Button()
        self.L1 = Button()
        self.start = Button()
        self.select = Button()
        self.R2 = Button()
        self.L2 = Button()
        self.F1 = Button()
        self.F2 = Button()
        self.A = Button()
        self.B = Button()
        self.X = Button()
        self.Y = Button()
        self.up = Button()
        self.right = Button()
        self.down = Button()
        self.left = Button()

    def update(self, key_data):
        self.lx = self.lx * (1 - self.smooth) + (0.0 if abs(key_data['lx']) < self.dead_zone else key_data['lx']) * self.smooth
        self.rx = self.rx * (1 - self.smooth) + (0.0 if abs(key_data['rx']) < self.dead_zone else key_data['rx']) * self.smooth
        self.ry = self.ry * (1 - self.smooth) + (0.0 if abs(key_data['ry']) < self.dead_zone else key_data['ry']) * self.smooth
        self.l2 = self.l2 * (1 - self.smooth) + (0.0 if abs(key_data['l2']) < self.dead_zone else key_data['l2']) * self.smooth
        self.ly = self.ly * (1 - self.smooth) + (0.0 if abs(key_data['ly']) < self.dead_zone else key_data['ly']) * self.smooth

        self.R1.update(key_data['R1'])
        self.L1.update(key_data['L1'])
        self.start.update(key_data['start'])
        self.select.update(key_data['select'])
        self.R2.update(key_data['R2'])
        self.L2.update(key_data['L2'])
        self.F1.update(key_data['F1'])
        self.F2.update(key_data['F2'])
        self.A.update(key_data['A'])
        self.B.update(key_data['B'])
        self.X.update(key_data['X'])
        self.Y.update(key_data['Y'])
        self.up.update(key_data['up'])
        self.right.update(key_data['right'])
        self.down.update(key_data['down'])
        self.left.update(key_data['left'])


def parse_remote_data(buffer):
    """
    Parses a 40-byte buffer into the structured gamepad data.
    :param buffer: A 40-byte array (bytes-like object).
    :return: A dictionary with parsed gamepad data.
    """
    # Unpack the first 24 bytes: 2 bytes for head, 2 bytes for buttons, 5 floats (20 bytes)
    head, btn_value, lx, rx, ry, l2, ly = struct.unpack('<2sH5f', buffer[:24])

    # Convert buttons from a 16-bit integer to individual flags
    buttons = {
        'R1': bool(btn_value & (1 << 0)),
        'L1': bool(btn_value & (1 << 1)),
        'start': bool(btn_value & (1 << 2)),
        'select': bool(btn_value & (1 << 3)),
        'R2': bool(btn_value & (1 << 4)),
        'L2': bool(btn_value & (1 << 5)),
        'F1': bool(btn_value & (1 << 6)),
        'F2': bool(btn_value & (1 << 7)),
        'A': bool(btn_value & (1 << 8)),
        'B': bool(btn_value & (1 << 9)),
        'X': bool(btn_value & (1 << 10)),
        'Y': bool(btn_value & (1 << 11)),
        'up': bool(btn_value & (1 << 12)),
        'right': bool(btn_value & (1 << 13)),
        'down': bool(btn_value & (1 << 14)),
        'left': bool(btn_value & (1 << 15)),
    }

    # Include analog stick data
    buttons.update({'lx': lx, 'rx': rx, 'ry': ry, 'l2': l2, 'ly': ly})

    return buttons


# Usage example
if __name__ == '__main__':
    # Example 40-byte buffer (mocked data)
    mock_buffer = bytes([0xAA, 0xBB, 0x01, 0x02] + [0x00] * 36)

    # Parse the buffer
    gamepad = Gamepad()
    parsed_data = parse_remote_data(mock_buffer)
    gamepad.update(parsed_data)

    # Print the parsed values
    print("Parsed Gamepad State:")
    print(f"LX: {gamepad.lx}, RX: {gamepad.rx}, RY: {gamepad.ry}, L2: {gamepad.l2}, LY: {gamepad.ly}")
    print(f"Button A pressed: {gamepad.A.pressed}, Button B pressed: {gamepad.B.pressed}")
