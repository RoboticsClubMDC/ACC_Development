#!/usr/bin/env python3
"""vo_capture.py — Clean fault status capture for demo recording.

Captures /vo/fault_status strings to a file without the ugly
ExternalShutdownException stack trace that timeout causes.

Usage:
  # Capture for 60 seconds:
  timeout 60s python3 vo_capture.py > /tmp/fault_capture.txt

  # Or run manually and Ctrl+C when done:
  python3 vo_capture.py > /tmp/fault_capture.txt

  # Quick analysis after capture:
  grep -o "flag=[^ ]*" /tmp/fault_capture.txt | sort | uniq -c
  grep "decision=ODOM_SUSPECT" /tmp/fault_capture.txt | wc -l
"""

import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class FaultCapture(Node):

    def __init__(self):
        super().__init__('fault_capture')
        self.create_subscription(
            String, '/vo/fault_status', self._cb, 10)

    def _cb(self, msg):
        print(msg.data, flush=True)


def main():
    rclpy.init()
    node = FaultCapture()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception:
        # Catches ExternalShutdownException from timeout
        # without printing a scary stack trace
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()