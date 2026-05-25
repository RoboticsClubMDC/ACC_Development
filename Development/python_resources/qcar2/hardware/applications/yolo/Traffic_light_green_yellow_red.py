import argparse
import threading
import time
import urllib.request
from urllib.error import HTTPError, URLError


DEFAULT_IP = "192.168.2.38"


def call_traffic_light(light_ip, endpoint, timeout_s=4):
    url = f"http://{light_ip}:5000/{endpoint}"
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return response.read().decode("utf-8")


def safe_call(light_ip, endpoint, timeout_s=4):
    try:
        response = call_traffic_light(light_ip, endpoint, timeout_s=timeout_s)
        print(f"{endpoint}: {response}")
        return response
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"{endpoint}: request failed: {exc}")
        return None


def reclaim_direct_control(light_ip):
    safe_call(light_ip, "close_stream")
    time.sleep(0.2)
    safe_call(light_ip, "immediate/off", timeout_s=1)
    time.sleep(0.2)


def hold_with_stop(duration_s, stop_event=None):
    if duration_s <= 0:
        return True

    if stop_event is None:
        time.sleep(duration_s)
        return True

    deadline = time.monotonic() + duration_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        if stop_event.wait(min(0.1, remaining)):
            return False


def run_traffic_light_loop(
    light_ip,
    red=30.0,
    green=30.0,
    yellow=3.0,
    cycles=0,
    leave_on=False,
    stop_event=None,
):
    print(f"Using traffic light at {light_ip}")

    reclaim_direct_control(light_ip)

    sequence = (
        ("red", red),
        ("green", green),
        ("yellow", yellow),
    )

    cycle_index = 0
    try:
        while cycles <= 0 or cycle_index < cycles:
            if stop_event is not None and stop_event.is_set():
                break

            cycle_index += 1
            print(f"Starting cycle {cycle_index}")
            for color, hold_time in sequence:
                if stop_event is not None and stop_event.is_set():
                    return
                if safe_call(light_ip, f"immediate/{color}", timeout_s=1) is None:
                    return
                status = safe_call(light_ip, "status", timeout_s=1)
                print(f"status after {color}: {status}")
                if not hold_with_stop(hold_time, stop_event=stop_event):
                    return
    finally:
        if not leave_on:
            safe_call(light_ip, "immediate/off", timeout_s=1)


def main():
    parser = argparse.ArgumentParser(
        description="Run a Quanser traffic light in a normal red-green-yellow loop."
    )
    parser.add_argument("--ip", default=DEFAULT_IP, help="Traffic light IP address.")
    parser.add_argument(
        "--red",
        type=float,
        default=30.0,
        help="Seconds to hold red.",
    )
    parser.add_argument(
        "--green",
        type=float,
        default=30.0,
        help="Seconds to hold green.",
    )
    parser.add_argument(
        "--yellow",
        type=float,
        default=3.0,
        help="Seconds to hold yellow.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of cycles to run. Use 0 to loop until Ctrl+C.",
    )
    parser.add_argument(
        "--leave-on",
        action="store_true",
        help="Leave the light on its last color instead of turning it off at the end.",
    )
    args = parser.parse_args()

    try:
        run_traffic_light_loop(
            light_ip=args.ip,
            red=args.red,
            green=args.green,
            yellow=args.yellow,
            cycles=args.cycles,
            leave_on=args.leave_on,
            stop_event=threading.Event(),
        )
    except KeyboardInterrupt:
        print("Stopped by user.")


if __name__ == "__main__":
    main()
