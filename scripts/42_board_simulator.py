# -*- coding: utf-8 -*-
"""42_board_simulator.py —— 采集板卡模拟器（无真 PCB 也能跑通全流程）。

按板卡协议 dgap-board/v1 周期发送 JSON：温度/湿度/振动/GPS/杆塔号。
支持 --stdout（打印）或 --mqtt（发到 broker）。
运行：.venv/Scripts/python.exe scripts/42_board_simulator.py --stdout
"""
import argparse, json, random, time

PROTO_VERSION = "dgap-board/v1"


def sample(seq, tower="T102", line="10kV 明珠线"):
    return {
        "ver": PROTO_VERSION,
        "ts": time.time(),
        "seq": seq,
        "line": line,
        "tower": tower,
        "gps": {"lat": round(31.23 + random.random() * 0.01, 5),
                "lng": round(121.47 + random.random() * 0.01, 5)},
        "thermal": {"grid_avg": round(random.uniform(25, 45), 1),
                    "max": round(random.uniform(30, 60), 1)},
        "humidity": round(random.uniform(40, 80), 1),
        "vibration": round(random.uniform(0, 0.5), 3),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stdout", action="store_true", help="打印 JSON（默认行为）")
    p.add_argument("--mqtt", action="store_true", help="发到 MQTT broker")
    p.add_argument("--broker", default="localhost")
    p.add_argument("--topic", default="dgap/board")
    p.add_argument("--interval", type=float, default=2.0)
    args = p.parse_args()

    seq = 0
    client = None
    if args.mqtt:
        import paho.mqtt.client as mqtt
        client = mqtt.Client()
        client.connect(args.broker)
    while True:
        line = json.dumps(sample(seq), ensure_ascii=False)
        if client:
            client.publish(args.topic, line)
        print(line, flush=True)
        seq += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
