#!/usr/bin/env python3
"""Eliza-Claw CLI -- [CLI / Input] on top of the microkernel.

  python -m elizaclaw            interactive loop with heartbeat thread
  python -m elizaclaw --demo     scripted conversation proving the pipeline
  python -m elizaclaw --once "text"   single turn (for tests/scripts)

The heartbeat is a real daemon thread ticking every config.heartbeat_interval
seconds; plugins wake up, check their context (JSON memory), and act.
"""
import argparse
import sys
import threading
import time

from elizaclaw.shell import Shell


def run_shell(shell):
    print("eliza-claw v%s  ::  zero tokens, one loop" %
          shell.cfg["meta"]["version"])
    print(shell.cfg["config"]["initial"])
    stop = threading.Event()

    def beat():
        while not stop.wait(shell.cfg["config"].get("heartbeat_interval", 1.0)):
            for msg in shell.tick():
                print("\n[heartbeat] %s\n> " % msg, end="", flush=True)

    hb = threading.Thread(target=beat, daemon=True)
    hb.start()
    try:
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break
            if not text:
                continue
            reply, done = shell.step(text)
            print(reply)
            for msg in shell.flush_queue():
                print("[heartbeat] " + msg)
            if done:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        print("\n" + shell.cfg["config"]["final"])


DEMO = [
    "hello",
    "my name is Alice",
    "i live in Seattle",
    "I'm feeling frustrated about my code",
    "I need to debug the authentication module",
    "there is an error in my script",
    "remind me to submit the report at 5pm",
    "what's the weather in Seattle",
    "search for python context engineering",
    "I am done with debug the authentication module",
    "please help me finish the project",
    "i think we are going nowhere",
    "i think we are going nowhere",
    "i think we are going nowhere",
    "list my reminders",
    "goodbye",
]


def demo(shell):
    for line in DEMO:
        print("\n> " + line)
        reply, done = shell.step(line)
        print(":: " + reply)
        for msg in shell.flush_queue():
            print(":: [heartbeat] " + msg)
        if done:
            break


def main(argv=None):
    ap = argparse.ArgumentParser(prog="eliza-claw")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--once", metavar="TEXT")
    ap.add_argument("--config", default="eliza.json")
    args = ap.parse_args(argv)
    shell = Shell(args.config)
    if args.once:
        print(shell.step(args.once)[0])
    elif args.demo:
        demo(shell)
    else:
        run_shell(shell)


if __name__ == "__main__":
    main()
