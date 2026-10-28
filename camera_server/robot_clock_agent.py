#!/usr/bin/env python3
"""SSH-side clock protocol used by set_g1_clock.py."""

import json
import sys
import time


def reply(value):
    print(json.dumps(value), flush=True)


def main():
    allow_set = len(sys.argv) > 1 and sys.argv[1] == 'set'
    reply({'ready': True})
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except ValueError:
            # sudo may leave the supplied password on stdin when cached.
            continue
        if not isinstance(request, dict):
            continue
        action = request.get('action')
        if action == 'probe':
            reply({'unix_ns': time.time_ns()})
        elif action == 'set' and allow_set:
            value = request.get('unix_ns')
            if isinstance(value, bool) or not isinstance(value, int) or not 1577836800000000000 <= value < 4102444800000000000:
                raise ValueError('Invalid proposed system time.')
            time.clock_settime_ns(time.CLOCK_REALTIME, value)
            reply({'set': True})
        elif action == 'quit':
            return
        else:
            raise ValueError('Invalid or unauthorized clock command.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f'G1 clock error: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
