"""How many cores do we really get? Time n concurrent pure-python busy loops (one per process)."""

import multiprocessing as mp
import sys
import time


def burn(_):
    t = time.perf_counter()
    x = 0
    for i in range(20_000_000):
        x += i
    return time.perf_counter() - t


if __name__ == "__main__":
    for n in (1, 2, 4, 8):
        with mp.Pool(n) as p:
            ts = p.map(burn, range(n))
        print(n, [round(t, 2) for t in ts], flush=True)
