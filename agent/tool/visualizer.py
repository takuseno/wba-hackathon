from builtins import input
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import subprocess
import argparse
from collections import deque


class AnimatedLineGraph:
    def __init__(self, row, column, max_val=1.0):
        self.row = row
        self.column = column
        self.proc = subprocess.Popen(
            [
                'python', '-u', __file__,
                '--row', str(row),
                '--column', str(column),
                '--max', str(max_val)
            ],
            stdin=subprocess.PIPE)

    def update(self, values):
        data = {
            'values': values,
            # 'hilights': hilights
        }
        self.proc.stdin.write(bytearray(json.dumps(data) + '\n', 'utf-8'))
        self.proc.stdin.flush()


if __name__ == '__main__':
    # get argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--row', type=int, default=1)
    parser.add_argument('--column', type=int, default=1)
    parser.add_argument('--max', type=float, default=1.0)
    args = parser.parse_args()

    # initialize graph space
    # range -5 to 5
    queue = deque(np.zeros(50), maxlen=50)  # TODO: remove magic
    x = np.arange(-50, 0)

    fig = plt.figure(figsize=(10, 6))
    plt.ylabel('V(s)')
    plt.xlabel('t(steps)')
    plt.grid(True)
    plt.ylim((-5, 5))

    def draw(i):
        plt.cla()  # clear graph
        # load data from input
        data = json.loads(input())
        value = data['values'][0]
        # hilight = data['hilight']

        print(value)
        queue.append(value)
        y = np.array(list(queue))

        plt.plot(x, y, "r")
        # plt.title(fig_title+'i='+str(i))

    ani = animation.FuncAnimation(fig, draw, interval=1)
    plt.show()
