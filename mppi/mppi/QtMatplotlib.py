# QtMatplotlib.py
import matplotlib.pyplot as plt

class QtPlotter:
    def __init__(self):
        plt.ion()
        self.fig, self.axs = plt.subplots(1, 1)
        self.plots = {}

    def scatter(self, x, y, c='b', s=10, plot_num=0, live=True):
        if plot_num not in self.plots:
            self.plots[plot_num] = self.axs.scatter(x, y, c=c, s=s)
        else:
            self.plots[plot_num].remove()
            self.plots[plot_num] = self.axs.scatter(x, y, c=c, s=s)
        if live:
            self.axs.set_title(f"Plot {plot_num}")
            plt.pause(0.001)
