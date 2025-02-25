import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset, zoomed_inset_axes
from matplotlib import gridspec
import numpy as np
import math
import os
import sys
import glob
import pandas as pd
import pickdir
import maketab as mt

def load_data(path_dir, sig_type):
    if sig_type == "bat":
        t, signal = mt.battery(path_dir)
        t = t/1000
        tleft = 1 - t / max(t)
        return signal, tleft, t
    elif sig_type == "motor":
        t = mt.time(path_dir)
        t = t/1000
        tleft = 1 - t / max(t)
        motor = mt.readcsv(path_dir+'motor.csv')
        motor = (motor/65535)*100
        thr = mt.thrust(path_dir)
        av = mt.ang_vel(path_dir)
        me = ((thr[1]/4)*av[1] + (thr[2]/4)*av[2] + (thr[3]/4)*av[3] + (thr[4]/4)*av[4])*0.047*0.1*0.05
        return me, tleft, t
    else:
        print("Undefined sig_type")
        sys.exit(1)

def norm(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def denorm(normalized_data, original_min, original_max):
    denormalized_data = normalized_data * (original_max - original_min) + original_min
    return denormalized_data

def reg(num, n, start, sig_type):
    path_dir = "data/"+num+"/"
    signal, tleft, t = np.array(load_data(path_dir, sig_type))[:, start:start+n]

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

    ax1.plot(t, signal, label='g(t)', color='tab:blue')
    ax1.set_xlabel('Čas t(s)', fontsize=10)
    slope2, slope, intercept = np.polyfit(t, signal, 2)
    mean = np.mean(signal)
    std = np.std(signal)
    ax1.plot(t, slope2*t**2 + slope*t + intercept, color="tab:green")
    ax1.legend()

    ax2.plot(t, tleft, label='f(t)', color='tab:orange')
    ax2.set_xlabel('Čas t(s)', fontsize=10)
    ax2.legend()

    plt.show()

def linear(num, n, start, sig_type):
    path_dir = "data/"+num+"/"
    signal, tleft, t = load_data(path_dir, sig_type)

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, signal, label='g(t)')
    ax1.plot(t, tleft, label='f(t)', color='tab:orangeorange')
    ax1.set_xlabel('Čas t(s)', fontsize=10)
    ax1.axvspan(t[start], t[start+n], color='black', alpha=0.1)
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t[start:start+n], signal[start:start+n], label='g(t)')
    ax2.set_xlabel('Čas t(s)', fontsize=10)
    ax2.legend()
    ax2.set_title("Napětí na baterii")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t[start:start+n], tleft[start:start+n], label='f(t)', color='tab:orangeorange')
    ax3.set_xlabel('Čas t(s)', fontsize=10)
    ax3.legend()
    ax3.set_title("Jednoduchá linearní čára")

    rect1 = Rectangle(
        (t[1000], ax2.get_ylim()[0]),  # Bottom-left corner
        t[1000 + n]-t[1000],  # Width
        ax2.get_ylim()[1] - ax2.get_ylim()[0],  # Height
        edgecolor='black',
        facecolor='none',
        linestyle='--',
        linewidth=1,
        label='Zoomed Region'
    )
    rect2 = Rectangle(
        (t[1000], ax3.get_ylim()[0]),  # Bottom-left corner
        t[1000 + n]-t[1000],  # Width
        ax3.get_ylim()[1] - ax3.get_ylim()[0],  # Height
        edgecolor='black',
        facecolor='none',
        linestyle='--',
        linewidth=1,
        label='Zoomed Region'
    )
    ax1.add_patch(rect1)
    ax1.add_patch(rect2)
    arrowprops = dict(arrowstyle="->", color="black", linewidth=1, shrinkA=0, shrinkB=0, linestyle=":")
    ax1.annotate(
        '',
        xy=(t[1000 + n // 2], signal[1000] - (ax3.get_ylim()[1] - ax3.get_ylim()[0])),
        xytext=(0.5, 0.5),
        textcoords=ax2.transAxes,
        arrowprops=arrowprops,
        xycoords='data',
    )
    ax1.annotate(
        '',  # No text, just an arrow
        xy=(t[1000 + n // 2], tleft[1000] - (ax3.get_ylim()[1] - ax3.get_ylim()[0])),
        xytext=(0.5, 0.5),
        textcoords=ax3.transAxes,
        arrowprops=arrowprops,
        xycoords='data',
    )
    plt.tight_layout()
    #plt.savefig("pics/figs/window.pdf")
    print("saved picture of making window")
    plt.show()
    plt.close()


def gen(num):
    path_dir = "data/"+num+"/"
    battery = mt.battery(path_dir)
    t = mt.time(path_dir)
    motor = mt.readcsv(path_dir+'motor.csv')
    motor = (motor/65535)*100
    thr = mt.thrust(path_dir)
    av = mt.ang_vel(path_dir)
    me = ((thr[1]/4)*av[1] + (thr[2]/4)*av[2] + (thr[3]/4)*av[3] + (thr[4]/4)*av[4])*0.047*0.1*0.05
    mech = np.sum(me)

    t = t/1000
    battery[0] = battery[0]/1000

    dist = 0
    try:
        dist = mt.position(path_dir)
    except:
        pass

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

    ax1.plot(t, me, label='Výkon motorů (W)')
    ax1.set_xlabel('čas t(s)', fontsize=12)
    ax1.set_ylabel('Výkon (W)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # Plot the second dataset
    ax2.plot(battery[0], battery[1], label='Napětí na baterii (V)')
    ax2.set_xlabel('čas t(s)', fontsize=12)
    ax2.set_ylabel('Napětí (V)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True)

    cas = t[-1]
    text = (
        f"{'Čas letu[s]':<20} {cas:>10.2f}\n"
        f"{'Celková energie[J]':<20} {mech:>10.2f}\n"
        f"{'Uletěno[m]':<20} {dist:>10.2f}\n"
    )
    ax2.text(
        0.02, 0.2,
        text,
        transform=ax2.transAxes,
        fontsize=11,
        fontfamily="monospace",  # Use a monospaced font
        verticalalignment="center",  # Align text vertically
        horizontalalignment="left",  # Align text horizontally
        bbox=dict(facecolor="lightgray", alpha=0.8, edgecolor="black"),  # Add a background box
    )

    #plt.title(num)
    # Adjust layout to prevent overlap
    plt.tight_layout()

    #save
    print(f"saving figure {num}")
    plt.savefig("pics/figs/"+num+".pdf")
    plt.close()

    # Show the plot
    plt.show()

if __name__ == '__main__':
    # gen('23-1-25')
    # gen('24-1-25')
    # gen('31-1-25')
    # gen('4-2-25')
    # gen('5-2-25')
    # gen('21-2-25')
    # linear('5-2-25', 300, 1000, 'bat') # n indicate window size, start is starting position of that window, sig_type can be 'bat' or 'motor'
    reg('5-2-25', 300, 1000, 'bat')
