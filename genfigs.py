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

cutoff = 50
sec_norm = 410

def norm(signal):
    norm_lower = 2
    norm_upper = 4
    signal = (signal - norm_lower) / (norm_upper - norm_lower)
    return signal

plt.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern font for math text

def load_data(path_dir, sig_type):
    if sig_type == "bat":
        t, signal = mt.battery(path_dir)
        t = t/1000
        tleft = 1 - t / max(t)
        tleft = tleft*(t[-1]/sec_norm)
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


def relu():
    x = np.linspace(-5, 5, 1000)
    y = np.maximum(0, x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=r'ReLU: $f(x) = \max(0, x)$', linewidth=2)

    plt.xlabel('x', fontsize=14)
    plt.ylabel('f(x)', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)

    plt.savefig("pics/figs/relu.pdf")
    print("saved relu function")
    #plt.show()
    plt.close

def sigmoid():
    x = np.linspace(-5, 5, 1000)
    y = 1 / (1 + np.exp(-x))  # Sigmoid function

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=r'Sigmoid: $f(x) = \frac{1}{1 + e^{-x}}$', linewidth=2)

    plt.xlabel('x', fontsize=14)
    plt.ylabel('f(x)', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)

    plt.savefig("pics/figs/sigmoid.pdf")
    print("saved sigmoid function")
    #plt.show()
    plt.close()

def tanh():
    x = np.linspace(-5, 5, 1000)
    y = np.tanh(x)  # Tanh function

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=r'Tanh: $f(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$', linewidth=2)

    plt.xlabel('x', fontsize=14)
    plt.ylabel('f(x)', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)

    plt.savefig("pics/figs/tanh.pdf")
    print("saved function tanh")
    #plt.show()
    plt.close()

def experiment_battery_real_time(num):
    path_dir = "data/"+num+"/"
    battery = mt.battery(path_dir)
    t = np.arange(0,battery.shape[1]*100, 100)*0.1
    pred = mt.prediction(path_dir)

    plt.figure(figsize=(8, 6))
    plt.plot(t[:len(pred)], pred, label='predikce kapacity', color='tab:green')
    plt.plot(t, norm(battery[1]), label='baterie (V)', color='tab:blue')
    plt.xlabel("čas t(s)", fontsize=12)
    plt.legend()

    plt.savefig('pics/figs/experiment_battery_real_time.pdf')
    print('saved fig of battery prediction in real time')
    plt.close
    #plt.show()

def reg(num, n, start, sig_type):
    path_dir = "data/"+num+"/"
    signal, tleft, t = load_data(path_dir, sig_type)
    signal = norm(signal[start:start+n])
    tleft = tleft[start:start+n]
    t = np.arange(0, len(signal))

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

    ax1.plot(t, signal, label=r'$g_i(\Delta t)$', color='tab:blue')
    ax1.set_xlabel(r'$\Delta t$', fontsize=10)
    slope, intercept = np.polyfit(t, signal, 1)
    r_t = slope*t + intercept
    detrended_g_t = signal - r_t
    std_detrended = np.std(detrended_g_t)
    mean = np.mean(signal)
    ax1.plot(t, slope*t + intercept, color="tab:green", label=r"$r_1(\Delta t)$")
    ax1.axhline(mean, color='black', linestyle='--', label=r'$\overline{g_i}$')
    ax1.fill_between(t, r_t - std_detrended, r_t + std_detrended, color="orange", alpha=0.3, label=r"$\sigma^2_{gi}$")
    ax1.legend(fontsize=12)
    ax1.grid(True)

    ax2.plot(t, tleft, label=r'$f_i(\Delta t)$', color='tab:orange')
    ax2.set_xlabel(r'$\Delta t$', fontsize=10)
    slope, intercept = np.polyfit(t, tleft, 1)
    r_t = slope*t + intercept
    detrended_g_t = tleft - r_t
    std_detrended = np.std(detrended_g_t)
    mean = np.mean(signal)
    ax2.plot(t, slope*t + intercept, color="tab:green", label=r"$r_1(\Delta t)$")
    ax2.axhline(mean, color='black', linestyle='--', label=r'$\overline{f_i}$')
    ax2.fill_between(t, r_t - std_detrended, r_t + std_detrended, color="orange", alpha=0.3, label=r"$\sigma^2_{fi}$")
    ax2.legend(fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('pics/figs/reg1.pdf')
    print("saved a picture of regression 1")
    #plt.show()
    plt.close()

def linear(num, sig_type):
    path_dir = "data/"+num+"/"
    signal, tleft, t = load_data(path_dir, sig_type)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(t, signal, label=r"$g(t)'$", color='tab:blue')
    plt.plot(t, tleft, label=r"$f(t)$", color='tab:orange')
    plt.xlabel('t')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.grid(True)
    #plt.savefig('pics/figs/linear.pdf')
    print('saved picture of linear')
    plt.show()
    plt.close()


def linear_norm(num, sig_type):
    path_dir = "data/"+num+"/"
    signal, tleft, t = load_data(path_dir, sig_type)
    signal = norm(signal[cutoff:])
    tleft = tleft[cutoff:]
    t = t[cutoff:]

    fig = plt.figure(figsize=(8, 6))
    plt.plot(t, signal, label='g(t)', color='tab:blue')
    plt.plot(t, tleft, label='f(t)', color='tab:orange')
    plt.xlabel('t')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('pics/figs/linear_norm.pdf')
    print('saved picture of linear_norm')
    #plt.show()
    plt.close()

def window(num, n, start, sig_type):
    path_dir = "data/"+num+"/"
    signal, tleft, t = load_data(path_dir, sig_type)
    signal = norm(signal[cutoff:])
    tleft = tleft[cutoff:]
    t = t[cutoff:]

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, signal, label='g(t)')
    ax1.plot(t, tleft, label='f(t)', color='tab:orange')
    ax1.set_xlabel('Čas t(s)', fontsize=10)
    ax1.axvspan(t[start], t[start+n], color='black', alpha=0.1)
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t[start:start+n], signal[start:start+n], label='g(t)')
    ax2.set_xlabel('Čas t(s)', fontsize=10)
    ax2.legend()
    ax2.set_title("Napětí na baterii")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t[start:start+n], tleft[start:start+n], label='f(t)', color='tab:orange')
    ax3.set_xlabel('Čas t(s)', fontsize=10)
    ax3.legend()
    ax3.set_title("Jednoduchá linearní čára")

    rect1 = Rectangle(
        (t[start], ax2.get_ylim()[0]),  # Bottom-left corner
        t[start + n]-t[start],  # Width
        ax2.get_ylim()[1] - ax2.get_ylim()[0],  # Height
        edgecolor='black',
        facecolor='grey',
        alpha=0.5,
        linestyle='--',
        linewidth=1,
        label='Zoomed Region'
    )
    rect2 = Rectangle(
        (t[start], ax3.get_ylim()[0]),  # Bottom-left corner
        t[start + n]-t[start],  # Width
        ax3.get_ylim()[1] - ax3.get_ylim()[0],  # Height
        edgecolor='black',
        facecolor='grey',
        alpha=0.5,
        linestyle='--',
        linewidth=1,
        label='Zoomed Region'
    )
    ax1.add_patch(rect1)
    ax1.add_patch(rect2)
    arrowprops = dict(arrowstyle="->", color="black", linewidth=1, shrinkA=0, shrinkB=0, linestyle=":")
    ax1.annotate(
        '',
        xy=(t[start + n // 2], signal[start] - (ax2.get_ylim()[1] - ax2.get_ylim()[0])),
        xytext=(0.5, 0.5),
        textcoords=ax2.transAxes,
        arrowprops=arrowprops,
        xycoords='data',
    )
    ax1.annotate(
        '',  # No text, just an arrow
        xy=(t[start + n // 2], tleft[start] - (ax3.get_ylim()[1] - ax3.get_ylim()[0])),
        xytext=(0.5, 0.5),
        textcoords=ax3.transAxes,
        arrowprops=arrowprops,
        xycoords='data',
    )
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig("pics/figs/window.pdf")
    print("saved picture of making window")
    #plt.show()
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
    #plt.show()

if __name__ == '__main__':
    # gen('23-1-25')
    # gen('24-1-25')
    # gen('31-1-25')
    # gen('4-2-25')
    # gen('5-2-25')
    # gen('21-2-25')
    linear('31-1-25', 'bat')
    # linear_norm('5-2-25', 'bat')
    # window('5-2-25', 300, 2000, 'bat') # n indicate window size, start is starting position of that window, sig_type can be 'bat' or 'motor'
    # reg('5-2-25', 300, 1000, 'bat')
    # relu()
    # sigmoid()
    # tanh()
    # experiment_battery_real_time('6-3-25')
