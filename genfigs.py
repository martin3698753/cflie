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
p_size=3
#PATH = 'pics/figs/'
PATH = '/home/martin/bak/bak(7)/figs/'

def norm_motor(signal):
    return signal

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
        me = mt.power(path_dir)
        return me, tleft, t
    else:
        print("Undefined sig_type")
        sys.exit(1)

def h_seq_norm(num):
    path_dir = "data/"+num+"/"
    me, tleft, t = load_data(path_dir, 'motor')
    me = me[cutoff:]
    t = t[cutoff:]

    fig = plt.figure(figsize=(8, 6))
    plt.plot(t, me, label=r"$m(t)$")
    plt.xlabel(r'$t(s)$', fontsize=12)
    plt.ylabel(r"$m(t)$", fontsize=12)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(PATH+'h_seq_norm.pdf')
    print('saved picture of h_seq_norm')
    #plt.show()
    plt.close()

def f_seq_norm(num):
    path_dir = "data/"+num+"/"
    _, tleft, t = load_data(path_dir, 'bat')

    fig = plt.figure(figsize=(8, 6))
    plt.plot(t, tleft, label=r"$\tau(t)$")
    plt.xlabel(r'$t(s)$', fontsize=12)
    plt.ylabel(r"$\tau(t)$", fontsize=12)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(PATH+'f_seq_norm.pdf')
    print('saved picture of f_seq_norm')
    #plt.show()
    plt.close()

def f_seq(num):
    path_dir = "data/"+num+"/"
    t, _ = mt.battery(path_dir)
    t = t/1000
    tleft = max(t) - t

    fig = plt.figure(figsize=(8, 6))
    plt.plot(t, tleft, label=r"$\tau(t)$")
    plt.xlabel(r'$t(s)$', fontsize=12)
    plt.ylabel(r"$\tau(t)$", fontsize=12)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(PATH+'f_seq.pdf')
    print('saved picture of f_seq')
    plt.close()

def g_seq(num):
    path_dir = "data/"+num+"/"
    signal, tleft, t = load_data(path_dir, 'bat')

    fig = plt.figure(figsize=(8, 6))
    plt.plot(t, signal, label=r'$u(t)$', color='tab:blue')
    plt.xlabel(r'$t(s)$', fontsize=12)
    plt.ylabel(r'$u(t)$', fontsize=12)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(PATH+'g_seq.pdf')
    print('saved picture of q_seq')
    #plt.show()
    plt.close()

def g_seq_norm(num):
    path_dir = "data/"+num+"/"
    signal, tleft, t = load_data(path_dir, 'bat')
    signal = norm(signal[cutoff:])
    tleft = tleft[cutoff:]
    t = t[cutoff:]

    fig = plt.figure(figsize=(8, 6))
    plt.plot(t, signal, label=r'$u(t)$', color='tab:blue')
    plt.xlabel(r'$t(s)$', fontsize=12)
    plt.ylabel(r'$u(t)$', fontsize=12)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(PATH+'g_seq_norm.pdf')
    print('saved picture of q_seq_norm')
    #plt.show()
    plt.close()


def relu():
    x = np.linspace(-5, 5, 1000)
    y = np.maximum(0, x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=r'ReLU: $f(x) = \max(0, x)$', linewidth=2)

    plt.xlabel('x', fontsize=14)
    plt.ylabel('f(x)', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)

    plt.savefig(PATH+"relu.pdf")
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

    plt.savefig(PATH+"sigmoid.pdf")
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

    plt.savefig(PATH+"tanh.pdf")
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
    #plt.scatter(t, me, s=p_size, label=r"$\tilde{h_t}$")
    path_dir = "data/"+num+"/"
    signal, tleft, t = load_data(path_dir, sig_type)
    signal = norm(signal[cutoff:])
    tleft = tleft[cutoff:]
    t = t[cutoff:]

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, signal, label=r"$u(t)$")
    ax1.plot(t, tleft, label=r"$\tau (t)$", color='tab:orange')
    ax1.set_xlabel(r'$\text{Čas } t(s)$', fontsize=12)
    ax1.axvspan(t[start], t[start+n], color='black', alpha=0.2, label=r'$\text{délka okna } n$')
    ax1.legend(fontsize=16)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t[start:start+n], signal[start:start+n], label=r"$u(t)$")
    ax2.set_xlabel(r'$\text{Čas } t(s)$', fontsize=12)
    ax2.legend(fontsize=14)
    ax2.set_title("Napětí baterie")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t[start:start+n], tleft[start:start+n], label=r"$\tau (t)$", color='tab:orange')
    ax3.set_xlabel(r'$\text{Čas } t(s)$', fontsize=12)
    ax3.legend(fontsize=14)
    ax3.set_title("Čas do vybití")

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
    plt.savefig(PATH+"window.pdf")
    print("saved picture of making window")
    #plt.show()
    plt.close()

def window2(num, n, start, sig_type):
    #plt.scatter(t, me, s=p_size, label=r"$\tilde{h_t}$")
    path_dir = "data/"+num+"/"
    signal, tleft, t = load_data(path_dir, sig_type)
    signal = norm(signal[cutoff:])
    tleft = tleft[cutoff:]
    t = t[cutoff:]

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, signal, label=r"$u(t)$")
    ax1.set_xlabel('Čas t(s)', fontsize=12)
    ax1.axvspan(t[start], t[start+n], color='black', alpha=0.2, label=r'$\text{délka okna } n$')
    ax1.axvspan(t[start+n], t[start+2*n], color='black', alpha=0.2)
    ax1.legend(fontsize=16)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t[start:start+n], signal[start:start+n], label=r"$u(t,t+n\Delta t)$")
    ax2.set_xlabel('Čas t(s)', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.set_title("Současné napětí baterie")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t[start:start+n], signal[start+n:start+2*n], label=r"$u(t+n\Delta t,t+2n\Delta t)$", color='tab:orange')
    ax3.set_xlabel('Čas t(s)', fontsize=12)
    ax3.legend(fontsize=12)
    ax3.set_title("Budoucí napětí baterie")

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
        (t[start+n], ax3.get_ylim()[0]),  # Bottom-left corner
        t[start + 2*n]-t[start+n],  # Width
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
        xy=(t[start + 2*n-(round(n/2))], signal[start+n] - (ax3.get_ylim()[1] - ax3.get_ylim()[0])),
        xytext=(0.5, 0.5),
        textcoords=ax3.transAxes,
        arrowprops=arrowprops,
        xycoords='data',
    )
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig(PATH+"window2.pdf")
    print("saved picture of making window2")
    #plt.show()
    plt.close()

def motor_graph(num):
    path_dir = "data/"+num+"/"
    motor = mt.readcsv(path_dir+'motor.csv')
    motor = (motor/65535)
    t = mt.time(path_dir)
    t = t/1000
    power = mt.power(path_dir)

    fig = plt.figure(figsize=(8, 4))
    plt.plot(t, motor[1], label=r"Motor $m_1 (\%)$")
    plt.plot(t, motor[2], label=r"Motor $m_2 (\%)$")
    plt.plot(t, motor[3], label=r"Motor $m_3 (\%)$")
    plt.plot(t, motor[4], label=r"Motor $m_4 (\%)$")
    plt.xlabel(r'$\text{Čas } t(s)$', fontsize=12)
    plt.ylabel(r'Signály PWM z motorů (%)', fontsize=12)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    print(f"saving figure motors {num}")
    plt.savefig(PATH+num+"_motors.pdf")
    #plt.show()
    plt.close()

    fig = plt.figure(figsize=(8, 4))
    plt.plot(t, power, label=r"$m (%)$")
    plt.xlabel('čas t(s)', fontsize=12)
    plt.ylabel('Průměr signálu PWM motorů (%)', fontsize=12)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    print(f"saving figure motors average {num}")
    plt.savefig(PATH+num+"_motors_avg.pdf")
    #plt.show()
    plt.close

def pos(num):
    path_dir = "data/"+num+"/"
    x, y, z = mt.position_graph(path_dir)
    t = mt.time(path_dir)
    t = t/1000

    fig = plt.figure(figsize=(8, 3))
    plt.plot(t, x, label="Trajektorie na ose x", color="purple")
    plt.plot(t, y, label="Trajektorie na ose y", color="blue")
    plt.plot(t, z, label="Trajektorie na ose z", color="red")
    plt.xlabel('čas t(s)', fontsize=12)
    plt.ylabel('Vzdálenost od počátku (m)', fontsize=12)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True)
    plt.tight_layout()

    #save
    print(f"saving figure position {num}")
    plt.savefig(PATH+num+"_pos.pdf")
    #plt.show()
    plt.close()

def gen(num):
    path_dir = "data/"+num+"/"
    battery = mt.battery(path_dir)
    t = mt.time(path_dir)
    me = mt.power(path_dir)
    mech = np.sum(me)

    t = t/1000
    battery[0] = battery[0]/1000

    dist = 0
    try:
        dist = mt.position(path_dir)
    except:
        pass

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))

    ax1.plot(t, me, label=r'funkce $m$ v $(\%)$')
    ax1.set_xlabel('čas t(s)', fontsize=12)
    ax1.set_ylabel(r'Aritmetický průměr motorů $(\%)$', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # Plot the second dataset
    ax2.plot(battery[0], battery[1], label=r'Napětí baterie $u(V)$')
    ax2.set_xlabel('čas t(s)', fontsize=12)
    ax2.set_ylabel('Napětí (V)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True)

    cas = t[-1]
    text = (
        f"{'Čas letu[s]':<20} {cas:>10.2f}\n"
        #f"{'Celková energie[J]':<20} {mech:>10.2f}\n"
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
    plt.savefig(PATH+num+".pdf")
    #plt.show()
    plt.close()

    if(dist != 0):
        pos(num)

if __name__ == '__main__':
    gen('23-1-25')
    gen('24-1-25')
    gen('31-1-25')
    gen('4-2-25')
    gen('5-2-25')
    gen('9-4-25')
    gen('11-4-25')
    motor_graph('5-2-25')
    gen('21-2-25')
    gen('8-4-25')
    #linear('31-1-25', 'bat')
    f_seq('31-1-25')
    f_seq_norm('31-1-25')
    g_seq('31-1-25')
    g_seq_norm('31-1-25')
    h_seq_norm('31-1-25')
    # linear_norm('5-2-25', 'bat')
    window('5-2-25', 300, 2000, 'bat') # n indicate window size, start is starting position of that window, sig_type can be 'bat' or 'motor'
    window2('5-2-25', 300, 2000, 'bat')
    # reg('5-2-25', 300, 1000, 'bat')
    # relu()
    # sigmoid()
    # tanh()
    # experiment_battery_real_time('6-3-25')
