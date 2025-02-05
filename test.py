import matplotlib.pyplot as plt
import numpy as np
import maketab as mt

if __name__ == '__main__':
    path_dir = 'data/4-2-25/'
    battery = mt.battery(path_dir)
    t = mt.time(path_dir)


    motor = mt.readcsv(path_dir+'motor.csv')
    motor = (motor/65535)*100
    thr = mt.thrust(path_dir)
    av = mt.ang_vel(path_dir)
    me = ((thr[1]/4)*av[1] + (thr[2]/4)*av[2] + (thr[3]/4)*av[3] + (thr[4]/4)*av[4])*0.047*0.1*0.05
    mech = np.sum(me)
    msum = mt.sum_ar(me)
    m1 = mt.sum_ar(motor[1])
    m2 = mt.sum_ar(motor[2])
    m3 = mt.sum_ar(motor[3])
    m4 = mt.sum_ar(motor[4])


    plt.plot(t, m1, label='m1')
    plt.plot(t, m2, label='m2')
    plt.plot(t, m3, label='m3')
    plt.plot(t, m4, label='m4')
    plt.legend()
    plt.show()
    print('m1: ', m1[-1])
    print('m2: ', m2[-1])
    print('m3: ', m3[-1])
    print('m4: ', m4[-1])
