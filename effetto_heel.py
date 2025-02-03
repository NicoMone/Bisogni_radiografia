import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def function(x, a, b):
    return (a - b*x**2)

posizione = np.array([0., 2., 4., 6., 8., 9., 10., 11., 12., 13., 14., 16., 18., 
                      20., 22., 24., 26., 28., 30.])

# data underscope voltaggio puicco
data_40 = np.array([ [103.2, 103.7, 103.2], [103.8, 104.1, 104.1], [103.8, 103.8, 103.2],
                    [103., 103., 103.3], [102.1, 102.7, 102.4], [100.9, 100.3, 101.5],
                    [101.3, 99.1, 100.1], [99.5, 100.1, 99.8],
                    [98.9, 98.4, 99.], [97.4, 98., 97.4],
                    [96.1, 96.4, 96.7], [93.5, 93.5, 94.1], [90.9, 90.3, 90.6], 
                    [87.1, 86.8, 87.1], [83., 83., 83.], [79.3, 78.7, 78.7], 
                    [73.2, 74., 73.7], [68.2, 68.2, 67.6], [61.2, 61.5, 61.5] ])

y40 = np.mean(data_40, axis=1)
sigma40 = np.std(data_40, axis=1)

popt, pcov = curve_fit(function, posizione, y40, p0 = [103, 2])

plt.figure()
plt.plot(posizione, function(posizione, *popt))
plt.title("40 kVp, 100 mA, 100ms")
plt.errorbar(posizione, y40, sigma40, fmt='o')
plt.xlabel('posizione [CM]')
plt.ylabel(f'Dose [$\mu$Gy]')


data_70 = np.array([  [473., 474., 474.], [473., 473., 472.], [472., 473., 471.],
                    [468., 470., 469.], [469., 466., 466.], [462., 463., 462.],
                    [460., 460., 457.], [456., 455., 455.],
                    [453., 451., 452.], [446., 444., 445.], 
                    [442., 441., 440.], [428., 429., 431.],
                    [415., 416., 415.], [399., 400., 400.], [386., 383., 385.],
                    [367., 365., 368.], [347., 346., 347.], [322.9, 323., 324.],
                    [296.1, 293.7, 295.2] ])

y70 = np.mean(data_70, axis=1)
sigma70 = np.std(data_70, axis=1)

popt, pcov = curve_fit(function, posizione, y70, p0=[473, 2])

plt.figure()
plt.plot(posizione, function(posizione, *popt))
plt.title("70 kVp, 100 mA, 100ms")
plt.errorbar(posizione, y70, sigma70, fmt='o')
plt.xlabel('posizione [cm]')
plt.ylabel(f'Dose [$\mu$Gy]')


data_100 = np.array([ [965., 970., 972.], [972., 975., 975.], [971., 970., 973.], 
                     [965., 965., 966.], [965., 959., 963.], [956., 953., 955.],
                     [949., 944., 949.], [941., 940., 943.],
                     [927., 929., 931.], [921., 920., 920.],
                     [907., 911., 908.], [881., 883., 883.],
                     [859., 856., 855.], [827., 825., 826.], [794., 793., 794.],
                     [761., 759., 759.], [715., 716., 716.], [667., 667., 666.],
                     [610., 614., 613.] ])

y100 = np.mean(data_100, axis=1)
sigma100 = np.std(data_100, axis=1)

popt, pcov = curve_fit(function, posizione, y100, p0 = [969, 2])

plt.figure()
plt.title("100 kVp, 100 mA, 100ms")
plt.plot(posizione, function(posizione, *popt))
plt.errorbar(posizione, y100, sigma100, fmt='o')
plt.xlabel('posizione [cm]')
plt.ylabel(f'Dose [$\mu$Gy]')
plt.show()
