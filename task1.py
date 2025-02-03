import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#%% riproducibilità

t = np.loadtxt("raggix_riproducibilita.txt")
tt = t.T
nn = [40, 60, 80, 100] 
c = []
d = []

for i in range(len(t[0])):
    c.append( np.mean(tt[i]) )
    d.append( np.std(tt[i]) )
    print(f"La misura di riproducibilita' a {nn[i]} kVp fornisce una media e deviazione standard di")
    print(c[i], d[i], "\n")


def linear(x, m, q):
    return m*x+q
    

#%% variazione corrente anodica

print("variazione corrente anodica \n")
t = np.loadtxt("variazione_corrente_anodica.txt")
corr = t[:, 0]  # Corrente anodiche
dose = t[:, 1]  # Dose

popt, pcov = curve_fit(linear, corr, dose, p0 = [4, 5])

plt.errorbar(corr, dose, color="red", fmt="o")
plt.plot(corr, linear(corr, *popt), color="blue")
plt.xlabel("Corrente anodica [mA]")
plt.ylabel("Dose [uGy]")
#plt.legend( )
plt.xlim(0, max(corr)+10)
plt.ylim(0, max(dose)+20)
plt.grid()
plt.show()



#%% variazione corrente anodica

print("variazione kV picco \n")
t = np.loadtxt("variazione_kVolt_picco.txt")
kV = t[:, 0]  # Corrente anodiche
dose = t[:, 1]  # Dose

popt, pcov = curve_fit(linear, kV, dose, p0 = [3, 5])

plt.errorbar(kV, dose, color="red", fmt="o")
plt.plot(kV, linear(kV, *popt), color="blue")
plt.xlabel("Differenza di potenziale [mV]")
plt.ylabel("Dose [uGy]")
#plt.legend( )
plt.xlim(min(kV)-10, max(kV)+10)
plt.ylim(min(dose)-10, max(dose)+20)
plt.grid()
plt.show()


#%% variazione tempo di esposizione

print("variazione tempo di esposizione \n")
t = np.loadtxt("variazione_tempo_exp.txt")
tt = t[:, 0]  # tempo di esposizione
dose = t[:, 1]  # Dose

popt, pcov = curve_fit(linear, tt, dose, p0 = [3, 5])

plt.errorbar(tt, dose, color="red", fmt="o")
plt.plot(tt, linear(tt, *popt), color="blue")
plt.xlabel("Tempo di esposizione [mA]")
plt.ylabel("Dose [uGy]")
#plt.legend( )
plt.xlim(0, max(tt)+10)
plt.ylim(0, max(dose)+30)
plt.grid()
plt.show()