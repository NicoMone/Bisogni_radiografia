import numpy as np 
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

cost = 0.693
ro = 2.7 #densità dell'alluminio g/cm^3

data_40 = np.array([[89.3, 89.9, 90.2],[53.2, 53.7, 54.0],[40.7,42.7, 42.1] ,[33.3, 33.6, 33.6], [27.2, 27.5, 27.2], [22.5, 22.5,22.8], [20.5, 20.5, 20.5], [14.9, 15.2, 15.5]])
data_70 = np.array([[414., 414., 415.],[303.1,303.7, 304.0],[265.1, 264.5, 265.1] ,[234.6, 233.5, 233.2], [207.2, 206.9, 208.1],[185., 185., 185.8],[176.,176.,176.],[152., 152.9, 151.4]])
data_100 = np.array([[842., 853., 849.],[675.0, 678., 675.],[603., 607., 609.], [555., 555., 557.],[508.0, 510., 509.],[471., 470., 471.],[451., 453., 452.],[408.0, 411.0, 408.]])

x_data = np.array([0., 0.95, 1.55, 1.98, 2.53, 2.96, 3.22, 3.91]) 
x_err = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

#per HVL a 40kV
y_data1 = np.mean(data_40, axis=1)
y_err1 = np.std(data_40, axis=1)
#per HVL a 70kV
y_data2 = np.mean(data_70, axis=1)
y_err2 = np.std(data_70, axis=1)
#per HVL a 100kV
y_data3 = np.mean(data_100, axis=1)
y_err3 = np.std(data_100, axis=1)

interpolazione1 = interp1d(x_data, y_data1, kind="cubic")
interpolazione2 = interp1d(x_data, y_data2, kind="cubic")
interpolazione3 = interp1d(x_data, y_data3, kind="cubic")

inversa1 = interp1d(y_data1, x_data, kind="cubic")
inversa2 = interp1d(y_data2, x_data, kind="cubic")
inversa3 = interp1d(y_data3, x_data, kind="cubic")

HVL1 = inversa1(y_data1[0]/2)
HVL2 = inversa2(y_data2[0]/2)
HVL3 = inversa3(y_data3[0]/2)
print(f"HVL-----\n{HVL1}\n{HVL2}\n{HVL3}")
mu1 = (cost/HVL1)/ro #massico 
mu2 = (cost/HVL2)/ro
mu3 = (cost/HVL3)/ro
print(f"mu attL-----\n{mu1}\n{mu2}\n{mu3}")
def exp(x,a,b):
    return a*np.exp(x*b)

popt1, pcov1 = curve_fit(exp,x_data[3:], y_data1[3:])
popt2, pcov2 = curve_fit(exp,x_data[3:], y_data2[3:])
popt3, pcov3 = curve_fit(exp,x_data[3:], y_data3[3:])
#test del chi 
y_fit = exp(x_data[3:], *popt1)
residuals = y_data1[3:] - y_fit

for i in range(len(y_data1[3:])):
    y_eqv = np.sqrt(y_err1[i]**2 + (popt1[0]*popt1[1]*np.exp(popt1[1]*x_err[i]))**2)
    chi_square = np.sum((residuals[i]/y_eqv)**2) #va diviso per gli errori 
    

dof = len(x_data[3:]) - len(popt1)
chi_norm = chi_square / dof

print("---------")
print(f"Parametri del fit {popt1}")
print(f"il chi2 è {chi_square} \nil chi normalizzato è {chi_norm}")

x_nuovi = np.linspace(0, 2.55, 50 )
y_interpolati1 = interpolazione1(x_nuovi)

plt.scatter(HVL1, interpolazione1(HVL1), color="purple", marker="^", label="HVL")
plt.errorbar(x_data, y_data1, y_err1, x_err, fmt=".", color="blue", capsize=2)
plt.plot(x_nuovi, y_interpolati1, color="red", label="interpolazione")
plt.plot(x_data[3:], exp(x_data[3:], *popt1), color="green", label="andamento esponenziale")
plt.xlabel("Spessore della lastra [mm]")
plt.ylabel("Dose [uGy]")
plt.title(f"HVL per 40kV picco")
plt.xlim(min(x_data), max(x_data))
plt.ylim()
plt.legend(title=f"$\mu$ att ={np.round(mu1,2)} [cm^2/g]")
plt.grid(which="both", linestyle="--")
plt.show()
#secondo plot
x_nuovi = np.linspace(0, 1.98, 50 )
y_interpolati2 = interpolazione2(x_nuovi)
y_interpolati3 = interpolazione3(x_nuovi)

plt.scatter(HVL2, interpolazione2(HVL2),color="purple", marker="^", label="HVL")
plt.errorbar(x_data, y_data2, y_err2, x_err, fmt=".", color="blue", capsize=2)
plt.plot(x_nuovi, y_interpolati2, color="red", label="interpolazione")
plt.plot(x_data[3:], exp(x_data[3:], *popt2), color="green", label="andamento esponenziale")
plt.xlabel("Spessore della lastra [mm]")
plt.xlabel("Spessore della lastra [mm]")
plt.ylabel("Dose [uGy]")
plt.title(f"HVL per 70kV picco")
plt.xlim(min(x_data), max(x_data))
plt.ylim()
plt.legend(title=f"$\mu$ att = {np.round(mu2/0.1,2)} [cm^2/g]")
plt.grid(which="both", linestyle="--")
plt.show()

#terzo plot
plt.scatter(HVL3, interpolazione3(HVL3), color="purple", marker="^", label="HVL")
plt.errorbar(x_data, y_data3, y_err3, x_err, fmt=".", color="blue", capsize=2)
plt.plot(x_nuovi, y_interpolati3, color="red",label="interpolazione")
plt.plot(x_data[3:], exp(x_data[3:], *popt3), color="green", label="andamento esponenziale")
plt.xlabel("Spessore della lastra [mm]")
plt.xlabel("Spessore della lastra [mm]")
plt.ylabel("Dose [uGy]")
plt.title(f"HVL per 100kV picco")
plt.xlim(min(x_data), max(x_data))
plt.ylim()
plt.legend(title=f"$\mu$ att = {np.round(mu3/0.1,2)} [cm^2/g]")
plt.grid(which="both", linestyle="--")
plt.show()

