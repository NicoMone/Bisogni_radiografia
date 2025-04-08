#effetto_heel_modificato_MC
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#dati sperimentali
# data underscope voltaggio puicco
posizione = np.array([0., 2., 4., 6., 8., 9., 10., 11., 12., 13., 14., 16., 18., 
                      20., 22., 24., 26., 28., 30.])
data_40 = np.array([ [103.2, 103.7, 103.2], [103.8, 104.1, 104.1], [103.8, 103.8, 103.2],
                    [103., 103., 103.3], [102.1, 102.7, 102.4], [100.9, 100.3, 101.5],
                    [101.3, 99.1, 100.1], [99.5, 100.1, 99.8],
                    [98.9, 98.4, 99.], [97.4, 98., 97.4],
                    [96.1, 96.4, 96.7], [93.5, 93.5, 94.1], [90.9, 90.3, 90.6], 
                    [87.1, 86.8, 87.1], [83., 83., 83.1], [79.3, 78.7, 78.7], 
                    [73.2, 74., 73.7], [68.2, 68.2, 67.6], [61.2, 61.5, 61.5] ])

y40 = np.mean(data_40, axis=1)
sigma40 = np.sqrt(np.std(data_40, axis=1)**2 + (0.03*y40)**2)

#Metodo Monte Carlo
N = 10000000 # numero di fotoni

x_fotoni = np.random.normal(0, 30, N)
theta = 13  # angolo di inclinazione dell'anodo in gradi
mu = 1.82  # coefficiente di attenuazione (cm^-1) del materiale dell'anodo (es. tungsteno)
spessore_base = 0.02  # spessore minimo (cm) attraversato dal fotone se perpendicolare all'anodo
L = 100 #distanza anodo-rivelatore
alpha_fotoni = np.arctan(x_fotoni/ L )

# punto d'interazione simulato con distribuzione esponenziale
c_fotoni =  np.random.exponential(scale=0.02, size=N)

# cammino nel tungsteno
d_fotoni = c_fotoni * np.cos(theta) / np.cos(alpha_fotoni - theta)

# probabilità di trasmissione
P_fotoni = np.exp(-mu * d_fotoni)

# histogram per posizione
bins = np.linspace(0, 30, 100)
intensita, _ = np.histogram(x_fotoni, bins=bins, weights=P_fotoni)
centri = 0.5 * (bins[:-1] + bins[1:])
intensita /= np.max(intensita)

#interpolazione e confronto con il MC
interp_sim = interp1d(centri, intensita, kind='linear', bounds_error=False, fill_value="extrapolate")
simulati_sui_dati = interp_sim(posizione)
chi2 = np.sum(((y40 / max(y40) - simulati_sui_dati)**2) / (sigma40 / max(y40))**2)
ndof = len(posizione) - 1  # gradi di libertà
print(f"Chi-quadro: {chi2:.2f}/{ndof:.2f}, Chi-quadro ridotto: {chi2/ndof:.2f}")

#plot
plt.plot(centri, intensita, label="Monte Carlo", color="teal")
plt.errorbar(posizione, y40/max(y40), sigma40/max(y40), fmt = ".",color="blue", capsize=1)
plt.xlabel("Posizione sul rivelatore (cm)")
plt.ylabel("Intensità (norm.)")
plt.title("Simulazione Monte Carlo - Effetto Heel")
plt.grid(True)
plt.legend()
plt.show()