import yaml
import math
import numpy as np
from random import randint
import matplotlib.pylab as plt
import numpy.matlib
import time
from scipy.integrate import ode
from scipy.signal import find_peaks


# INPUTS:
# sig ... 1D signal 
# T ... vector of time steps
# threshold ... minimal oscillation amplitudes to threat the behaviour as oscillatory
# plot_on ... 0: no plotting, 1: plotting
#
# OUTPUTS:
# oscillatory ... 0: NO, 1: YES
# frequency 
# amplitude 
# spikiness  
def measureOsc3(sig, T, threshold):
	damped = 0
	
	# params
    #minDist = 1/dt;        
	
	#[peaks,locs,~,p] = findpeaks(sig,'MinPeakProminence',threshold);
	peaks, _ = find_peaks(sig, prominence=threshold)
	#plt.plot(sig)	
	#plt.plot(peaks, sig[peaks], "x")
	#plt.show()
	if peaks.any():
		if (len(peaks) >= 2):
			threshold2 = 0.1 * sig[int(math.ceil(len(peaks)/2))]
			peaks, _ = find_peaks(sig, prominence=threshold2)
			if (peaks.any() or len(peaks) < 2):
				damped = 1

	amplitude = 0
	period = 0
	oscillatory = 0
	frequency = 0
		
	if peaks.any():
		if len(peaks) >= 2:
			amplitude = sig[peaks[len(peaks)-2]] - min(sig[peaks[len(peaks)-2]:peaks[len(peaks)-1]])
			period = T[peaks[len(peaks)-1]]-T[peaks[len(peaks)-1]]
			if (T[peaks[len(peaks)-1]] < T[len(T)-1] - 1.5*period): #if oscillations are not damped the last peak should lie in the interval t_end - period (1.5*period - last peak can be misdetected)
				amplitude = 0
				period = 0
				damped = 1
				print("someting")
			else:         
				frequency = 1/period
				oscillatory = 1		
				
	
	return [oscillatory, frequency, period, amplitude, damped]
	
# ----------------
# Prepis Matlab kode repressilator_S_PDE.m, avtorja doc. dr. Miha Moskon - https://fri.uni-lj.si/sl/o-fakulteti/osebje/miha-moskon
# ----------------

# ali rob predstavlja konec prostora ali so meje neskoncne?
# Brez sinhronizacijske molekule: periodic_bounds = 1
# Z difuzijo sinhronizacijske molekule: periodic_bounds = 0 
periodic_bounds = 1
# nalaganje shranjene konfiguracije?
load_conf = 0
# shranjevanje koncne konfiguracije?
save_conf = 0
# fiksni robovi ali spremenljivi
borderfixed = 0
# snemanje videa - casovno potratno
# movie_on = 0

# Read YAML file
with open("params.yaml", 'r') as stream:
    p = yaml.load(stream)

# nalaganje vrednosti parametrov
#p = load('params.mat')
alpha = p['alpha']
alpha0 = 0.001 * alpha
Kd = p['Kd']
delta_m = p['delta_m']
delta_p = p['delta_p']
n = p['n']
beta = p['beta']
kappa = p['kappa']
kS0 = p['kS0']
kS1 = p['kS1']
kSe = p['kSe']
D1 = p['D1']
eta = p['eta']

size = p['size']                               
density = p['density']
n_cells = int(math.ceil(density * math.pow(size, 2)))

t_end = p['t_end']
dt = p['dt']
h = p['h']
h2 = h*h
	
S_e = np.random.rand(size, size)
S_i = np.zeros((size, size), dtype=float)

#CELLS = np.random.randint(2, size=(size, size))
CELLS = np.zeros((size, size), dtype=int)
for i in range(0, n_cells):
	idxI = randint(0, size-1)
	idxJ = randint(0, size-1)
	while CELLS[idxI][idxJ] == 1:
		idxI = randint(0, size-1)
		idxJ = randint(0, size-1)
	CELLS[idxI][idxJ] = 1

A = CELLS * np.random.rand(size, size) * 100
B = CELLS * np.random.rand(size, size) * 100
C = CELLS * np.random.rand(size, size) * 100

mA = CELLS * np.random.rand(size, size) * 100
mB = CELLS * np.random.rand(size, size) * 100
mC = CELLS * np.random.rand(size, size) * 100


def model(var, t):
	dmA = CELLS * (alpha/(1 + np.power((C/Kd), n)) + alpha0 - delta_m * mA)
	dmB = CELLS * (alpha/(1 + np.power((A/Kd), n)) + alpha0 - delta_m * mB)
	dmC = CELLS * (alpha/(1 + np.power((B/Kd), n)) + alpha0 - delta_m * mC + (kappa * S_i)/(1 + S_i))

	dA = CELLS * (beta * mA - delta_p * A)
	dB = CELLS * (beta * mB - delta_p * B)
	dC = CELLS * (beta * mC - delta_p * C)
	   
	dS_i = CELLS * (- kS0 * S_i + kS1 * A - eta * (S_i - S_e))
	dS_e = - kSe * S_e + CELLS * (eta * (S_i - S_e))
	
	return [dmA, dmB, dmC, dA, dB, dC, dS_i, dS_e]
	
	
#A_series = np.zeros(int(t_end/dt), dtype=float)
#S_e_series = np.zeros(int(t_end/dt), dtype=float)
A_full = np.zeros((int(t_end/dt)+1, size*size), dtype=float)

t = 0
k = 0
step = 0

A_full[step, :] = A.flatten('F') #[A>0]


a = np.arange(1, size, dtype=int)
b = np.arange(0, size-1, dtype=int)
i = np.argsort(np.append(a, 0))
j = np.argsort(np.append(size-1, b))

# merjenje casa
timeMeasure = time.time()

while t <= t_end:
	
	S_e_xx = []
	S_e_yy = []
	if (periodic_bounds):
		S_e_xx = D1 * (S_e[:, i] + S_e[:, j] - 2 * S_e)/h2 # D1 * ([S_e(:,end),S_e(:,1:end-1)] + [S_e(:,2:end),S_e(:,1)] -2*S_e)/h2 
		S_e_yy = D1 * (S_e[i, :] + S_e[j, :] - 2 * S_e)/h2 # D1 * ([S_e(end,:);S_e(1:end-1,:)] + [S_e(2:end,:);S_e(1,:)] -2*S_e)/h2  
	else:
		X = np.zeros((size+2, size+2), dtype=float)
		X[0,1:size+1] = S_e[1,:]
		X[size+1,1:size+1] = S_e[len(S_e)-2,:]
		X[1:size+1, 0] = S_e[:,1]
		X[1:size+1, size+1] = S_e[:, len(S_e)-2]
		X[1:size+1,1:size+1] = S_e

		S_e_xx = D1 * (X[1:len(X) - 1, 0:len(X) - 2] + X[1:len(X) - 1, 2:len(X)] - 2*S_e)/h2 
		S_e_yy = D1 * (X[0:len(X) - 2, 1:len(X) - 1] + X[2:len(X), 1:len(X) - 1] - 2*S_e)/h2 
        
	D2S_e = S_e_xx + S_e_yy

	# Calculate dx/dt
    #[dmA, dmB, dmC, dA, dB, dC, dS_i, dS_e] = repressilator_S_ODE(CELLS, mA, mB, mC, A, B, C, S_i, S_e, alpha, alpha0, Kd, beta, delta_m, delta_p, n, kS0, kS1, kSe, kappa, eta)
	#time = np.linspace(0.0,100.0,1000)
	#vec = odeint(model, A.flatten('F'), time)
	
	dmA = CELLS * (alpha/(1 + np.power((C/Kd), n)) + alpha0 - delta_m * mA)
	dmB = CELLS * (alpha/(1 + np.power((A/Kd), n)) + alpha0 - delta_m * mB)
	dmC = CELLS * (alpha/(1 + np.power((B/Kd), n)) + alpha0 - delta_m * mC + (kappa * S_i)/(1 + S_i))

	dA = CELLS * (beta * mA - delta_p * A)
	dB = CELLS * (beta * mB - delta_p * B)
	dC = CELLS * (beta * mC - delta_p * C)
	   
	dS_i = CELLS * (- kS0 * S_i + kS1 * A - eta * (S_i - S_e))
	dS_e = - kSe * S_e + CELLS * (eta * (S_i - S_e))
		
	dS_e = dS_e + D2S_e
	
	if (borderfixed):
		# leave border as distrotion centers
		dS_e[0:size,0] = 0
		dS_e[0:size,size-1] = 0
		dS_e[0, 0:size] = 0
		dS_e[size-1, 0:size] = 0
		         
	mA = mA + dt * dmA
	mB = mB + dt * dmB
	mC = mC + dt * dmC
	A = A + dt * dA
	B = B + dt * dB
	C = C + dt * dC
	S_i = S_i + dt * dS_i
	S_e = S_e + dt * dS_e
        
	t = t + dt
	step = step + 1
	
	A_full[step,:] = A.flatten('F') #[A>0]

# izpis casa
print("Porabljen cas: {} s.".format(time.time() - timeMeasure))

# T = 0:dt:t_end-dt
T = np.arange(0, t_end-dt, dt, dtype=int)

result = measureOsc3(A_full[1000:10000,size*size-20], T[1000:10000], 0.1)
print(result)

#TT = T.'
# TMat = repmat(TT,1,n_cells);
TMat = np.matlib.repmat(T, 1, n_cells)
y = np.arange(0, n_cells, dtype=int)
# yMat = repmat(y, numel(TT), 1); #//For plot3
yMat = np.matlib.repmat(y, len(T), 1)

fig, ax = plt.subplots()
ax.plot(A_full[1000:10000,size*size-1])
plt.show()


