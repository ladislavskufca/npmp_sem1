import yaml
import math
import numpy as np
from random import randint
import matplotlib.pylab as plt
import numpy.matlib
import time

# ----------------
# Prepis kode Matlab kode repressilator_S_PDE.m, avtorja doc. dr. Miha Moskon - https://fri.uni-lj.si/sl/o-fakulteti/osebje/miha-moskon
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

# merjenje casa
timeMeasure = time.time()

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
S_i = np.zeros((size, size), dtype=int)

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

A_series = np.zeros(int(t_end/dt), dtype=int)
S_e_series = np.zeros(int(t_end/dt), dtype=int)
A_full = np.zeros((int(t_end/dt)+1, n_cells), dtype=int)

####TODO
#A_series[1] = A(first_idx)
#S_e_series(1) = S_e(first_idx)
#A_full[1,:] = A(cell_idx)




t = 0
k = 0
step = 0

A_full[step,:] = filter(lambda i: i > 0, A.flatten('F'))


a = np.arange(1, size, dtype=int)
b = np.arange(0, size-1, dtype=int)
i = np.argsort(np.append(a, 0))
j = np.argsort(np.append(size-1, b))
while t <= t_end:
	
	## todo -> kar je v else je treba pretipkat v python
	#if (periodic_bounds):
	S_e_xx = D1 * (S_e[:, i] + S_e[:, j] - 2 * S_e)/h2 # D1 * ([S_e(:,end),S_e(:,1:end-1)] + [S_e(:,2:end),S_e(:,1)] -2*S_e)/h2 
	S_e_yy = D1 * (S_e[i, :] + S_e[j, :] - 2 * S_e)/h2 # D1 * ([S_e(end,:);S_e(1:end-1,:)] + [S_e(2:end,:);S_e(1,:)] -2*S_e)/h2  
	#else:
		#todo
		#SS_e = [[0, S_e[2,:], 0][S_e[:,2], S_e, S_e[:,len(S_e)-1]][0, S_e[len(S_e)-1,:], 0]]
		#S_e_xx= D1 * (SS_e(2:end-1,1:end-2) + SS_e(2:end-1,3:end) -2*S_e)/h2; 
		#S_e_yy= D1 * (SS_e(1:end-2,2:end-1) + SS_e(3:end,2:end-1) -2*S_e)/h2; 

	D2S_e = S_e_xx + S_e_yy
	# Calculate dx/dt
    #[dmA, dmB, dmC, dA, dB, dC, dS_i, dS_e] = repressilator_S_ODE(CELLS, mA, mB, mC, A, B, C, S_i, S_e, alpha, alpha0, Kd, beta, delta_m, delta_p, n, kS0, kS1, kSe, kappa, eta)
	dmA = CELLS * (alpha/(1 + np.power((C/Kd), n)) + alpha0 - delta_m * mA)
	dmB = CELLS * (alpha/(1 + np.power((A/Kd), n)) + alpha0 - delta_m * mB)
	dmC = CELLS * (alpha/(1 + np.power((B/Kd), n)) + alpha0 - delta_m * mC + (kappa * S_i)/(1 + S_i))

	dA = CELLS * (beta * mA - delta_p * A)
	dB = CELLS * (beta * mB - delta_p * B)
	dC = CELLS * (beta * mC - delta_p * C)
	   
	dS_i = CELLS * (- kS0 * S_i + kS1 * A - eta * (S_i - S_e))
	dS_e = - kSe * S_e + CELLS * (eta * (S_i - S_e))
		
	dS_e = dS_e + D2S_e
    
    #if (borderfixed == 1)
    #    % leave border as distrotion centers
    #    width = length(dS_e)
    #    dS_e(1:width,[1 width])=0
    #    dS_e([1 width],1:width)=0
    #    
    #end
        
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
	
	A_full[step,:] = filter(lambda i: i > 0, A.flatten('F'))

    
    ####TODO
    #A_series(step) = A(first_idx)
    #S_e_series(step) = S_e(first_idx)
    #A_full(step,:) = A(cell_idx)

# izpis casa
print("Porabljen cas: {} s.".format(time.time() - timeMeasure))


# T = 0:dt:t_end-dt
T = np.arange(0, t_end-dt, dt, dtype=int)

#TT = T.'
# TMat = repmat(TT,1,n_cells);
TMat = np.matlib.repmat(T, 1, n_cells)
y = np.arange(0, n_cells, dtype=int)
# yMat = repmat(y, numel(TT), 1); %//For plot3
yMat = np.matlib.repmat(y, len(T), 1)

fig, ax = plt.subplots()
ax.plot(A_full)
plt.show()


