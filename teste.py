import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io



def exec1(sinal,janela):
    
		m = [np.nan]*10000
		m = np.matrix(m)
		win = np.blackman(janela)
		init = 0
	    
		while True:
				amostras = sinal[init:(init + janela)]
				if(len(amostras) == janela):
						amostras = amostras * win
						amostras = np.pad(amostras, (0,(10000-janela)), 'constant', constant_values=(0))
						amostras = np.fft.fftshift(abs(np.fft.fft(amostras)))**2
						amostras = np.matrix(amostras)
						m = np.concatenate((m,amostras), axis=0)
						init += int(janela/2)
				else:
						break

		m = np.delete(m,0,0)
		m = np.matrix.tolist(m.mean(0).T)
																																											    
		return m

def separa_QRS(sinal, tamanhoQRS, limiar):
    
		i = 1
		D = [] * tamanhoQRS
		incremento = int(np.round(tamanhoQRS/2))
		D = np.matrix(D)

		
		while (i <= (len(x1)-300)):
				if (sinal[i] >= limiar):       
						amostras = x1[i-incremento:i+incremento]
						amostras = np.matrix(amostras)
						D = np.concatenate((D,amostras), axis=1)
						i+=incremento
				else:
						i+=1
																													
																																
																																			
		D = np.matrix.tolist(D.T)       
																																
		return D


data = scipy.io.loadmat('ECG_1.mat')

fs = 360
y1 = data['x']
x1 = [row[0] for row in y1]

E = separa_QRS(x1,40,100)
print(len(E))

B = exec1(E,500)
print(len(B))
