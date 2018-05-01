#coding: utf-8


import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal


#Carregando o sinal

data = scipy.io.loadmat('ECG_1.mat')

fs = 360
y1 = data['x']
x1 = [row[0] for row in y1]
x = np.insert(x1, 30720, 0)

#Criando o vetor de tempo e plotando o sinal "continuo"

t = np.linspace(0, len(x)/fs, len(x))

#calculo da fft do sinal

X1 = np.fft.fft(x)

X = np.fft.fftshift(X1)

f = np.linspace(-0.5, 0.5, len(x))

#criando filtro FIR janelamento com firwin

#passa-altas

w1 = signal.firwin(801, 1.5, window='blackman', pass_zero=False, nyq=fs/2)
w = np.pad(w1, (0,len(x) - len(w1)), 'constant', constant_values = (0))
W = np.fft.fft(w)
W2 = np.fft.fftshift(W)

#rejeita faixa 60, 120 e 180 HZ

w3 = signal.firwin(801, [59,61, 119,121, 179,179.9999], nyq=fs/2)
w4 = np.pad(w3, (0,len(x) - len(w3)), 'constant', constant_values = (0))
W5 = np.fft.fft(w4)
W6 = np.fft.fftshift(W5)



#multiplicando sinais na frequencia

Z = np.multiply(X1,W)
z = np.fft.ifft(Z)

Q = np.multiply(Z,W5)
q = np.fft.ifft(Q)
Q1 = np.fft.fftshift(Q)

#Obtenção das amostras do complexo QRS + zero padding + |fft|²

r1 = np.pad(q[245:265], (0,980),'constant',  constant_values = (0))
R1 = (abs(np.fft.fft(r1)))**2

r2 = np.pad(q[555:575], (0,980),'constant',  constant_values = (0))
R2 = (abs(np.fft.fft(r2)))**2

r3 = np.pad(q[865:885], (0,980),'constant',  constant_values = (0))
R3 = (abs(np.fft.fft(r3)))**2

r4 = np.pad(q[1160:1180], (0,980),'constant',  constant_values = (0))
R4 = (abs(np.fft.fft(r4)))**2

r5 = np.pad(q[1450:1470], (0,980),'constant',  constant_values = (0))
R5 = (abs(np.fft.fft(r5)))**2

r6 = np.pad(q[1735:1755], (0,980),'constant',  constant_values = (0))
R6 = (abs(np.fft.fft(r6)))**2

r7 = np.pad(q[2020:2040], (0,980), 'constant',  constant_values = (0))
R7 = (abs(np.fft.fft(r7)))**2

r8 = np.pad(q[2305:2325], (0,980), 'constant',  constant_values = (0))
R8 = (abs(np.fft.fft(r8)))**2

r9 = np.pad(q[2595:2615], (0,980),'constant',  constant_value = (0))
R9 = (abs(np.fft.fft(r9)))**2

r10 = np.pad(q[2832:2852], (0,980),'constant',  constant_values = (0))
R10 = (abs(np.fft.fft(r10)))**2

r11 = np.pad(q[3192:3212], (0,980),'constant',  constant_values = (0))
R11 = (abs(np.fft.fft(r11)))**2

r12 = np.pad(q[3495:3515], (0,980),'constant',  constant_values = (0))
R12 = (abs(np.fft.fft(r12)))**2

r13 = np.pad(q[3785:3805], (0,980),'constant',  constant_values = (0))
R13 = (abs(np.fft.fft(r13)))**2

r14 = np.pad(q[4073:4093], (0,980),'constant',  constant_values = (0))
R14 = (abs(np.fft.fft(r14)))**2

r15 = np.pad(q[4347:4367], (0,980),'constant',  constant_values = (0))
R15 = (abs(np.fft.fft(r15)))**2

r16 = np.pad(q[4652:4672], (0,980),'constant',  constant_values = (0))
R16 = (abs(np.fft.fft(r16)))**2

r17 = np.pad(q[4957:4977], (0,980),'constant',  constant_values = (0))
R17 = (abs(np.fft.fft(r17)))**2

r18 = np.pad(q[5254:5274], (0,980),'constant',  constant_values = (0))
R18 = (abs(np.fft.fft(r18)))**2

r19 = np.pad(q[5553:5573], (0,980),'constant',  constant_values = (0))
R19 = (abs(np.fft.fft(r16)))**2

r20 = np.pad(q[5848:5868], (0,980),'constant',  constant_values = (0))
R20 = (abs(np.fft.fft(r20)))**2

r21 = np.pad(q[6135:6155], (0,980),'constant',  constant_values = (0))
R21 = (abs(np.fft.fft(r21)))**2

r22 = np.pad(q[6420:6440], (0,980),'constant',  constant_values = (0))
R22 = (abs(np.fft.fft(r22)))**2

r23 = np.pad(q[6708:6728], (0,980),'constant',  constant_values = (0))
R23 = (abs(np.fft.fft(r23)))**2

r24 = np.pad(q[7003:7023], (0,980),'constant',  constant_values = (0))
R24 = (abs(np.fft.fft(r24)))**2

r25 = np.pad(q[7315:7335], (0,980),'constant',  constant_values = (0))
R25 = (abs(np.fft.fft(r25)))**2

r26 = np.pad(q[7613:7633], (0,980),'constant',  constant_values = (0))
R26 = (abs(np.fft.fft(r26)))**2

#calculo da média dos valores da DFT dos complexos QRS separados

L = []

for i in range (len(R1)):

    L.append(((R1[i]+R2[i]+R3[i]+R4[i]+R5[i]+R6[i]+R7[i]+R8[i]+R9[i]+R10[i]\
            +R11[i]+R12[i]+R13[i]+R14[i]+R15[i])+R16[i]+R17[i]+R18[i]+R19[i]\
            +R20[i]+R21[i]+R22[i]+R23[i]+R24[i]+R25[i]+R26[i]/26.0))

#Graficos


f_qrs = np.linspace(0, 180, len(L))

#sinal no tempo
plt.figure('Sinal no tempo e filtro passa-altas')
plt.subplot(2,2,1)
plt.title('Sinal no dominio do tempo')
plt.xlabel('Tempo em segundos')
plt.ylabel('xc(t)')
plt.plot(t, x, lw = 0.5)
plt.grid()

#Espectro do sinal
plt.subplot(2,2,2)
plt.title('Espectro do sinal')
plt.ylabel('|X(f)|')
plt.xlabel('Frequencia normalizada')
axes = plt.gca()
plt.xticks(np.arange(min(x), max(x)+1, 0.1))
axes.set_ylim(0, 100000)
plt.plot(f,abs(X), lw = 0.5)
plt.grid()

#Espectro do filtro
plt.subplot(2,2,3)
plt.title('Espectro do filtro passa-altas')
plt.ylabel('|W(f)|')
plt.xlabel('Frequencia normalizada')
plt.plot(f,abs(W2), lw = 0.5)
plt.xticks(np.arange(min(f), max(f)+1, 0.1))
axes = plt.gca()
axes.set_xlim(-0.5, 0.5)
axes.set_ylim(0, 1.5)
plt.grid()

#sinal no tempo filtrado
plt.subplot(2,2,4)
plt.title('Sinal filtrado')
plt.xlabel('Tempo em segundos')
plt.ylabel('xc(t)')
plt.plot(t,np.real(z), lw = 0.5)
plt.grid()

plt.subplots_adjust(left=0.12, bottom=0.08, right=0.90, top=0.95, wspace=0.25, hspace=0.36)

#Espectro do filtro
plt.figure('Sinal no tempo pos filtro passa-altas e filtragem rejeita faixa')
plt.subplot(2,2,2)
plt.title('Espectro do filtro rejeita faixa')
plt.ylabel('|W(f)|')
plt.xlabel('Frequencia normalizada')
plt.plot(f,abs(W6), lw = 0.5)
plt.xticks(np.arange(min(f), max(f)+1, 0.1))
axes = plt.gca()
axes.set_xlim(-0.5, 0.5)
axes.set_ylim(0, 1.5)
plt.grid()

#sinal no tempo filtrado
plt.figure(2)
plt.subplot(2,2,1)
plt.title('Sinal no dominio do tempo apos filtro passa-altas')
plt.xlabel('Tempo em segundos')
plt.ylabel('xc(t)')
plt.plot(t, np.real(z), lw = 0.5)
plt.grid()

#Espectro do sinal filtrado
plt.subplot(2,2,3)
plt.title('Espectro do sinal filtrado')
plt.ylabel('|X(f)|')
plt.xlabel('Frequencia normalizada')
plt.xticks(np.arange(min(f), max(f)+1, 0.1))
plt.plot(f,abs(Q1), lw = 0.5)
plt.grid()

#Espectro teste
plt.subplot(2,2,4)
plt.title('Sinal no tempo filtrado')
plt.ylabel('x_c(t)')
plt.xlabel('Tempo em segundos')
plt.plot(np.real(q), lw = 0.5)
plt.grid()

plt.subplots_adjust(left=0.12, bottom=0.08, right=0.90, top=0.95, wspace=0.25, hspace=0.36)

#Gráfico Energia Extração QRS

plt.figure('Resposta')
plt.plot(f_qrs[0:225], L[0:225], lw = 0.5)
plt.ylabel('AVG(|QRS|)**2')
plt.xlabel('Frequencia em hertz')
plt.grid()
plt.xticks(np.arange(min(f_qrs[0:225]), max(f_qrs[0:225])+1, 1))
#axes = plt.gca()
#axes.set_xlim(0, 40)
#axes.set_ylim(0, 1)

plt.show()


