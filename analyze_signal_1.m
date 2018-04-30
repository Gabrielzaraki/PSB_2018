clear all; close all; clc;

load('ECG_1.mat', 'fs', 'x');

x1 = x(:, 1);
n = 0 : length(x1) - 1;
t = n / fs;

plot(t, x1);
grid on;
xlabel('Tempo (segundos)')
ylabel('x_c(t)')

X1 = fft(x1 - mean(x1));
figure;
f = linspace(-0.5, 0.5, length(X1)) * fs;
plot(f, fftshift(abs(X1)))
grid on;
xlabel('Frequências (hertz)')
ylabel('|X_c(f)|')
