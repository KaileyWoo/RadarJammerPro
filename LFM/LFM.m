clear
clc
%%
time_width=10^(-4);    %����100us
fre_width=10^6;        %����1Mhz
A=1;                   %����1
k=2*pi*fre_width/time_width;    %����б��
Fs=5*10^6;              %����Ƶ��
N=time_width*Fs;                %��������
%n=0:1/Fs:(N-1)/Fs;
n=linspace(-time_width/2,time_width/2,N);
f=linspace(-Fs/2,Fs/2,N);
%%
u=exp(1i*(k*n.^2/2));
Y=fft(u);
%% figure
figure
plot(n,real(u));
figure
plot(n,imag(u));
figure
plot(f,fftshift(abs(Y)));