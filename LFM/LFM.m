clear
clc
%%
time_width=10^(-4);    %脉宽100us
fre_width=10^6;        %带宽1Mhz
A=1;                   %幅度1
k=2*pi*fre_width/time_width;    %调制斜率
Fs=5*10^6;              %采样频率
N=time_width*Fs;                %采样点数
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