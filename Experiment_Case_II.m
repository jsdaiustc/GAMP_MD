clear;
close all;
rand('seed',10)
randn('seed',10)
addpath(genpath(fileparts(mfilename('fullpath'))));


%% NSFI/UCRC bearing dataset with the theoretical fault frequency being 236.4Hz %%%%
y=load('2004.02.16.06.02.39_550');    % early stage
Fs=20480;                             % the sampling rate is 20480 Hz;
y=y(:,1);
N=length(y);
t = (0 : N-1) / Fs;


%% 50% signals in small intervals lasting 40 samples are randomly chosen to be missing
y=reshape(y,40,512);
ind = randperm(512);
indice = ind(1:256);
y(:,indice) = 0;
y=y(:);

%% The proposed GAMP method
indexG=find(y==0);
y_envo= abs(hilbert(y))-mean(abs(hilbert(y)));            % EQ.(6)-obtain the noisy fault impulse signal envelope
y_h=  hilbert(y_envo);                                    % EQ.(7)- remove negative/conjuate components
f_sample=[50:2:1000];                                     % set the grid covering the frequency domain
[res_x,res_sample] =GAMP_MD(y_h,f_sample,Fs,indexG);

%% SBFL [26]
[x_SBFL,sample_SBFL] =fault_frequency_learning(y_h,f_sample,Fs);

%% RV_ESPRIT [27]
M=1000;                                      % window size
range=[0,1000];                              % filtering range
[hat_f,~,hat_s]=RV_ESPRIT(y_h,M,Fs,range);

%% PCA [22]
y_temp=y;
y_temp(indexG)=NaN;
y_PCA1=reshape(y_temp,length(y_temp)/10,10);                 % transform the signal vector into a matrix of order 2048 * 10
[coeff1,score1,latent,tsquared,explained,mu1] = pca(y_PCA1,'algorithm','als');
y_PCA= score1*coeff1'+ repmat(mu1,length(y_temp)/10,1);      % impute the missing samples
y_PCA=y_PCA(:);

%% PCA-SBFL
y_envo_PCA= abs(hilbert(y_PCA))-mean(abs(hilbert(y_PCA)));   % obtain the noisy fault impulse signal envelope of PCA
y_h_PCA=hilbert(y_envo_PCA);                                 % remove negative/conjuate components
[x_PCA, sample_PCA] =fault_frequency_learning(y_h_PCA,f_sample,Fs);

%% BCFP [23]
Y=reshape(y,128,16,10);
DIM=[128,16,10];
index=find(Y==0);
O=ones(DIM);
O(index)=0;
[model] = BCPF_TC(Y, 'obs', O, 'init', 'ml', 'maxRank', 128, 'dimRed', 1, 'tol', 1e-6, 'maxiters', 100, 'verbose', 2);
x_BCPF=reshape(double(model.X),20480,1);

%% P-GSL [16]
[P_GSL_result] = P_GSL(y, Fs);

%% AdaESPGL, downloaded from https://zhaozhibin.github.io/  %%%%%%%%%%%%%%%%%%%%%%%%%
Params.Fs            = Fs;     % The sampling frequency of the simulation signal
Params.N             = N;      % The length of the signal
Params.N1    = 4;              % The samples of one impulse
Params.M     = 4;              % The number of periods
Params.Fn_N  = 0;              % a vector which contains the period of each component (Fs / fc)
Params.mu    = 9.235e-4;       % The parameter related to sparsity within groups
Params.pen   = 'atan';         % The penalty function
Params.rho   = 1;              % The degree of nonconvex
Params.Nit   = 100;            % The number of iteration
% Estimate noise
[C,L]=wavedec(y,5,'sym8');
c1=detcoef(C,L,1);
est_noise=median(abs(c1-median(c1)))/0.678;
Params.lam= 0.272*est_noise + 0.044;
[AdaESPGL_result] = AdaESPGL(y, Params);


%% original signal
figure (1);
subplot(5,2,1);
plot(t,y,'black')
axis([0 1 -0.4 0.4]);
ylabel('\fontname{Times New Roman}Amplitude\fontname{Times New Roman}(m/s^2)');
title('(a) original','fontname','Times New Roman');
%% original signal envelope spectrum
F = ([1:N]-1)*Fs/N;
F2= F(1:2001);
temp1=2001;temp2=0.02;
subplot(5,2,2);
y_spec=abs(fft(abs(hilbert(y))-mean(abs(hilbert(y)))))/(N/2);
plot(F2,y_spec(1:2001),'blue')
axis([0 temp1 0 temp2]);
ylabel('\fontname{Times New Roman}Amplitude\fontname{Times New Roman}(m/s^2)');
title('(b) original','fontname','Times New Roman');
%% signal recoved by P_GSL
subplot(5,2,3);
plot(t,P_GSL_result,'black')
axis([0 1 -0.4 0.4]);
ylabel('\fontname{Times New Roman}Amplitude\fontname{Times New Roman}(m/s^2)');
title('(c) P-GSL','fontname','Times New Roman');
%% signal envelope spectrum of P-GSL
subplot(5,2,4);
our_PSBL_enve=abs(fft(abs(hilbert(P_GSL_result)) -mean(abs(hilbert(P_GSL_result))) ))/(N/2);
plot(F2,  our_PSBL_enve(1:2001),'blue')
axis([0 temp1 0 temp2]);
ylabel('\fontname{Times New Roman}Amplitude\fontname{Times New Roman}(m/s^2)');
title('(d) P-GSL','fontname','Times New Roman');
%% signal recoved by AdaESPGL
subplot(5,2,5);
plot(t,AdaESPGL_result,'black')
axis([0 1 -0.4 0.4]);
ylabel('\fontname{Times New Roman}Amplitude\fontname{Times New Roman}(m/s^2)');
title('(e) AdaESPGL','fontname','Times New Roman');
%% signal envelope spectrum of AdaESPGL
subplot(5,2,6);
y_AdaESPGL_enve= abs(fft(abs(hilbert(AdaESPGL_result))-mean(abs(hilbert(AdaESPGL_result)))))/(N/2);
plot(F2,  y_AdaESPGL_enve(1:2001),'blue')
axis([0 temp1 0 temp2]);
ylabel('\fontname{Times New Roman}Amplitude\fontname{Times New Roman}(m/s^2)');
title('(f) AdaESPGL','fontname','Times New Roman');
%% signal recoved by PCA
subplot(5,2,7);
plot(t,y_PCA,'black')
axis([0 1 -0.4 0.4]);
ylabel('\fontname{Times New Roman}Amplitude\fontname{Times New Roman}(m/s^2)');
title('(g) PCA','fontname','Times New Roman');
%% signal envelope spectrum of PCA
subplot(5,2,8);
y_PCA_enve= abs(fft(abs(hilbert(y_PCA))-mean(abs(hilbert(y_PCA)))))/(N/2);
plot(F2,  y_PCA_enve(1:2001),'blue')
axis([0 temp1 0 temp2]);
ylabel('\fontname{Times New Roman}Amplitude\fontname{Times New Roman}(m/s^2)');
title('(h) PCA','fontname','Times New Roman');
%%  signal recovered by BCPF
subplot(5,2,9);
plot(t,x_BCPF,'black')
axis([0 1 -0.4 0.4]);
xlabel('\fontname{Times New Roman}Time\fontname{Times New Roman}(s)');
ylabel('\fontname{Times New Roman}Amplitude\fontname{Times New Roman}(m/s^2)');
title('(i) BCPF','fontname','Times New Roman');
%% signal envelope spectrum of BCPF
subplot(5,2,10);
y_BCPF_enve= abs(fft(abs(hilbert(x_BCPF))-mean(abs(hilbert(x_BCPF)))))/(N/2);
plot(F2,  y_BCPF_enve(1:2001),'blue')
axis([0 temp1 0 temp2]);
xlabel('\fontname{Times New Roman}Frequency\fontname{Times New Roman}(Hz)');
ylabel('\fontname{Times New Roman}Amplitude\fontname{Times New Roman}(m/s^2)');
title('(j) BCPF','fontname','Times New Roman');

%% fault frequency detection result of GAMP
figure (2);
fw=2001;
subplot(3,2,1);
stem(res_sample,res_x/2,'marker','none','color','red');
axis([0 fw 0 0.04]);
ylabel('\fontname{Times New Roman}Amplitude\fontname{Times New Roman}(m/s^2)');
title('(a) GAMP','fontname','Times New Roman');
%% fault frequency detection result of SBFL
subplot(3,2,2);
stem(sample_SBFL,x_SBFL/2,'marker','none','color','blue');
axis([0 fw 0 0.04]);
ylabel('\fontname{Times New Roman}Amplitude\fontname{Times New Roman}(m/s^2)');
title('(b) SBFL','fontname','Times New Roman');
%% fault frequency detection result of PCA-SBFL
subplot(3,2,3);
stem(sample_PCA,x_PCA/2,'marker','none','color','blue');
axis([0 fw 0 0.04]);
xlabel('\fontname{Times New Roman}Frequency\fontname{Times New Roman}(Hz)');
ylabel('\fontname{Times New Roman}Amplitude\fontname{Times New Roman}(m/s^2)');
title('(c) PCA-SBFL','fontname','Times New Roman');
%% fault frequency detection result of RV-ESPRIT
subplot(3,2,4);
stem(hat_f,abs(hat_s),'marker','none','color','blue');
axis([0 fw 0 0.04]);
xlabel('\fontname{Times New Roman}Frequency\fontname{Times New Roman}(Hz)');
ylabel('\fontname{Times New Roman}Amplitude\fontname{Times New Roman}(m/s^2)');
title('(d) RV-ESPRIT','fontname','Times New Roman');


