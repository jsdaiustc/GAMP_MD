function [res_x,f_sample] = GAMP_MD(y,f_sample,Fs,indexG)


N=length(f_sample);
[T,M]=size(y);
norm_y=norm(y,'fro')/sqrt(T*M);
y=y/norm_y;

BT=ones(T,1);
BT(indexG)=0;
T_all=[0:1:T-1]';
A=exp(1i*2*pi*T_all/Fs*f_sample)/sqrt(N);
A=BT.*A;
B=(1i*2*pi*T_all/Fs).*A;
reslu=f_sample(2)-f_sample(1);



%% GAMP initialization
maxiter=100;
beta0=1;
delta=ones(N,1)*1;
X_hat=zeros(N,1);

hold_max=10^10;
hold_min=10^(-10);
V_x   =  ones(N,1);
S_hat =  zeros(T,1);
rho=0;
 
converged = false;
truncated_active=false;
iter = 0;
tol=1e-5;
etc=20;

while ~converged && iter<maxiter

    if iter==0
        rho=0.5;
    end
     
    for zzz=1:1
       %% factor update part
        z=A*X_hat;
        z(indexG)=hold_min;                          
        V_p= (sum(V_x)/N)*ones(T,1);                % (5a)  
        V_p=min(max(V_p,hold_min),hold_max);
        P_hat=z - (V_p.*S_hat);                     % (5b)  
        
       %% step 2
        eta=ones(T,1)*beta0;
        eta(indexG)=hold_max;
        tao_z=(V_p.*eta)./(V_p+eta);
        z_hat=tao_z.*(y./eta+P_hat./V_p);                    
        V_s=(1-tao_z./V_p)./V_p;                 % (6a)
        V_s=min(max(V_s,hold_min),hold_max);            
        S_hat=(z_hat-P_hat)./V_p;                % (6b)
               
       %% variable update part                
        V_r= N./( sum(V_s) );                      % (7a)
        V_r=min(max(V_r,hold_min),hold_max);
        R_hat= X_hat + (V_r.*( A'* S_hat));        % (7b)
        
       %% step 4
        X_hat_old=X_hat;
        V_x_old=V_x;
        V_x=1./(  (1./delta)*ones(1,M)  + 1./V_r   );           % (8a)
        V_x=min(max(V_x,hold_min),hold_max);
        X_hat= V_x.*( 0 +  R_hat./V_r   );                      % (8b) 
        X_hat= (1-rho)*X_hat + (rho)*X_hat_old;
        V_x= (1-rho)*V_x + rho*V_x_old;
    end
    
    mu=X_hat;
    Sigma=V_x;
    Exx =  mu.*conj(mu) + Sigma;
    
    %% EQ.(33)-update noise variance 
    beta_old=beta0;
    resid=y-z_hat;
    resid(indexG)=[]; 
    z_temp=tao_z;
    z_temp(indexG)=[];
    sum_z=sum(z_temp);
    beta0=( norm(resid(:), 'fro')^2+  real(sum_z)   )/(T-length(indexG));
    beta0=  (1-rho)*beta0 + rho*beta_old;
    
    %% EQ.(35)-update signal precision
    delta_last=delta;
    sum_temp1=sum(Exx,2);
    delta=sum_temp1;

    
    %% off-grid    
  if ~truncated_active
    Pm=sum( mu.*conj(mu), 2);
    [~,sort_ind]=sort(Pm, 'descend');    
    idx=sort_ind(1:etc);
    BB=B(:,idx);
    BHB = BB' * BB;
    P2=diag(Sigma(idx));
    P = real( conj(BHB) .* ((mu(idx,:) * mu(idx,:)') +   P2   )  );
    Tes=sum(Sigma,2);
    v2= real(diag(BB' * A .* Tes(idx) ));
    v = sum( real(conj(mu(idx,:)) .* (BB' * (y - A * mu))),2) -   v2;
    temp_grid=v./diag(P);
    temp_grid=temp_grid';
    
    theld=reslu/20*0.95^(iter);
    ind_small=find(abs(temp_grid)<theld);
    temp_grid(ind_small)=sign(temp_grid(ind_small))*theld;
    ind_unchang=find (abs(temp_grid)>reslu);
    temp_grid(ind_unchang)=sign(temp_grid(ind_unchang)) * reslu/20;
    f_sample(idx)=f_sample(idx) + temp_grid;
    F_active=exp(1i*2*pi*T_all/Fs*f_sample(idx))/sqrt(N);
    A(:,idx)=F_active;
    B(:,idx)=(1i*2*pi*T_all/Fs).*A(:,idx);
    
  else
      BHB = B' * B;
      P2= M * diag(Sigma);
      P = real( conj(BHB) .* ((mu * mu') +   P2   )  );
      v2= M * real(diag(B' * A * diag(Sigma) ));
      v = sum( real(conj(mu) .* (B' * (y - A * mu))),2) -   v2;
      vect1=[1:N]';
      P=vect1'*P*vect1;
      temp_grid=(vect1'*v)/P;
      theld=reslu/20*0.95^(iter);
      ind_small=find(abs(temp_grid)<theld);
      temp_grid(ind_small)=sign(temp_grid(ind_small))*theld;
      ind_unchang=find (abs(temp_grid)>reslu);
      temp_grid(ind_unchang)=sign(temp_grid(ind_unchang)) * reslu/20;
      f_sample=f_sample + temp_grid*vect1';
      A=exp(1i*2*pi*T_all/Fs*f_sample)/sqrt(N);
      B=(1i*2*pi*T_all/Fs).*A;
   end
    
      if iter==30
          Pm2=delta;
          fn=search_Pm(Pm2,f_sample);
          fn_all= fn*[1:1:10];
          f_sample=fn_all;
          A= exp(1i*2*pi*T_all/Fs*f_sample)/sqrt(N);
          B=(1i*2*pi*T_all/Fs).*A;
          N=length(f_sample);
          delta=ones(N,1)*1;
          X_hat=zeros(N,1);
          V_x=ones(N,1);
          etc=min(etc,N);
          delta_last=100;
      end

 %% prune out irrelevant term
    if iter<30
       theld2=20;
       ind_remove= find (delta<(max(delta)/theld2));
       A(:,ind_remove)=[];
       B(:,ind_remove)=[];
       delta(ind_remove)=[];
       delta_last(ind_remove)=[];
       f_sample(ind_remove)=[];
       X_hat(ind_remove)=[];
       V_x(ind_remove)=[];
       N=length(f_sample);
       etc=min(etc,N);
    end

     
    %% stopping criteria
    erro=  max(max(abs(delta - delta_last)));  
    if erro < tol || iter >= maxiter
        converged = true;
    end
    iter = iter + 1;
    
end
  
Pm=sum( mu.*conj(mu), 2);
res_x=sqrt(Pm)*norm_y;  


