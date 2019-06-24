function [regret] = Thompson_PG_semi(data,opt)
% full feedback case

[d,T] = size(data.feature);
k = data.K;
b = opt.b;
B = opt.B;
invB = repmat(zeros(d),[1 1 k]);
for j = 1:k
    invB(:,:,j) = inv(B(:,:,j));
end
M = opt.M;

regret = zeros(T+1, 1);
theta = zeros(k,d);
for j = 1:k
    theta(j,:) = mvnrnd(b(j,:),B(:,:,j));
end

Y = zeros(k,T);

for t = 2:T
    
    for m = 1:M        
      kappa = -1/2 * ones(k,t-1);
      C = zeros(t-1,k);
      Omega = repmat(zeros(t-1),[1 1 k]);
      
      for i = 1:t-1
         xi = data.feature(:,i);
         
         for j = 1:k
             kappa(j,i) = kappa(j,i) + Y(j,i);
         end
         
         temp = zeros(k,1);
         sum = 0;
         for j = 1:k
             if theta(j,:) * xi > 500
                temp(j,1) = exp(500);
             else
                temp(j,1) = exp(theta(j,:) * xi);
             end
             sum = sum + temp(j,1);
             %fprintf("\n a %3.2e,%3.2e,%3.2e\n",sum,temp(j,1),theta(j,:) * xi);
         end
         
         for j = 1:k
             if abs(sum-temp(j,1)) < 1
                C(i,j) = 0;
             else
                C(i,j) = log(sum-temp(j,1));
             end
             %fprintf("\n a %3.2e,%3.2e\n",sum,theta(j,:) * xi);
             Omega(i,i,j) = PG(1,1,theta(j,:) * xi-C(i,j));
             %fprintf("\n %3.2e\n",Omega(i,i,j));
         end        
      end
      
      X = data.feature(:,1:t-1);
      for j = 1:k
          V = inv(X * Omega(:,:,j) * X' + invB(:,:,j));
          s = V * (X * (kappa(j,:)' - Omega(:,:,j) * C(:,j)) + invB(:,:,j) * b(j,:)');
          theta(j,:) = mvnrnd(s',(V+V')/2);
      end    
      
    end
    
    x = data.feature(:,t);
    y = data.label(t);
    
    [~,a] = max(theta * x);
    regret(t,1) = regret(t-1,1) + (a ~= y);
    Y(a,t) = (a == y);
    fprintf("\n %d period: %3.2e, label: %d, true: %d\n",t,regret(t,1),a,y);
end


end



