function [regret] = Thompson_PG(data,opt)
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
for j = 1:k-1
    theta(j,:) = mvnrnd(b(j,:),B(:,:,j));
end

for t = 2:T+1
    
    for m = 1:M        
      kappa = -1/2 * ones(k,t-1);
      C = zeros(t-1,k);
      Omega = repmat(zeros(t-1),[1 1 k]);
      
      for i = 1:t-1
         xi = data.feature(:,i);
         yi = data.label(i);
         
         for j = 1:k
             kappa(j,i) = kappa(j,i) + (yi == j);
         end
         
         temp = zeros(k,1);
         sum = 0;
         for j = 1:k
             temp(j,1) = exp(theta(j,:) * xi);
             sum = sum + temp(j,1);
         end
         
         for j = 1:k
             if sum-temp(j,1)< 1
                 C(i,j) = 0;
             else
                C(i,j) = log(sum-temp(j,1));
             end
             Omega(i,i,j) = PG(1,1,theta(j,:) * xi-C(i,j));
         end        
      end
      
      X = data.feature(:,1:t-1);
      for j = 1:k-1
          V = inv(X * Omega(:,:,j) * X' + invB(:,:,j));
          s = V * (X * (kappa(j,:)' - Omega(:,:,j) * C(:,j)) + invB(:,:,j) * b(j,:)');
          theta(j,:) = mvnrnd(s',V);
      end    
      
    end
    
    x = data.feature(:,t);
    y = data.label(t);
    [~,a] = max(theta * x);
    regret(t,1) = regret(t-1,1) + (a ~= y);
    fprintf("\n %d period: %3.2e\n",t,regret(t,1));
end


end

