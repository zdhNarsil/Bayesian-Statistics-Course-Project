function [regret] = Thompson1_ensemble(data, N, pre)
    
    [d, T] = size(data.feature);
    k = data.K;
    mu = zeros(k, d);
    for i = 2:N
        mu(:, :, i) = zeros(k, d);
    end
    Sigma = repmat(eye(d), [1, 1, k, N]);
    regret = zeros(T+1, 1);
    
    for j = 1:N
       for i = 1:k
           mu(i, :, j) = mvnrnd(zeros(1, d), eye(d));
       end
    end
    
    if pre
    
    for t = 1:T
        x = data.feature(:, t);
        y = data.label(t);
        index = randi([1, N]);
        prob = probability(x, mu(:, :, index));
        [~, predict] = max(prob);
        regret(t+1) = regret(t) + (predict ~= y);
        for j = 1:N
            prob = probability(x, mu(:, :, j));
            z = poissrnd(1);
            for i = 2:k
                Sigma(:, :, i, j) = diag(diag(Sigma(:, :, i, j) + z * prob(i)*(1-prob(i))*(x*x')));
                mu(i, :, j) = mu(i, :, j) + z * ((y==i)-prob(i)) * (x'*diag(1./diag(Sigma(:, :, i, j))));
            end
        end
    end
    
    else
      
    for t = 1:T
        x = data.feature(:, t);
        y = data.label(t);
        index = randi([1, N]);
        prob = probability(x, mu(:, :, index));
        [~, predict] = max(prob);
        regret(t+1) = regret(t) + (predict ~= y);
        for j = 1:N
            prob = probability(x, mu(:, :, j));
            z = poissrnd(1);
            for i = 2:k
                Sigma(:, :, i, j) = morrison(Sigma(:, :, i, j), prob(i), x, z);
                mu(i, :, j) = mu(i, :, j) + z * ((y==i)-prob(i)) * (x'*Sigma(:, :, i, j));
            end
        end
    end    
        
    end
    
    regret = regret(2:end);
end

