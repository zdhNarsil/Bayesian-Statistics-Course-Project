function [regret] = linear(data)

    [d, T] = size(data.feature);
    k = data.K;
    
    % 超参
    lambda_prior = 0.25;
    a0 = 6;
    b0 = 6;
    
    mu = zeros(K, d + 1);
    % cov = eye(d); 
    for i = 2:k
        cov(:, :, i) = 1 / lambda_prior * eye(d + 1);
    end
    
    for i = 2:k
        precision(:, :, i) = lambda_prior * eye(d + 1);
    end
    
    a = repelem(a0, K);
    b = repelem(b0, K);
    
    regret = zeros(T+1, 1);
    for t = 1:T
        x = data.feature(:, t);
        y = data.label(t);
        
        % sample sigma
        sigma2_s = zeros(k);
        for i = 2:k
            sigma2_s(i) = 1. / (gamrnd(a(i), 1. / b(i)));
        end
        % sample beta
        beta_s = zeros(k, d+1);
        for i = 2:k
            beta_s(i, :) = mvnrnd(mu(i, :), sigma2_s(i) * cov(:, :, i)); %横着的向量
        end
        
        % select action
        vals = zeros(k);
        for i = 2:k
            vals(i) = beta_s(i, 1:d) * x + beta_s(i, d+1);
        end
        [~, action] = max(vals);
        
        % 
        s = x * x.';
        precision_a = s + lambda_prior * eye(d + 1);
        cov_a = inv(precision_a);
        mu_a = cov_a * (x.' * y); %？？
        
        a_post = a0 + t/2; %github上是 + size(x)(1) / 2. ??
        b_upd = 0.5 * (y.*y - mu_a.T * precision_a * mu_a);
        b_post = b0 + b_upd;
        
        mu(action, :) = mu_a;
        cov(:, :, action) = cov_a;
        precision(:, :, action) = precision_a;
        a(action) = a_post;
        b(action) = b_post;
        
        regret(t+1) = regret(t) + (predict ~= y);
    end
    regret = regret(2:end);
    
    