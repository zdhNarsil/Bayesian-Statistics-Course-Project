function [regret] = Thompson2(data, pre)

    [d, T] = size(data.feature);
    k = data.K;
    mu = zeros(k, d);
    Sigma = eye(d)/10;
    for i = 2:k
        Sigma(:, :, i) = eye(d)/10;
    end
    omg = zeros(k, d);
    regret = zeros(T+1, 1);
    
    if pre
    
    for t = 1:T
        x = data.feature(:, t);
        y = data.label(t);
        
        for i = 2:k
            omg(i, :) = mvnrnd(mu(i, :), diag(1./diag(Sigma(:, :, i))));
        end
        prob = probability(x, omg);
        [~, predict] = max(prob);
        regret(t+1) = regret(t) + (predict ~= y);
        prob = probability(x, mu);
        
        if y == predict
            for i = 2:k
                Sigma(:, :, i) = diag(diag(Sigma(:, :, i) + prob(i)*(1-prob(i))*(x*x')));
                mu(i, :) = mu(i, :) + ((y==i)-prob(i)) * (x'*diag(1./diag(Sigma(:, :, i))));
            end
        else
            i = predict;
            Sigma(:, :, i) = diag(diag(Sigma(:, :, i) + prob(i)*(1-prob(i))*(x*x')));
            mu(i, :) = mu(i, :) + ((y==i)-prob(i)) * (x'*diag(1./diag(Sigma(:, :, i))));
        end
    end
    
    else
        
    for t = 1:T
        x = data.feature(:, t);
        y = data.label(t);
        for i = 2:k
            omg(i, :) = mvnrnd(mu(i, :), Sigma(:, :, i));
        end
        prob = probability(x, omg);
        [~, predict] = max(prob);
        regret(t+1) = regret(t) + (predict ~= y);
        prob = probability(x, mu);
        
        if y == predict
            for i = 2:k
                Sigma(:, :, i) = morrison(Sigma(:, :, i), prob(i), x, 1);
                mu(i, :) = mu(i, :) + ((y==i)-prob(i)) * (x'*Sigma(:, :, i));
            end
        else
            i = predict;
            Sigma(:, :, i) = morrison(Sigma(:, :, i), prob(i), x, 1);
            mu(i, :) = mu(i, :) + ((y==i)-prob(i)) * (x'*Sigma(:, :, i));
        end
    end 
        
    end
    
    regret = regret(2:end);
    
end
