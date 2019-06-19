function [data] = generate_data(k, d, T)
    
    omega = mvnrnd(zeros(1, d), d * eye(d), k);
    omega(1, :) = zeros(1, d);
    data.feature = zeros(d, T);
    data.label = zeros(1, T);
    
    for i = 1:T
        x = mvnrnd(zeros(d, 1), 10 * eye(d));
        % x = unifrnd(-1, 1, d, 1);
        x = reshape(x, [], 1);
        prob = probability(x, omega);
        prob = reshape(prob, 1, []);
        % [~, y] = max(prob);
        y = randsrc(1, 1, [1:k; prob]);
        data.feature(:, i) = x;
        data.label(i) = y;
    end

end

