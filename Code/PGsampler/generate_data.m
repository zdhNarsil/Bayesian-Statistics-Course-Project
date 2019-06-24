function [data] = generate_data(k, d, T, type)

% k: number of arms
% d: number of features
% T: number of periods
% type: 0: simulated data
%       1: shuttle data
%       2: covtype data

if type == 0
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
    
    data.K = k;
end

if type == 1
    d = importdata('/Users/apple/Desktop/shuttle.csv');
    d = d.data;
    d = [ones(length(d), 1), d];
    T = min(T, length(d));
    p = T/length(d);
    lab = d(:, end);
    rng(11);
    row = cvpartition(lab, 'HoldOut', p);
    d = d(row.test, :);
    data.feature = d(:, 1:end-1)';
    data.label = d(:, end)';

    for j = 1:d
       if norm(data.feature(j,:)) ~= 0
           data.feature(j,:) = (data.feature(j,:)-max(abs(data.feature(j,:))))/max(abs(data.feature(j,:)));
       end
    end
    
    data.K = 7;
 end

if type == 2
    d = importdata('/Users/apple/Desktop/covtype.csv');
    d = d.data;
    d = [ones(length(d), 1), d];
    p = T/length(d);
    lab = d(:, end);
    rng(22);
    row = cvpartition(lab, 'HoldOut', p);
    d = d(row.test, :);
    data.feature = d(:, 1:end-1)';
    data.label = d(:, end)';
    
    [d,~] = size(data.feature);
    for j = 1:d
        if norm(data.feature(j,:)) ~= 0
            data.feature(j,:) = data.feature(j,:)/max(abs(data.feature(j,:)));
        end
    end
    
    data.K = 7;
end

rng('default');

end

