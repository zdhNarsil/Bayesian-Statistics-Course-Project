function [S] = morrison(S, p, x, lr)

sx = S * x;
pp = lr * p * (1-p);
S = S - pp*(sx*sx') / (1+pp*x'*sx);

end

