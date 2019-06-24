function [a] = a_coef(n,x)
% calculate nth coefficient in density of PG(1, 0)

t = 0.64;

if x > t
    a = pi * (n+0.5) * exp(-(n + 0.5)^2 * pi^2 * x / 2);
else
    a = (2 / pi / x)^1.5 * pi * (n + 0.5) * exp(-2 * (n + 0.5)^2 / x);
end

end

