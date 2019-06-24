function [x] = mass_texpon(z)
% p/(p+q) in PG1
%   p = (0.5 * pi) * exp( -1.0 * fz * t) / fz
%   q = 2 * exp(-1.0 * Z) * pigauss(TRUNC, 1.0/Z, 1.0)

t = 0.64;
fz = pi^2 / 8 + z^2 / 2;
b = sqrt(1 / t) * (t * z - 1);
a = -sqrt(1/t) * (t * z + 1);
x0 = log(fz) + fz * t;

xb = x0 - z + log(cdf('Normal',b));
xa = x0 + z + log(cdf('Normal',a));

qdivp = 4 / pi * (exp(xb) + exp(xa));

x = 1 / ( 1 + qdivp);

end

