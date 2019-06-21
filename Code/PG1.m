function [x] = PG1(z)
% sample from PG(1.0, z)--following from bayesLogit(R package)
%   PG(1,z) = 1/4 J*(1,z/2) with z>0

z = abs(z) * 0.5;
t = 0.64;
fz = pi^2 / 8 + z^2 / 2;

total.trials = 0;
total.iter = 0;

while true
    
    total.trials = total.trials + 1;
    if rand < mass_texpon(z)
        % Truncated exponential
        x = t + random('Exponential',1) / fz;
    else
        % Truncated inverse normal
        x = tigauss(z);
    end
    
    S = a_coef(0,x);
    Y = rand * S;
    n = 0;
    
    while true
        
        n = n + 1;
        total.iter = total.iter + 1;
        
        if rem(n,2) == 1
            S = S - a_coef(n,x);
            if Y <= S
                break;
            end
        else
            S = S + a_coef(n,x);
            if Y > S
                break;
            end
        end

    end
    
            
    if Y <= S
        break;
    end
        
end

x = 0.25 * x;
fprintf("x = %3.2e, n = %d, total iter = %d. ",x,total.trials,total.iter);

end

