function [x] = tigauss(z)
% sample from truncated Inv-Gauss(1/abs(z), 1.0) 1_{(0, t)}

t = 0.64;
z = abs(z);
mu = 1 / z;
x = t + 1;

if mu > t
    alpha = 0.0;
    
    while rand > alpha        
      E1 = random('Exponential',1);
      E2 = random('Exponential',1);
      
      while E1^2 > 2 * E2 / t
        E1 = random('Exponential',1);
        E2 = random('Exponential',1);
      end
      
      x = t / (1 + t * E1)^2;
      alpha = exp(-0.5 * z^2 * x);
    end
    
else
    
    while x > t
      lambda = 1.0;
      Y = random('Normal',0,1)^2;
      x = mu + 0.5 * mu^2 / lambda * Y - 0.5 * mu / lambda * sqrt(4 * mu * lambda * Y + (mu * Y)^2);     
      if rand > mu / (mu + x)
        x = mu^2 / x;
      end
    end
    
end

end

