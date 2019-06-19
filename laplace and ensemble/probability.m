function [prob] = probability(x, omega)

    y = exp(omega * x);
    if any(isinf(y))
        [~, t] = max(y);
        y = zeros(length(y), 1);
        y(t) = 1;
    end
    prob = y / sum(y);

end

