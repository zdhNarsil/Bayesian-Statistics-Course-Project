function [x] = PG(num,n,z)
% sample num rv from PG(n, z) with z>0
%   n is an integer


x = zeros(1,num);
for i = 1:num
    for j = 1:n
        x(i) = x(i) + PG1(z);
    end
end

end

