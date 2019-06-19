function [Regret] = draw(data, set, k, info)

if ~isfield(info, 'ensem_num')
    info.ensem_num = 50; end
if ~isfield(info, 'test_num')
    info.test_num = 50; end
if ~isfield(info, 'diff')
    info.diff = 100; end

N = info.ensem_num;
num = info.test_num;
diff = info.diff;

[~, T] = size(data.feature);
Regret = [zeros(T, 1), zeros(T, 1), zeros(T, 1), zeros(T, 1)];
name = ["Laplace", "Laplace-pre", "Ensemble", "Ensemble-pre"];
j = 1;
if set == 1
for i = 1:num
    Regret(:, 1) = Regret(:, 1) + Thompson1(data, k, 0)/num;
    fprintf("%d %d\n", j, i);
end
j = j + 1;
for i = 1:num
    Regret(:, 2) = Regret(:, 2) + Thompson1(data, k, 1)/num;
    fprintf("%d %d\n", j, i);
end
j = j + 1;
for i = 1:num
    Regret(:, 3) = Regret(:, 3) + Thompson1_ensemble(data, k, N, 0)/num;
    fprintf("%d %d\n", j, i);
end
j = j + 1;
for i = 1:num
    Regret(:, 4) = Regret(:, 4) + Thompson1_ensemble(data, k, N, 1)/num;
    fprintf("%d %d\n", j, i);
end
end

if set == 2
for i = 1:num
    Regret(:, 1) = Regret(:, 1) + Thompson2(data, k, 0)/num;
    fprintf("%d %d\n", j, i);
end
j = j + 1;
for i = 1:num
    Regret(:, 2) = Regret(:, 2) + Thompson2(data, k, 1)/num;
    fprintf("%d %d\n", j, i);
end
j = j + 1;
for i = 1:num
    Regret(:, 3) = Regret(:, 3) + Thompson2_ensemble(data, k, N, 0)/num;
    fprintf("%d %d\n", j, i);
end
j = j + 1;
for i = 1:num
    Regret(:, 4) = Regret(:, 4) + Thompson2_ensemble(data, k, N, 1)/num;
    fprintf("%d %d\n", j, i);
end
end

figure();

for i = 1:4
    plot(1:T, Regret(:, i), 'Displayname', name(i));
    hold on
end
hold off;
legend('Location', 'Best');
xlabel('time period');
ylabel('cumulative regret');

figure();

for i = 1:4
    Regret_diff = Regret(diff:end, i) - Regret(1:end-diff+1, i);
    n = length(Regret_diff);
    plot(1:n, Regret_diff/diff, 'Displayname', name(i));
    hold on;
end
hold off;
legend('Location', 'Best');
xlabel('time period');
ylabel(strcat('average (', int2str(diff), ') regret'));

end
