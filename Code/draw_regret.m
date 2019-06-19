function [] = draw_regret(Regret, name)

diff = 100;
[T, ~] = size(Regret);
% name = ["Laplace", "Laplace-pre", "Ensemble", "Ensemble-pre"];
for i = 1:length(name)
    plot(1:T, Regret(:, i), 'Displayname', name(i), 'Linewidth', 2);
    hold on
end
hold off;
legend('Location', 'Best');
xlabel('time period');
ylabel('cumulative regret');

figure();

for i = 1:length(name)
    Regret_diff = Regret(diff:end, i) - Regret(1:end-diff+1, i);
    n = length(Regret_diff);
    plot(1:n, Regret_diff/diff, 'Displayname', name(i), 'Linewidth', 1.5);
    hold on;
end
hold off;
legend('Location', 'Best');
xlabel('time period');
ylabel(strcat('average (', int2str(diff), ') regret'));


end

