%% Allen-Cahn equation - High Resolution Version
% 原版本: nn = 511 (512 points), steps = 200 (dt = 0.005)
% 高精度版本: nn = 2047 (2048 points) - 4倍空间精度
%             steps = 1000 (dt = 0.001) - 5倍时间精度
nn = 2047;  % 增加到2048个空间点，提高4倍空间精度
steps = 1000;  % 增加到1000步，dt = 0.001，提高5倍时间精度

dom = [-1 1]; x = chebfun('x',dom); t = linspace(0,1,steps+1);
S = spinop(dom,t);
S.lin = @(u) 5*u + 0.0001*diff(u,2);
S.nonlin = @(u) - 5*u.^3;
S.init = x.^2 .* cos(pi*x);
u = spin(S,nn,1e-6,'plot','off');  % 提高时间精度到1e-6

usol = zeros(nn,steps+1);
for i = 1:steps+1
    usol(:,i) = u{i}.values;
end

x = linspace(-1,1,nn+1);
usol = [usol;usol(1,:)];
pcolor(t,x,usol); shading interp, axis tight, colormap(jet);
usol = usol'; % shape = (steps+1, nn+1)
save('allen_cahn_highres.mat','t','x','usol')

fprintf('High-resolution reference solution generated:\n');
fprintf('  Spatial points: %d (4x original: %d -> %d)\n', nn+1, 512, nn+1);
fprintf('  Time points: %d (5x original: %d -> %d)\n', steps+1, 201, steps+1);
fprintf('  Spatial resolution: dx = %.6f (original: %.6f)\n', 2/(nn), 2/511);
fprintf('  Time resolution: dt = %.6f (original: %.6f)\n', 1/steps, 1/200);
fprintf('  Time precision: 1e-6 (10x original: 1e-5)\n');
fprintf('  Saved to: allen_cahn_highres.mat\n');