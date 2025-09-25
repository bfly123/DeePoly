%% Test Allen-Cahn equation solver step by step
try
    fprintf('Starting Allen-Cahn test...\n');

    % Reduce parameters for testing
    nn = 127;  % Much smaller for testing
    steps = 100;  % Much fewer steps
    fprintf('Using reduced parameters: nn=%d, steps=%d\n', nn, steps);

    % Set up domain and time
    dom = [-1 1];
    x = chebfun('x', dom);
    t = linspace(0, 1, steps+1);
    fprintf('Domain and time array created\n');

    % Create spinop
    S = spinop(dom, t);
    fprintf('Spinop created\n');

    % Set operators
    S.lin = @(u) 5*u + 0.0001*diff(u,2);
    S.nonlin = @(u) -5*u.^3;
    S.init = x.^2 .* cos(pi*x);
    fprintf('Operators set\n');

    % Try to solve
    fprintf('Starting solution...\n');
    u = spin(S, nn, 1e-4, 'plot', 'off');
    fprintf('Solution completed!\n');

    fprintf('Test PASSED - Allen-Cahn solver works\n');

catch ME
    fprintf('ERROR: %s\n', ME.message);
    fprintf('Identifier: %s\n', ME.identifier);
    if ~isempty(ME.stack)
        fprintf('Error location:\n');
        for i = 1:length(ME.stack)
            fprintf('  File: %s, Function: %s, Line: %d\n', ...
                ME.stack(i).file, ME.stack(i).name, ME.stack(i).line);
        end
    end
end