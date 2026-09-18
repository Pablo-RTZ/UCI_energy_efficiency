function [beta, intercept] = simple_lasso(X, y, alpha, opts)
% SIMPLE_LASSO Performs LASSO regression using coordinate descent
%
% [beta, intercept] = simple_lasso(X, y, alpha)
%
% Inputs:
%   X         - n x p predictor matrix
%   y         - n x 1 response vector
%   alpha    - regularization parameter (non-negative scalar)
%   opts.maxiter - maximum number of iterations (default: 1000)
%   opts.tol     - convergence tolerance (default: 1e-6)
%
% Outputs:
%   beta      - p x 1 vector of coefficients
%   intercept - scalar intercept term

arguments
    X (:,:) double
    y (:,1) double
    alpha (1,1) double {mustBeNonnegative}
    opts.tol (1,1) double {mustBePositive} = 1e-6
    opts.maxiter (1,1) double {mustBeInteger, mustBePositive} = 1000
end

tol = opts.tol;
maxiter = opts.maxiter;
    
    p = size(X, 2);
    
    % Center the data
    X_mean = mean(X, 1);
    y_mean = mean(y);
    X_centered = X - X_mean;
    y_centered = y - y_mean;
    
    % Initialize coefficients
    beta = zeros(p, 1);
    
    % Precompute for efficiency
    X_squared = sum(X_centered.^2, 1)';
    
    % Coordinate descent
    iter = 0;
    while iter < maxiter
        beta_old = beta;
        
        for j = 1:p
            % Compute residual without j-th coefficient
            r_j = y_centered - X_centered * beta + X_centered(:, j) * beta(j);
            
            % Compute correlation
            rho = X_centered(:, j)' * r_j;
            
            % Soft thresholding
            if rho > alpha
                beta(j) = (rho - alpha) / X_squared(j);
            elseif rho < -alpha
                beta(j) = (rho + alpha) / X_squared(j);
            else
                beta(j) = 0;
            end
        end

        iter = iter + 1;
        
        % Check convergence
        if norm(beta - beta_old) <= tol
            break;
        end
    end
    
    % Compute intercept
    intercept = y_mean - X_mean * beta;
    
end