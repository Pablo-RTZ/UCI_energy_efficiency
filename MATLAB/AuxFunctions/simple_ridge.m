function beta = simple_ridge(X, y, lambda)
% SIMPLE_RIDGE Performs Ridge regression using the analytical solution
%
% beta = simple_ridge(X, y, lambda)
%
% Inputs:
%   X         - n x p predictor matrix
%   y         - n x 1 response vector
%   lambda    - regularization parameter (non-negative scalar)
%
% Outputs:
%   beta      - p x 1 vector of coefficients

arguments
    X (:,:) double
    y (:,1) double
    lambda (1,1) double {mustBeNonnegative}
end

P = diag([0, ones(1,size(X,2)-1)]); % Not penalize intercept

beta = (X' * X + P * lambda) \ X' * y;

end