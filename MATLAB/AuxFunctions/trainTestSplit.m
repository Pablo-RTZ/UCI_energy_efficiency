function [Xtrain, Xtest, trainIdx, testIdx] = trainTestSplit(X, trainCoef, seed)

%trainTestSplit Randomly splits a dataset into training and test sets.
%
%   [Xtrain, Xtest, trainIdx, testIdx] = trainTestSplit(X, trainCoef, seed)
%   Splits the observations in X into training and test sets according to
%   the specified training coefficient.
%
%   Inputs:
%       X         - Data matrix, with observations in rows
%       trainCoef - Proportion of observations assigned to the training set
%       seed      - Optional integer seed. If omitted, uses the current RNG
%                   state so repeated calls can generate independent splits.
%
%   Outputs:
%       Xtrain  - Training data
%       Xtest   - Test data
%       trainIdx - Indices of observations assigned to the training set
%       testIdx  - Indices of observations assigned to the test set

arguments
    X (:,:) double
    trainCoef (1,1) double
    seed (1,1) double = []
end

if trainCoef <= 0 || trainCoef >= 1
    error('trainCoef must be between 0 and 1.');
end

if ~isempty(seed)
    if ~isfinite(seed) || floor(seed) ~= seed
        error('seed must be an integer scalar when provided.');
    end
    rng(seed, 'twister');
end

n = size(X, 1);
nTrain = max(1, min(n - 1, round(n * trainCoef)));

% Randomly shuffle observation indices
idx = randperm(n);

% Split indices
trainIdx = idx(1:nTrain);
testIdx  = idx(nTrain+1:end);

% Split matrix
Xtrain = X(trainIdx, :);
Xtest  = X(testIdx, :);

if nargout() < 4
    clear testIdx
elseif nargout() < 3
    clear trainIdx
end

end
