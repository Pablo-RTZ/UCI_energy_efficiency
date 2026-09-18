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
%       seed      - Seed for the random number generator
%
%   Outputs:
%       Xtrain  - Training data
%       Xtest   - Test data
%       trainIdx - Indices of observations assigned to the training set
%       testIdx  - Indices of observations assigned to the test set

arguments
    X (:,:) double
    trainCoef (1,1) double
    seed (1,1) int
end

if trainCoef <= 0 || trainCoef >= 1
    error('trainCoef must be between 0 and 1.');
end

rng(seed);
n = size(X, 1);
nTrain = round(n * trainCoef);

% Randomly shuffle observation indices
idx = randperm(n);

% Split indices
trainIdx = idx(1:nTrain);
testIdx  = idx(nTrain+1:end);

% Split matrix
Xtrain = X(trainIdx, :);
Xtest  = X(testIdx, :);

if nargin() < 4
    clear testIdx
elseif nargin() < 3
    clear trainIdx
end

end
