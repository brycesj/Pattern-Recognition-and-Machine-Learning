function results = nearest_class_mean_digit_classifier(dataFile)
%NEAREST_CLASS_MEAN_DIGIT_CLASSIFIER Classify digits with class prototypes.
%
% OVERVIEW
%   The program estimates one mean image for each digit and assigns every
%   test image to the closest mean in Euclidean distance. This is also the
%   Bayes decision rule for Gaussian classes with a shared identity
%   covariance matrix and equal class priors. The program visualizes the
%   learned prototypes, empirical errors, confusion matrix, and selected
%   within-class covariance matrices.
%
%   Unlike a full-covariance Gaussian classifier, this model deliberately
%   ignores correlations between pixels during prediction. Covariance plots
%   are therefore diagnostics rather than classifier parameters.

if nargin < 1
    dataFile = "data.mat";
end
[imageTrain, imageTest, labelTrain, labelTest] = load_digit_data(dataFile);
[height, width, nTrain] = size(imageTrain);
nTest = size(imageTest, 3);
d = height * width;

Xtrain = reshape(double(imageTrain), d, nTrain).'; % observations in rows
Xtest = reshape(double(imageTest), d, nTest).';
classes = 0:9;
classMeans = zeros(numel(classes), d);

% A class mean is a low-variance prototype built by averaging all training
% images carrying the same digit label.
for i = 1:numel(classes)
    mask = labelTrain == classes(i);
    if ~any(mask)
        error('Training data contains no observations for digit %d.', classes(i));
    end
    classMeans(i, :) = mean(Xtrain(mask, :), 1);
end

testNorm2 = sum(Xtest.^2, 2);
meanNorm2 = sum(classMeans.^2, 2).';
distance2 = testNorm2 + meanNorm2 - 2 * Xtest * classMeans.';
[~, winner] = min(distance2, [], 2);
predictedLabels = classes(winner).';

[perClassError, confusion] = classification_metrics( ...
    labelTest, predictedLabels, classes);
totalError = mean(predictedLabels ~= labelTest);

fprintf('Nearest-class-mean digit classifier\n');
fprintf('Accuracy: %.2f%% | Error rate: %.2f%%\n', ...
    100 * (1 - totalError), 100 * totalError);

figure('Name', 'Learned Digit Prototypes');
tiledlayout(2, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:numel(classes)
    nexttile;
    imagesc(reshape(classMeans(i, :), height, width));
    axis image off; colormap gray;
    title(sprintf('Mean of %d', classes(i)));
end

figure('Name', 'Prototype Classifier Performance');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
bar(classes, 100 * perClassError);
xlabel('True digit'); ylabel('Error rate (%)');
title('Per-class error rate'); grid on;
nexttile;
imagesc(classes, classes, confusion); axis image;
xlabel('Predicted digit'); ylabel('True digit');
title('Confusion matrix'); colorbar;

% Covariance matrices expose pairs of pixels that vary together. Their
% 784-by-784 size also illustrates why estimating and inverting a separate
% full covariance matrix for every class can be statistically fragile.
digitsToInspect = [0, 2, 5, 8];
figure('Name', 'Selected Within-Class Covariance Matrices');
tiledlayout(1, numel(digitsToInspect), 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:numel(digitsToInspect)
    digit = digitsToInspect(i);
    classSamples = Xtrain(labelTrain == digit, :);
    covarianceMatrix = cov(classSamples, 1);
    nexttile;
    imagesc(covarianceMatrix); axis image off;
    title(sprintf('Covariance: %d', digit));
end
colorbar;

results = struct( ...
    'classMeans', classMeans, ...
    'predictedLabels', predictedLabels, ...
    'totalError', totalError, ...
    'perClassError', perClassError, ...
    'confusionMatrix', confusion);
end

function [imageTrain, imageTest, labelTrain, labelTest] = load_digit_data(dataFile)
data = load_required_data(dataFile);
imageTrain = data.imageTrain; imageTest = data.imageTest;
labelTrain = normalize_digit_labels(data.labelTrain);
labelTest = normalize_digit_labels(data.labelTest);
if size(imageTrain, 1) ~= 28 || size(imageTrain, 2) ~= 28 || ...
        size(imageTest, 1) ~= 28 || size(imageTest, 2) ~= 28
    error('Expected image arrays with dimensions 28-by-28-by-N.');
end
if numel(labelTrain) ~= size(imageTrain, 3) || numel(labelTest) ~= size(imageTest, 3)
    error('The number of labels must match the number of images.');
end
end

function data = load_required_data(dataFile)
if ~isfile(dataFile), error('Data file not found: %s', dataFile); end
data = load(dataFile);
if ~isfield(data, 'labelTrain') || ~isfield(data, 'labelTest')
    labelPath = fullfile(fileparts(char(dataFile)), 'label.mat');
    if ~isfile(labelPath), error('Digit labels were not found.'); end
    labels = load(labelPath);
    names = fieldnames(labels);
    for k = 1:numel(names), data.(names{k}) = labels.(names{k}); end
end
required = {'imageTrain','imageTest','labelTrain','labelTest'};
for k = 1:numel(required)
    if ~isfield(data, required{k}), error('Missing variable: %s', required{k}); end
end
end

function labels = normalize_digit_labels(labels)
labels = double(labels(:)); values = unique(labels).';
if all(ismember(values, 0:9))
    return;
elseif all(ismember(values, 1:10))
    labels = labels - 1;
else
    error('Digit labels must use either 0:9 or 1:10.');
end
end

function [perClassError, confusion] = classification_metrics(truth, predicted, classes)
truth = truth(:); predicted = predicted(:);
confusion = zeros(numel(classes)); perClassError = nan(numel(classes), 1);
for i = 1:numel(classes)
    mask = truth == classes(i);
    if any(mask), perClassError(i) = mean(predicted(mask) ~= classes(i)); end
    for j = 1:numel(classes)
        confusion(i, j) = sum(mask & predicted == classes(j));
    end
end
end
