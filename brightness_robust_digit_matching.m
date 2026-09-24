function results = brightness_robust_digit_matching(dataFile)
%BRIGHTNESS_ROBUST_DIGIT_MATCHING Compare two nearest-neighbor metrics.
%
% OVERVIEW
%   Standard Euclidean nearest-neighbor matching is sensitive to global
%   brightness. This program also evaluates a scaled model y = a*x + noise,
%   where x is a training image and y is a test image. For every candidate
%   x, the maximum-likelihood scale estimate under independent Gaussian
%   noise is a_hat = (x'*y)/(x'*x). Substituting a_hat into the residual
%   produces a brightness-adjusted matching score.
%
%   The scale is constrained to be nonnegative because image intensity
%   inversion is not a meaningful brightness correction here. The program
%   reports empirical classification errors; these are test-set estimates,
%   not theoretical probabilities derived from a known data distribution.

if nargin < 1
    dataFile = "data.mat";
end
[imageTrain, imageTest, labelTrain, labelTest] = load_digit_data(dataFile);
[height, width, nTrain] = size(imageTrain);
nTest = size(imageTest, 3);
d = height * width;

Xtrain = reshape(double(imageTrain), d, nTrain);
Xtest = reshape(double(imageTest), d, nTest);

trainNorm2 = sum(Xtrain.^2, 1).';      % nTrain-by-1
testNorm2 = sum(Xtest.^2, 1);          % 1-by-nTest
crossProduct = Xtrain.' * Xtest;       % nTrain-by-nTest

% Standard squared Euclidean distance.
euclideanDistance2 = trainNorm2 + testNorm2 - 2 * crossProduct;
euclideanDistance2 = max(euclideanDistance2, 0);
[~, euclideanIndex] = min(euclideanDistance2, [], 1);
euclideanPrediction = labelTrain(euclideanIndex(:));

% Profile out the nuisance brightness scale. A zero-energy training image
% cannot define a scale, so eps prevents division by zero.
scaleEstimate = max(crossProduct, 0) ./ max(trainNorm2, eps);
scaledResidual2 = testNorm2 ...
    - 2 * scaleEstimate .* crossProduct ...
    + (scaleEstimate.^2) .* trainNorm2;
scaledResidual2 = max(scaledResidual2, 0);
[~, scaledIndex] = min(scaledResidual2, [], 1);
scaledPrediction = labelTrain(scaledIndex(:));

classes = 0:9;
[euclideanPerClass, euclideanConfusion] = classification_metrics( ...
    labelTest, euclideanPrediction, classes);
[scaledPerClass, scaledConfusion] = classification_metrics( ...
    labelTest, scaledPrediction, classes);
euclideanError = mean(euclideanPrediction ~= labelTest);
scaledError = mean(scaledPrediction ~= labelTest);

fprintf('Brightness-robust digit matching\n');
fprintf('Standard Euclidean error: %.2f%%\n', 100 * euclideanError);
fprintf('Scaled-residual error:   %.2f%%\n', 100 * scaledError);

figure('Name', 'Distance-Metric Comparison');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
bar(classes, 100 * [euclideanPerClass, scaledPerClass]);
xlabel('True digit'); ylabel('Error rate (%)');
title('Per-class empirical error');
legend('Euclidean', 'Brightness adjusted', 'Location', 'best'); grid on;
nexttile;
bar([100 * euclideanError, 100 * scaledError]);
set(gca, 'XTickLabel', {'Euclidean', 'Brightness adjusted'});
ylabel('Error rate (%)'); title('Total empirical error'); grid on;

% Visualize one example whose decision changed between metrics.
changed = find(euclideanPrediction ~= scaledPrediction, 1, 'first');
if ~isempty(changed)
    scaledTrainIndex = scaledIndex(changed);
    euclideanTrainIndex = euclideanIndex(changed);
    chosenScale = scaleEstimate(scaledTrainIndex, changed);
    figure('Name', 'Effect of Brightness Adjustment');
    tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    nexttile; imagesc(imageTest(:, :, changed)); axis image off; colormap gray;
    title(sprintf('Test: true %d', labelTest(changed)));
    nexttile; imagesc(imageTrain(:, :, euclideanTrainIndex)); axis image off;
    title(sprintf('Euclidean match: %d', euclideanPrediction(changed)));
    nexttile; imagesc(chosenScale * double(imageTrain(:, :, scaledTrainIndex)));
    axis image off;
    title(sprintf('Scaled match: %d, a = %.2f', scaledPrediction(changed), chosenScale));
end

results = struct( ...
    'euclideanPrediction', euclideanPrediction, ...
    'scaledPrediction', scaledPrediction, ...
    'euclideanError', euclideanError, ...
    'scaledError', scaledError, ...
    'euclideanPerClassError', euclideanPerClass, ...
    'scaledPerClassError', scaledPerClass, ...
    'euclideanConfusionMatrix', euclideanConfusion, ...
    'scaledConfusionMatrix', scaledConfusion);
end

function [imageTrain, imageTest, labelTrain, labelTest] = load_digit_data(dataFile)
if ~isfile(dataFile), error('Data file not found: %s', dataFile); end
data = load(dataFile);
if ~isfield(data, 'labelTrain') || ~isfield(data, 'labelTest')
    labelPath = fullfile(fileparts(char(dataFile)), 'label.mat');
    if ~isfile(labelPath), error('Digit labels were not found.'); end
    labels = load(labelPath); names = fieldnames(labels);
    for k = 1:numel(names), data.(names{k}) = labels.(names{k}); end
end
required = {'imageTrain','imageTest','labelTrain','labelTest'};
for k = 1:numel(required)
    if ~isfield(data, required{k}), error('Missing variable: %s', required{k}); end
end
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
