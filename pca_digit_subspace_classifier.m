function results = pca_digit_subspace_classifier(dataFile)
%PCA_DIGIT_SUBSPACE_CLASSIFIER Analyze and classify digits with PCA.
%
% OVERVIEW
%   Principal component analysis finds orthogonal directions of greatest
%   variation after subtracting the global training mean. Test images and
%   class means are projected into PCA subspaces of several dimensions, then
%   classified by the nearest projected class mean.
%
%   A second PCA model is learned from digit-5 images only. Its 40-dimensional
%   reconstruction residual provides a direct measure of how poorly each test
%   image fits the learned digit-5 subspace; the largest residual identifies
%   the least-5-like image under this model.
%
%   Principal components are signed variation directions, not average digit
%   images. Their signs are arbitrary, so the display uses a symmetric color
%   range around zero instead of interpreting bright pixels as strokes.

if nargin < 1
    dataFile = "data.mat";
end
[imageTrain, imageTest, labelTrain, labelTest] = load_digit_data(dataFile);
[height, width, nTrain] = size(imageTrain);
nTest = size(imageTest, 3);
d = height * width;

Xtrain = reshape(double(imageTrain), d, nTrain);
Xtest = reshape(double(imageTest), d, nTest);
classes = 0:9;

% Global PCA. The singular values produce covariance eigenvalues through
% lambda_i = sigma_i^2/(N-1), ordered from largest to smallest.
globalMean = mean(Xtrain, 2);
centeredTrain = Xtrain - globalMean;
[basis, singularValues, ~] = svd(centeredTrain, 'econ');
eigenvalues = diag(singularValues).^2 / (nTrain - 1);
explainedVariance = eigenvalues / sum(eigenvalues);
cumulativeVariance = cumsum(explainedVariance);
dimension95 = find(cumulativeVariance >= 0.95, 1, 'first');

figure('Name', 'Leading Principal Components');
tiledlayout(2, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
componentLimit = max(abs(basis(:, 1:10)), [], 'all');
for component = 1:10
    nexttile;
    imagesc(reshape(basis(:, component), height, width), ...
        [-componentLimit, componentLimit]);
    axis image off; title(sprintf('PC %d', component));
end
colormap gray;

figure('Name', 'PCA Variance Spectrum');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
semilogy(eigenvalues, 'LineWidth', 1.2);
xlabel('Principal-component index'); ylabel('Eigenvalue (log scale)');
title('Variance spectrum'); grid on;
nexttile;
plot(cumulativeVariance, 'LineWidth', 1.2); hold on;
yline(0.95, '--', '95%'); xline(dimension95, '--');
xlabel('Number of components'); ylabel('Cumulative explained variance');
title(sprintf('95%% variance at %d components', dimension95)); grid on;

% Build class means once in the original feature space.
classMeans = zeros(d, numel(classes));
for i = 1:numel(classes)
    mask = labelTrain == classes(i);
    if ~any(mask), error('No training samples found for digit %d.', classes(i)); end
    classMeans(:, i) = mean(Xtrain(:, mask), 2);
end

requestedDimensions = [5, 10, 20, 30, 40, 60, 90, 130, 180, 250, 350];
subspaceDimensions = requestedDimensions(requestedDimensions <= size(basis, 2));
totalError = zeros(size(subspaceDimensions));
predictions = cell(size(subspaceDimensions));

for i = 1:numel(subspaceDimensions)
    q = subspaceDimensions(i);
    projectionBasis = basis(:, 1:q);
    projectedTest = projectionBasis.' * (Xtest - globalMean);
    projectedMeans = projectionBasis.' * (classMeans - globalMean);
    distance2 = sum(projectedTest.^2, 1).' + sum(projectedMeans.^2, 1) ...
        - 2 * projectedTest.' * projectedMeans;
    [~, winner] = min(distance2, [], 2);
    predictions{i} = classes(winner).';
    totalError(i) = mean(predictions{i} ~= labelTest);
end

[bestError, bestIndex] = min(totalError);
bestDimension = subspaceDimensions(bestIndex);
bestPrediction = predictions{bestIndex};
[bestPerClassError, bestConfusion] = classification_metrics( ...
    labelTest, bestPrediction, classes);

figure('Name', 'PCA Classifier Performance');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
plot(subspaceDimensions, 100 * totalError, 'o-', 'LineWidth', 1.2);
xlabel('PCA subspace dimension'); ylabel('Error rate (%)');
title('Dimension-versus-error tradeoff'); grid on;
nexttile;
imagesc(classes, classes, bestConfusion); axis image; colorbar;
xlabel('Predicted digit'); ylabel('True digit');
title(sprintf('Best tested dimension: %d', bestDimension));

% Learn a class-specific subspace using only digit-5 training images.
digitOfInterest = 5;
fiveTrain = Xtrain(:, labelTrain == digitOfInterest);
fiveMean = mean(fiveTrain, 2);
centeredFive = fiveTrain - fiveMean;
[fiveBasis, ~, ~] = svd(centeredFive, 'econ');
fiveDimension = min(40, size(fiveBasis, 2));
fiveBasis = fiveBasis(:, 1:fiveDimension);

centeredTestForFive = Xtest - fiveMean;
fiveReconstruction = fiveMean + fiveBasis * (fiveBasis.' * centeredTestForFive);
fiveResidual = Xtest - fiveReconstruction;
residualEnergy = sum(fiveResidual.^2, 1);
[largestResidualEnergy, leastFiveIndex] = max(residualEnergy);

figure('Name', 'Least-5-like Image Under the Digit-5 Subspace Model');
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile; imagesc(reshape(Xtest(:, leastFiveIndex), height, width));
axis image off; colormap gray;
title(sprintf('Selected image (true %d)', labelTest(leastFiveIndex)));
nexttile; imagesc(reshape(fiveReconstruction(:, leastFiveIndex), height, width));
axis image off; title(sprintf('%d-D reconstruction', fiveDimension));
nexttile; imagesc(reshape(abs(fiveResidual(:, leastFiveIndex)), height, width));
axis image off; title('Absolute residual');

fprintf('PCA digit-subspace classifier\n');
fprintf('Components required for 95%% variance: %d\n', dimension95);
fprintf('Best tested classifier dimension: %d | Error rate: %.2f%%\n', ...
    bestDimension, 100 * bestError);
fprintf(['Least-5-like image index: %d | True label: %d | ' ...
    '40-D residual energy: %.4g\n'], leastFiveIndex, ...
    labelTest(leastFiveIndex), largestResidualEnergy);

results = struct( ...
    'globalMean', globalMean, ...
    'principalComponents', basis, ...
    'eigenvalues', eigenvalues, ...
    'explainedVariance', explainedVariance, ...
    'dimension95', dimension95, ...
    'subspaceDimensions', subspaceDimensions, ...
    'totalError', totalError, ...
    'bestDimension', bestDimension, ...
    'bestPrediction', bestPrediction, ...
    'bestPerClassError', bestPerClassError, ...
    'bestConfusionMatrix', bestConfusion, ...
    'leastFiveIndex', leastFiveIndex, ...
    'leastFiveTrueLabel', labelTest(leastFiveIndex), ...
    'leastFiveResidualEnergy', largestResidualEnergy);
end

function [imageTrain, imageTest, labelTrain, labelTest] = load_digit_data(dataFile)
if ~isfile(dataFile), error('Data file not found: %s', dataFile); end
data = load(dataFile);
if ~isfield(data, 'labelTrain') || ~isfield(data, 'labelTest')
    labelPath = fullfile(fileparts(char(dataFile)), 'label.mat');
    if ~isfile(labelPath), error('Digit labels were not found.'); end
    labels = load(labelPath); names = fieldnames(labels);
    for i = 1:numel(names), data.(names{i}) = labels.(names{i}); end
end
required = {'imageTrain','imageTest','labelTrain','labelTest'};
for i = 1:numel(required)
    if ~isfield(data, required{i}), error('Missing variable: %s', required{i}); end
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
