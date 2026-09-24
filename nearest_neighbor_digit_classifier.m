function results = nearest_neighbor_digit_classifier(dataFile)
%NEAREST_NEIGHBOR_DIGIT_CLASSIFIER Classify handwritten digits with 1-NN.
%
% OVERVIEW
%   This standalone program treats every 28-by-28 grayscale image as a
%   784-dimensional feature vector. For each test image, it finds the
%   training image with the smallest Euclidean distance and transfers that
%   image's label. It reports total and per-class error rates, builds a
%   confusion matrix, and visualizes representative predictions.
%
% INPUT
%   dataFile - MAT-file containing imageTrain, imageTest, labelTrain, and
%              labelTest. If omitted, the program uses "data.mat". Labels
%              may be encoded as 0:9 or 1:10; both are normalized to 0:9.
%
% OUTPUT
%   results - structure containing predictions, nearest-neighbor indices,
%             distances, error rates, and the confusion matrix.

if nargin < 1
    dataFile = "data.mat";
end

[imageTrain, imageTest, labelTrain, labelTest] = load_digit_data(dataFile);
[height, width, nTrain] = size(imageTrain);
nTest = size(imageTest, 3);

% Each column is one observation. Scaling to [0,1] keeps distances readable
% without changing which observation is nearest.
Xtrain = reshape(double(imageTrain), height * width, nTrain);
Xtest = reshape(double(imageTest), height * width, nTest);
scale = max(Xtrain(:));
if scale > 1
    Xtrain = Xtrain / scale;
    Xtest = Xtest / scale;
end

% Pairwise squared distance identity:
% ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x'*y.
% This vectorized form avoids a slow loop over every train/test pair.
trainNorm2 = sum(Xtrain.^2, 1).';
testNorm2 = sum(Xtest.^2, 1);
distance2 = trainNorm2 + testNorm2 - 2 * (Xtrain.' * Xtest);
distance2 = max(distance2, 0); % suppress tiny negative round-off values

[nearestDistance2, nearestIndex] = min(distance2, [], 1);
predictedLabels = labelTrain(nearestIndex(:));

[perClassError, confusion] = classification_metrics( ...
    labelTest, predictedLabels, 0:9);
totalError = mean(predictedLabels ~= labelTest);

fprintf('1-nearest-neighbor digit classifier\n');
fprintf('Training samples: %d | Test samples: %d | Features: %d\n', ...
    nTrain, nTest, height * width);
fprintf('Accuracy: %.2f%% | Error rate: %.2f%%\n', ...
    100 * (1 - totalError), 100 * totalError);

figure('Name', '1-NN Error Analysis');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
bar(0:9, 100 * perClassError);
xlabel('True digit'); ylabel('Error rate (%)');
title('Per-class error rate'); grid on;
nexttile;
imagesc(0:9, 0:9, confusion); axis image;
xlabel('Predicted digit'); ylabel('True digit');
title('Confusion matrix'); colorbar;

% Show up to five errors and the training examples that drove the decision.
wrong = find(predictedLabels ~= labelTest);
nShow = min(5, numel(wrong));
if nShow > 0
    figure('Name', 'Representative 1-NN Errors');
    tiledlayout(2, nShow, 'TileSpacing', 'compact', 'Padding', 'compact');
    for k = 1:nShow
        testIndex = wrong(k);
        trainIndex = nearestIndex(testIndex);
        nexttile(k);
        imagesc(imageTest(:, :, testIndex)); axis image off; colormap gray;
        title(sprintf('Test: true %d, predicted %d', ...
            labelTest(testIndex), predictedLabels(testIndex)));
        nexttile(nShow + k);
        imagesc(imageTrain(:, :, trainIndex)); axis image off;
        title(sprintf('Nearest train: %d', labelTrain(trainIndex)));
    end
end

results = struct( ...
    'predictedLabels', predictedLabels, ...
    'nearestIndex', nearestIndex(:), ...
    'nearestDistance', sqrt(nearestDistance2(:)), ...
    'totalError', totalError, ...
    'perClassError', perClassError, ...
    'confusionMatrix', confusion);
end

function [imageTrain, imageTest, labelTrain, labelTest] = load_digit_data(dataFile)
if ~isfile(dataFile)
    error('Data file not found: %s', dataFile);
end
data = load(dataFile);
requiredImages = {'imageTrain', 'imageTest'};
for k = 1:numel(requiredImages)
    if ~isfield(data, requiredImages{k})
        error('The MAT-file must contain %s.', requiredImages{k});
    end
end

% Some versions of the dataset store labels in a separate label.mat file.
if ~isfield(data, 'labelTrain') || ~isfield(data, 'labelTest')
    labelPath = fullfile(fileparts(char(dataFile)), 'label.mat');
    if ~isfile(labelPath)
        error('labelTrain and labelTest were not found in %s or label.mat.', dataFile);
    end
    labels = load(labelPath);
    data = merge_structs(data, labels);
end

imageTrain = data.imageTrain;
imageTest = data.imageTest;
labelTrain = normalize_digit_labels(data.labelTrain);
labelTest = normalize_digit_labels(data.labelTest);

if size(imageTrain, 1) ~= 28 || size(imageTrain, 2) ~= 28 || ...
        size(imageTest, 1) ~= 28 || size(imageTest, 2) ~= 28
    error('Expected image arrays with dimensions 28-by-28-by-N.');
end
if numel(labelTrain) ~= size(imageTrain, 3) || ...
        numel(labelTest) ~= size(imageTest, 3)
    error('The number of labels must match the number of images.');
end
end

function labels = normalize_digit_labels(labels)
labels = double(labels(:));
values = unique(labels).';
if all(ismember(values, 0:9))
    return;
elseif all(ismember(values, 1:10))
    labels = labels - 1;
else
    error('Digit labels must use either the range 0:9 or 1:10.');
end
end

function [perClassError, confusion] = classification_metrics(truth, predicted, classes)
truth = truth(:); predicted = predicted(:);
confusion = zeros(numel(classes));
perClassError = nan(numel(classes), 1);
for i = 1:numel(classes)
    mask = truth == classes(i);
    if any(mask)
        perClassError(i) = mean(predicted(mask) ~= classes(i));
    end
    for j = 1:numel(classes)
        confusion(i, j) = sum(mask & predicted == classes(j));
    end
end
end

function output = merge_structs(a, b)
output = a;
names = fieldnames(b);
for k = 1:numel(names)
    output.(names{k}) = b.(names{k});
end
end
