function results = kmeans_digit_clustering(dataFile)
%KMEANS_DIGIT_CLUSTERING Explore unsupervised digit prototypes with K-means.
%
% OVERVIEW
%   Images are flattened into 784-dimensional observations and grouped into
%   ten clusters by minimizing within-cluster squared Euclidean distance.
%   K-means++ initialization, multiple restarts, and explicit empty-cluster
%   recovery make the optimization more reliable than a single random run.
%
%   Training labels are used only after clustering to interpret each cluster
%   by majority vote. This evaluation permits multiple clusters to describe
%   the same digit and does not invent a 50% error for an unrepresented
%   class. Such duplicate or missing semantic classes are genuine evidence
%   that geometric clusters do not necessarily align one-to-one with labels.

if nargin < 1
    dataFile = "data.mat";
end
[imageTrain, imageTest, labelTrain, labelTest] = load_digit_data(dataFile);
[height, width, nTrain] = size(imageTrain);
nTest = size(imageTest, 3);
d = height * width;

Xtrain = reshape(double(imageTrain), d, nTrain).';
Xtest = reshape(double(imageTest), d, nTest).';
scale = max(Xtrain(:));
if scale > 1
    Xtrain = Xtrain / scale;
    Xtest = Xtest / scale;
end

k = 10;
maxIterations = 200;
tolerance = 1e-7;
nRestarts = 8;
rng(1, 'twister');

bestObjective = inf;
restartObjectives = zeros(nRestarts, 1);
restartCentroids = cell(nRestarts, 1);
for restart = 1:nRestarts
    initialCentroids = initialize_kmeans_plus_plus(Xtrain, k);
    [assignments, centroids, objectiveHistory, emptyRecoveries] = ...
        run_kmeans(Xtrain, initialCentroids, maxIterations, tolerance);
    restartObjectives(restart) = objectiveHistory(end);
    restartCentroids{restart} = centroids;
    if restartObjectives(restart) < bestObjective
        bestObjective = restartObjectives(restart);
        bestAssignments = assignments;
        bestCentroids = centroids;
        bestHistory = objectiveHistory;
        bestEmptyRecoveries = emptyRecoveries;
        bestRestart = restart;
    end
end

% Attach a semantic digit label to each unsupervised cluster. Majority vote
% is transparent and does not force a false one-to-one cluster/class match.
clusterLabels = nan(k, 1);
clusterPurity = nan(k, 1);
clusterSizes = zeros(k, 1);
for cluster = 1:k
    members = labelTrain(bestAssignments == cluster);
    clusterSizes(cluster) = numel(members);
    counts = histcounts(members, -0.5:1:9.5);
    [largestCount, winner] = max(counts);
    clusterLabels(cluster) = winner - 1;
    clusterPurity(cluster) = largestCount / max(numel(members), 1);
end

% A new image is assigned to its nearest learned centroid and inherits the
% cluster's majority label.
testDistance2 = squared_distances(Xtest, bestCentroids);
[~, testCluster] = min(testDistance2, [], 2);
predictedLabels = clusterLabels(testCluster);
[perClassError, confusion] = classification_metrics( ...
    labelTest, predictedLabels, 0:9);
totalError = mean(predictedLabels ~= labelTest);

fprintf('K-means digit clustering\n');
fprintf('Selected restart: %d of %d | Objective: %.4g\n', ...
    bestRestart, nRestarts, bestObjective);
fprintf('Empty-cluster recoveries in selected run: %d\n', bestEmptyRecoveries);
fprintf('Majority-label test accuracy: %.2f%%\n', 100 * (1 - totalError));
for cluster = 1:k
    fprintf('Cluster %2d -> digit %d | size %4d | purity %.1f%%\n', ...
        cluster, clusterLabels(cluster), clusterSizes(cluster), ...
        100 * clusterPurity(cluster));
end

figure('Name', 'K-means Digit Prototypes');
tiledlayout(2, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
for cluster = 1:k
    nexttile;
    imagesc(reshape(bestCentroids(cluster, :), height, width));
    axis image off; colormap gray;
    title(sprintf('C%d -> %d (%.0f%%)', cluster, ...
        clusterLabels(cluster), 100 * clusterPurity(cluster)));
end

figure('Name', 'K-means Diagnostics');
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile; plot(bestHistory, 'o-', 'LineWidth', 1.2);
xlabel('Iteration'); ylabel('Within-cluster SSE');
title('Convergence'); grid on;
nexttile; bar(1:nRestarts, restartObjectives);
xlabel('Restart'); ylabel('Final SSE');
title('Initialization sensitivity'); grid on;
nexttile; bar(0:9, 100 * perClassError);
xlabel('True digit'); ylabel('Error rate (%)');
title('Downstream classification'); grid on;

% Compare centroid sets from the two best restarts. Small nearest-centroid
% distances indicate stable solutions, while large distances reveal local
% minima or different subdivisions of a digit class.
[~, order] = sort(restartObjectives);
if nRestarts >= 2
    centroidComparison = sqrt(squared_distances( ...
        restartCentroids{order(1)}, restartCentroids{order(2)}));
    figure('Name', 'Centroid Stability Across Best Restarts');
    imagesc(centroidComparison); axis image; colorbar;
    xlabel('Centroids from second-best restart');
    ylabel('Centroids from best restart');
    title('Pairwise Euclidean distance between centroids');
else
    centroidComparison = [];
end

results = struct( ...
    'centroids', bestCentroids, ...
    'trainingAssignments', bestAssignments, ...
    'clusterLabels', clusterLabels, ...
    'clusterPurity', clusterPurity, ...
    'predictedLabels', predictedLabels, ...
    'totalError', totalError, ...
    'perClassError', perClassError, ...
    'confusionMatrix', confusion, ...
    'objectiveHistory', bestHistory, ...
    'restartObjectives', restartObjectives, ...
    'centroidComparison', centroidComparison);
end

function centroids = initialize_kmeans_plus_plus(X, k)
n = size(X, 1);
centroids = zeros(k, size(X, 2));
centroids(1, :) = X(randi(n), :);
closestDistance2 = squared_distances(X, centroids(1, :));
for cluster = 2:k
    weights = closestDistance2 / sum(closestDistance2);
    if ~all(isfinite(weights)) || sum(weights) == 0
        index = randi(n);
    else
        index = find(cumsum(weights) >= rand, 1, 'first');
    end
    centroids(cluster, :) = X(index, :);
    newDistance2 = squared_distances(X, centroids(cluster, :));
    closestDistance2 = min(closestDistance2, newDistance2);
end
end

function [assignments, centroids, objectiveHistory, emptyRecoveries] = ...
        run_kmeans(X, centroids, maxIterations, tolerance)
k = size(centroids, 1);
assignments = zeros(size(X, 1), 1);
objectiveHistory = zeros(maxIterations, 1);
emptyRecoveries = 0;

for iteration = 1:maxIterations
    distance2 = squared_distances(X, centroids);
    [pointLoss, newAssignments] = min(distance2, [], 2);
    objectiveHistory(iteration) = sum(pointLoss);

    newCentroids = zeros(size(centroids));
    reserved = false(size(X, 1), 1);
    for cluster = 1:k
        members = newAssignments == cluster;
        if any(members)
            newCentroids(cluster, :) = mean(X(members, :), 1);
        else
            % Re-seed an empty cluster at a poorly represented observation.
            candidates = pointLoss;
            candidates(reserved) = -inf;
            [~, farthest] = max(candidates);
            newCentroids(cluster, :) = X(farthest, :);
            reserved(farthest) = true;
            newAssignments(farthest) = cluster;
            emptyRecoveries = emptyRecoveries + 1;
        end
    end

    centroidShift = norm(newCentroids - centroids, 'fro');
    assignmentStable = isequal(newAssignments, assignments);
    centroids = newCentroids;
    assignments = newAssignments;
    if assignmentStable || centroidShift <= tolerance * max(1, norm(centroids, 'fro'))
        objectiveHistory = objectiveHistory(1:iteration);
        return;
    end
end
objectiveHistory = objectiveHistory(1:maxIterations);
end

function distance2 = squared_distances(A, B)
distance2 = sum(A.^2, 2) + sum(B.^2, 2).' - 2 * (A * B.');
distance2 = max(distance2, 0);
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
