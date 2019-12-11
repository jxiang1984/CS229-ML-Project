%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Dec. 8, 2019
% This script presents:
% (1) Train the dataset using the CNN classifier
% (2) Define the convolutional neural network architecture
% (3) Use k-fold cross-validation to evaluate the performance of the trained system
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; close all; clear variables;
Ver =1;

%% data pre-processing
load('AllData/PalletImages/RGBImages/PalletRGBImages/Pallet250size/CNNDataForPose.mat');
Data1 = RGBImages';
a = Data1{6};
imagesc(a)

load('AllData/PalletImages/RGBImages/PalletRGBImages/Pallet250size/DataForSecondCNN.mat');
Data2 = RGBImages';
% The whole data of class 1 (Pallet) which is 450 images
PalletData = [Data1; Data2(1:110,:)];
% The whole data of class 2 (No Pallet) which is 500 images
NoPalletData = Data2(111:end,:);
% The dataset for both classes
DataSet = [PalletData; NoPalletData];
% Extract both classes in order to find the training and the testing sets
PalletClass = DataSet(1:length(PalletData));
NoPalletClass = DataSet(length(PalletData)+1:end);
% Set the corresponding Class labels (Targets)
Labels = [ones(1,length(PalletClass)), -1*ones(1,length(NoPalletClass))]';


%% initial output file name
pathname = pwd;
new_directory = strcat(pathname,'/output/');
mkdir(new_directory);
fid = fopen(strcat(new_directory,'_TrainingParaOpt', num2str(Ver), '.csv'),'w');
% parameters: 6; results: 6x
fprintf(fid, '%s,%s, %s,%s, %s,%s,%s,%s,%s,%s,%s,%s \n','KFold', 'MaxEpochs','filterNum', 'filterSize','learningRate','batchSize', 'Av_Accuracy', 'Av_TruePos', 'Av_TrueNeg', 'Av_FalsePos', 'Av_FalseNeg', 'time');


%% Data training
% Set the parameters
KFold_range = [2];        
MaxEpochs_range = [1];
filterNum_range = [15];
filterSize_range = [20];
learningRate_range = [0.3];
batchSize_range = [50];

%% Parallel Computting trial
% % myPool = parpool(6);
% p = gcp;
% p.NumWorkers

for i=1:size(KFold_range,2)
    for j=1:size(MaxEpochs_range,2)
        for k=1:size(filterNum_range,2)
            for m=1:size(filterSize_range,2)
                for n=1:size(learningRate_range,2)
                    for p=1:size(batchSize_range,2)
                        KFold = KFold_range(1,i);
                        MaxEpochs = MaxEpochs_range(1,j);
                        filterNum = filterNum_range(1,k);
                        filterSize = filterSize_range(1,m);
                        learningRate = learningRate_range(1,n);
                        batchSize = batchSize_range(1,p);
                        % Create a pool with 2 workers.
                        output = CNN_train_core(DataSet, Labels, KFold,  MaxEpochs,filterNum, filterSize, learningRate, batchSize);
                        % write the Traininginfo in CSV file
                        fprintf(fid, '%s,%s, %s,%s, %s,%s,%s,%s,%s,%s,%s,%s \n', num2str(KFold),num2str(MaxEpochs),num2str(filterNum),num2str(filterSize), num2str(learningRate),num2str(batchSize),...
                            num2str(output(1)),num2str(output(2)),num2str(output(3)),num2str(output(4)),num2str(output(5)),num2str(output(6)));
                    end
                end
            end
        end
    end
end
fclose(fid);
aa =1;


% %% The next step after Kfold validation phase is to consider all data set for training.
%
% % convert 2D cell to 4D array. The 1st three dimensions must be the height,
% % width, and channels, the last one is index of individual images.
% DataSet4D = reshape(cat(3,DataSet{:}),ImSize,ImSize,Ch,length(DataSet));
% DataSet4D = im2double(DataSet4D);
% Labels = categorical(Labels);
%
% % Create a Convolutional Neural Network (CNN)
%
% % Define the convolutional neural network architecture
% layers = [imageInputLayer([ImSize ImSize Ch])
%         convolution2dLayer(20,25)
%         reluLayer
%         %convolution2dLayer(5,30) reluLayer
%         maxPooling2dLayer(5,'Stride',2)
%         fullyConnectedLayer(2)
%         softmaxLayer
%         classificationLayer()];
%
% options = trainingOptions('sgdm','MaxEpochs',MaxEpochs, ...
%     'InitialLearnRate',0.03, ...
%     'MiniBatchSize',50);
%
% % The network will be used for testing the new data (images).
% ConvNet = trainNetwork(DataSet4D,Labels,layers,options);


function output = CNN_train_core(DataSet, Labels, KFold, MaxEpochs,filterNum, filterSize, learningRate,batchSize)
tic;
ImSize = 250;           Ch = 3;
Indices = crossvalind('Kfold',Labels,KFold);
accuracy = zeros(KFold,1);
truePos = zeros(KFold,1);
trueNeg = zeros(KFold,1);
falsePos = zeros(KFold,1);
falseNeg = zeros(KFold,1);

for i = 1:KFold
    TestingIdx = find(Indices == i);                    TrainingIdx = find(Indices~=1);
    TestingData = DataSet(TestingIdx,:);                TrainingData = DataSet(TrainingIdx,:);
    TestingLabels = Labels(TestingIdx,:);               TrainingLabels = Labels(TrainingIdx,:);
    TestingLabels = categorical(TestingLabels);         TrainingLabels = categorical(TrainingLabels);
    
    % convert 2D cell to 4D array.
    % The 1st three dimensions must be the height, width, and channels, the last one is index of individual images.
    TrainData4D = reshape(cat(3,TrainingData{:}),ImSize,ImSize,Ch,length(TrainingData));
    TrainData4D = im2double(TrainData4D);
    TestData4D = reshape(cat(3,TestingData{:}),ImSize,ImSize,Ch,length(TestingData));
    TestData4D = im2double(TestData4D);
    
    %% Create a Convolutional Neural Network (CNN)
    % Define the convolutional neural network architecture
    layers = [imageInputLayer([ImSize ImSize Ch])
        convolution2dLayer(filterSize,filterNum) % 25= # of filters % 20# size of filter
        reluLayer
        %convolution2dLayer(5,30)
        %reluLayer
        maxPooling2dLayer(5,'Stride',2)
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer()];
    
    options = trainingOptions('sgdm','MaxEpochs',MaxEpochs, ...
        'InitialLearnRate',learningRate, ...
        'ExecutionEnvironment', 'auto',...
        'MiniBatchSize',batchSize);
    
    [ConvNet, traininfo] = trainNetwork(TrainData4D,TrainingLabels,layers,options);
    
    % Run the trained network on the test set
    YTest = classify(ConvNet,TestData4D);
    TTest = TestingLabels;
    
    correct = 0;
    pos_pos = 0; % True positive
    neg_neg = 0; % True negative
    pos_neg = 0; % False positive
    neg_pos = 0; % False negative
    
    % Calculate the metrics.
    for j = 1:numel(YTest)
        if YTest(j) == TTest(j)
            correct = correct + 1;
        end
        if double(YTest(j)) == 1
            if double(TTest(j)) == 1
                pos_pos = pos_pos + 1;
            else
                pos_neg = pos_neg + 1;
            end
        else
            if double(TTest(j)) == 1
                neg_pos = neg_pos + 1;
            else
                neg_neg = neg_neg + 1;
            end
        end
    end
    accuracy(i,:) = correct / numel(TTest);
    truePos(i,:) = pos_pos / numel(TTest);
    trueNeg(i,:) = neg_neg / numel(TTest);
    falsePos(i,:) = pos_neg / numel(TTest);
    falseNeg(i,:) = neg_pos / numel(TTest);
end
% Calculate the average accuracy of the kfold validation step.
Av_Accuracy = mean(accuracy);
Av_TruePos = mean(truePos);
Av_TrueNeg = mean(trueNeg);
Av_FalsePos = mean(falsePos);
Av_FalseNeg = mean(falseNeg);

fprintf('Average Accuracy: %.4f \n', Av_Accuracy);
fprintf('Average True Positive: %.4f \n', Av_TruePos);
fprintf('Average True Negative: %.4f \n', Av_TrueNeg);
fprintf('Average False Positive: %.4f \n', Av_FalsePos);
fprintf('Average False Negative: %.4f \n', Av_FalseNeg);

elapseTime = toc;
output = [Av_Accuracy Av_TruePos Av_TrueNeg Av_FalsePos Av_FalseNeg elapseTime];

end

