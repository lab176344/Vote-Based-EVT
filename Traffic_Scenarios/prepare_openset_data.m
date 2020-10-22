clc;clear;close all;

load('HighDScenario.mat');
DataNum = 800;

ogLaneRight = shuffle_dataset(ogLaneRight);
ogLaneLeft = shuffle_dataset(ogLaneLeft);
ogLeftCut = shuffle_dataset(ogLeftCut);
ogLeftCutOut = shuffle_dataset(ogLeftCutOut);
ogRightCut = shuffle_dataset(ogRightCut);
ogFollDec = shuffle_dataset(ogFollDec);
ogFollAcc = shuffle_dataset(ogFollAcc);
ogRightCutOut = shuffle_dataset(ogRightCutOut);
ogFoll = shuffle_dataset(ogFoll);

TData = 600;
TeData = TData+1;
Tlimit = 150;
%Training Data
X_train = [ogLaneRight(1:TData,:,:,:)];
y_train = zeros(TData,1);
X_train = [X_train; ogLaneLeft(1:TData,:,:,:)];
y_train = [y_train; 1*ones(TData,1)];
X_train = [X_train; ogLeftCut(1:TData,:,:,:)];
y_train = [y_train; 2*ones(TData,1)];
X_train = [X_train; ogLeftCutOut(1:TData,:,:,:)];
y_train = [y_train; 3*ones(TData,1)];
X_train = [X_train; ogRightCut(1:TData,:,:,:)];
y_train = [y_train; 4*ones(TData,1)];
X_train = [X_train; ogFollDec(1:TData-200,:,:,:)];
y_train = [y_train; 7*ones(TData-200,1)];
X_train = [X_train; ogFollAcc(1:TData-200,:,:,:)];
y_train = [y_train; 5*ones(TData-200,1)];
X_train = [X_train; ogRightCutOut(1:TData,:,:,:)];
y_train = [y_train; 6*ones(TData,1)];
X_train = [X_train; ogFoll(1:TData,:,:,:)];
y_train = [y_train; 8*ones(TData,1)];


X_test = [ogLaneRight(TeData:TData+Tlimit,:,:,:)];
y_test = zeros(size(X_test,1),1);
X_test = [X_test; ogLaneLeft(TeData:TData+Tlimit,:,:,:)];
y_test = [y_test; 1*ones(size(ogLaneLeft(TeData:TData+Tlimit,:,:,:),1),1)];
X_test = [X_test; ogLeftCut(TeData:TData+Tlimit,:,:,:)];
y_test = [y_test; 2*ones(size(ogLeftCut(TeData:TData+Tlimit,:,:,:),1),1)];
X_test = [X_test; ogLeftCutOut(TeData:TData+Tlimit,:,:,:)];
y_test = [y_test; 3*ones(size(ogLeftCutOut(TeData:TData+Tlimit,:,:,:),1),1)]; 
X_test = [X_test; ogRightCut(TeData:TData+Tlimit,:,:,:)];
y_test = [y_test; 4*ones(size(ogRightCut(TeData:TData+Tlimit,:,:,:),1),1)]; 
X_test = [X_test; ogFollDec(TeData-200:end,:,:,:)];
y_test = [y_test; 7*ones(size(ogFollDec(TeData-200:end,:,:,:),1),1)]; 
X_test = [X_test; ogFollAcc(TeData-200:end,:,:,:)];
y_test = [y_test; 5*ones(size(ogFollAcc(TeData-200:end,:,:,:),1),1)]; 
X_test = [X_test; ogRightCutOut(TeData:TData+Tlimit,:,:,:)];
y_test = [y_test; 6*ones(size(ogRightCutOut(TeData:TData+Tlimit,:,:,:),1),1)];
X_test = [X_test; ogFoll(TeData:TData+Tlimit,:,:,:)];
y_test = [y_test; 8*ones(size(ogFoll(TeData:TData+Tlimit,:,:,:),1),1)];


X_train1 = X_train(1:2500,:,:,:);
X_train2 = X_train(2501:end,:,:,:);

save('HighDScenarioClassV1_Train.mat','X_train1',...
    'X_train2','y_train')
save('HighDScenarioClassV1_Test.mat','X_test','y_test');

function [Ogs_shuffled]=shuffle_dataset(Ogs)
rand_pos = randperm(size(Ogs,1)); %array of random positions
% new array with original data randomly distributed
    Ogs_shuffled = Ogs(rand_pos,:,:,:);
end



