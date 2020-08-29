%--------------------------------------------------------------------------
clc;clear all;close all;
d1=['EVTCalib\'];
addpath(d1);
mat = dir([d1,'*Test.mat']);
mat2=dir([d1,'*Val.mat']);
EVTandRF=0;
outputF1=[];
nTrees=200;
% Threshold to filter out with respect to the number of trees
threshold=0.9;
threshold=threshold*nTrees;
% Run through all the thrshold
ScenarioNmae='CIFAR_OpenSet';
mkdir(ScenarioNmae);
% Run through all the mat files of the experiments
for experiment=1:length(mat)
    % Load the calibration dataset results
    fileVal=strcat(mat2(experiment).name);
    load(fileVal);
    % Save the calibration confidence values which is the number of
    % trees for each class
    confidence_check=confidence;clear confidence outputRF;
    % Load the test data set--> Loads the confidence and the output
    % from RF for test dataset
    fileTest=strcat(mat(experiment).name);
    load(fileTest);
    outputF1=zeros(1,5000);
    outputknown=7*ones(1,5000);
    % For each clas go through the test dataset and get statictics
    for classRun=1:6
        % The class to check
        ClassCheck=classRun;
        % Threshold for each EVT distribution
        ConidenceRatioEVT=0.5;
        % Test dataset sample numbers
        CheckLimit=5000;
        %Unknown Data number
        LimitOtherData=3001:CheckLimit;
        % Normal Thresholding
        SumKnown=0;
        SumUnkown=0;
        SumKnownWrong=0;
        % EVT Based Thresholding
        SumKnownEVT=0;
        SumUnknownEVT=0;
        SumKnownWrongEVT=0;
        % Only EVT
        SumKnownUnkown=0;
        SumKnownUnkwonEVT=0;
        % Slect the appropriate data IDS for each class
        switch ClassCheck
            case 1
                limits=[1,CheckLimit];
                limits_claas=[1,500];
                limits_val=[1,500];
                classCheck=1;
            case 2
                limits=[1,CheckLimit];
                limits_claas=[501,1000];
                limits_val=[501,1000];
                classCheck=2;
            case 3
                limits=[1,CheckLimit];
                limits_claas=[1001,1500];
                limits_val=[1001,1500];
                classCheck=3;
            case 4
                limits=[1,CheckLimit];
                limits_claas=[1501,2000];
                limits_val=[1501,2000];
                classCheck=4;
            case 5
                limits=[1,CheckLimit];
                limits_claas=[2001,2500];
                limits_val=[2001,2500];
                classCheck=5;
            case 6
                limits=[1,CheckLimit];
                limits_claas=[2501,3000];
                limits_val=[2501,3000];
                classCheck=6;
            case 7
                limits=[1,CheckLimit];
                limits_claas=[3001,3500];
                limits_val=[3001,3500];
                classCheck=7;
            case 8
                limits=[1,CheckLimit];
                limits_claas=[3501,4000];
                limits_val=[3501,4000];
                classCheck=8;
            case 9
                limits=[1,CheckLimit];
                limits_claas=[4001,4500];
                limits_val=[4001,4500];
                classCheck=9;
            case 10
                limits=[1,CheckLimit];
                limits_claas=[4501,5000];
                limits_val=[4501,5000];
                classCheck=10;
        end
        % Limit the tailsize depending on the number of minimum trees
        % to be selected
        LimitTailSize=sum(confidence_check(limits_val(1):limits_val(2))<threshold);
        if(LimitTailSize==0)
            LimitTailSize=sum(confidence_check(limits_val(1):limits_val(2))<0.8*nTrees);
        end
                
        % Sort the set S_k and save the IDs and sorted array
        [ConfSort,idxSort]=sort(confidence_check(limits_val(1):limits_val(2)),'ascend');
        % The \tau number of datapoints for which the EVT has to be
        % fitted
        ConfFit=ConfSort(1:LimitTailSize);
        % Fit EVT
        parmhat =  wblfit(ConfFit);
        % First filter to remove the highly confident known datapoints
        for j=limits(1):limits(2)
            ConfFilterEVT = wblcdf(confidence((j)),parmhat(1),parmhat(2),0);
            %Check if the output is the class and the IDs are within
            %the class numbers
            % TP-> Known classified as Known
            if(outputRF(1,j)==classCheck && j>=limits_claas(1)&&j<=limits_claas(2))
                % First filter
                % SumKnown-->TP
                if(ConfFilterEVT>=ConidenceRatioEVT)
                    outputF1(j)=1;
                    outputknown(j)=classCheck;
                end
                % For data points that are outside the ID range of the class
                % is considered unknown
                % check if the class is classified correctly as the
                % classRub
            elseif(outputRF(1,j)==classCheck && j>=LimitOtherData(1)&&j<=LimitOtherData(end))
                % If highly confident then
                % TN -> Unknown classified as known-->SumKnownUnkown
                if(ConfFilterEVT>=ConidenceRatioEVT)
                    outputF1(j)=1;
                    outputknown(j)=classCheck;
                end
                % if the testsamples are from known but classified wrongly
            elseif(outputRF(1,j)==classCheck)
                if(ConfFilterEVT>=ConidenceRatioEVT)
                    outputF1(j)=1;
                    outputknown(j)=classCheck;
                end
            end
            
        end
        
        
    end
    outputTrue=[ones(1,500),...
        2*ones(1,500),...
        3*ones(1,500),...
        4*ones(1,500),...
        5*ones(1,500),...
        6*ones(1,500),...
        7*ones(1,500),...
        7*ones(1,500),...
        7*ones(1,500),...
        7*ones(1,500)];    
    save(strcat(ScenarioNmae,'\fcheck',num2str(experiment)),'outputF1','outputknown','outputTrue');
end
