airport_dataset=readtable("airport-quarterly-passenger-survey-1.csv");



%Splitting Date
dateRecorded=airport_dataset.DateRecorded;
airport_dataset.DateRecorded_Day=dateRecorded.Day;
airport_dataset.DateRecorded_Month=dateRecorded.Month;
airport_dataset.DateRecorded_Year=dateRecorded.Year;

%Splitting Time
departureTime=split(airport_dataset.DepartureTime,":");
airport_dataset.DepartureTime_Hour=cellfun(@str2num,departureTime(:,1));
airport_dataset.DepartureTime_Min=departureTime(:,2);

%Converting AM/PM formats into 24 hour dates
nDepartureTime=size(departureTime);
nDepartureTime=nDepartureTime(1);

for i=1:nDepartureTime
    if contains(airport_dataset.DepartureTime_Min(i),"AM")
        if airport_dataset.DepartureTime_Hour(i)==12
            airport_dataset.DepartureTime_Hour(i)=0;
        end
    end
    if contains(airport_dataset.DepartureTime_Min(i),"PM")
        if airport_dataset.DepartureTime_Hour(i)<=11
            airport_dataset.DepartureTime_Hour(i)=12+airport_dataset.DepartureTime_Hour(i);
        end
    end
end

%-----------------------------------------------------------------------------------------------

%Dropping Unwanted Collumns
airport_dataset.Quarter=[];
airport_dataset.DateRecorded=[];
airport_dataset.DepartureTime=[];
airport_dataset.DepartureTime_Min=[];

%Fill in missing values in the dataset
[airport_dataset,missing_perc]=FillMissingValues(airport_dataset);



%Rearrange the collumns in the table
airport_dataset=movevars(airport_dataset,"DateRecorded_Day","Before","GroundTransportationTo_fromAirport");
airport_dataset=movevars(airport_dataset,"DateRecorded_Month","Before","GroundTransportationTo_fromAirport");
airport_dataset=movevars(airport_dataset,"DateRecorded_Year","Before","GroundTransportationTo_fromAirport");
airport_dataset=movevars(airport_dataset,"DepartureTime_Hour","Before","GroundTransportationTo_fromAirport");

%Save Table to File
try
    writetable(airport_dataset,"airport_dataset_cleaned.csv","Delimiter",",");
catch ex
  display("File/Folder Access Prevention or File Already Exsists");
  display(ex.message);
end

%Split features and target
features=table2array(airport_dataset(:,1:37));
target=table2array(airport_dataset(:,38));

%==========================
%   Correlation Analy
%==========================
corr_=corr(features,target);
varNames=airport_dataset.Properties.VariableNames;

%Performing PCA
%Standardising the data
airport_data_standardized=bsxfun(@minus,features,mean(features))./std(features); 
airport_data_correlationMatrix=corrcoef(airport_data_standardized);

%Calculating the Eigen Vectors and Eigen Values
[eigen_vectors,eigen_values]=eig(airport_data_correlationMatrix);

eigen_vectors;
eigen_vectors=fliplr(eigen_vectors);
eigen_vectors=eigen_vectors';

eigen_values;
dataInPrincipalComponentSpace=eigen_vectors * airport_data_standardized';
dataInPrincipalComponentSpace=dataInPrincipalComponentSpace';

[coeff,score,latent,tsquared,explained,mu]=pca(airport_data_standardized);
%heatmap()
%hold off
varNames=airport_dataset.Properties.VariableNames';
varNames(end,:)=[];
heatmapTable=table(varNames,coeff);
%heatmap(coeff,'XLabel','Principal Components','YLabel','Variable Names','YData',varNames);
%TODO: 



MSE_Values=[];
f=waitbar(0,"Processing...");
for perc=10:100
    waitbar(perc/100,f,perc);
    split_amnt=round((perc/100)*nDepartureTime);

    features_lim=features(1:split_amnt,:);
    target_lim=target(1:split_amnt,:);
    
    %Extracting Test and Train rows

    split_percentage=70;
    train_ammount=(split_percentage/100)*split_amnt;
    test_ammount=((100-split_percentage)/100)*split_amnt;

    training_features=features_lim(1:train_ammount,:);
    training_target=target_lim(1:train_ammount,:);
    
    testing_features=features_lim(1:test_ammount,:);
    testing_target=target_lim(1:test_ammount,:);

    %=========================
    % Neural Network
    %=========================
    adaboost=fitctree(training_features,training_target,"MaxNumSplits",19);
    
    predictions_train=predict(adaboost,training_features);
    predictions_test=predict(adaboost,testing_features);

     RMSE_decisiontree_train=mean((training_target-predictions_train).^2);

    RMSE_decisiontree_test=mean((testing_target-predictions_test).^2);

  

    MSE_Values(end+1,:)=[perc,RMSE_decisiontree_train,RMSE_decisiontree_test];
end

plot(MSE_Values(:,1),MSE_Values(:,2));
hold on
plot(MSE_Values(:,1),MSE_Values(:,3));
legend("Training Data","Testing Data")
xlabel("Data %");
ylabel("RMSE");
title("Learning Curve");
















