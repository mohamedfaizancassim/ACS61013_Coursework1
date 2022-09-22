housing=readtable("CW_dataset.csv");

%Convert Categorical Datatypes to Numberical Datatypes
housing.MSZoning_Numeric=grp2idx(housing.MSZoning);
housing.LotShape_Numeric=grp2idx(housing.LotShape);
housing.BldgType_Numeric=grp2idx(housing.BldgType);
housing.HouseStyle_Numeric=grp2idx(housing.HouseStyle);
housing.Foundation_Numeric=grp2idx(housing.Foundation);
housing.BsmtQual_Numeric=grp2idx(housing.BsmtQual);
housing.BsmtCond_Numeric=grp2idx(housing.BsmtCond);
housing.Heating_Numeric=grp2idx(housing.Heating);
housing.HeatingQC_Numeric=grp2idx(housing.HeatingQC);
housing.CentralAir_Numeric=grp2idx(housing.CentralAir);
housing.Electrical_Numeric=grp2idx(housing.Electrical);
housing.KitchenQual_Numeric=grp2idx(housing.KitchenQual);
housing.Fireplaces_Numeric=grp2idx(housing.Fireplaces);
housing.GarageQual_Numeric=grp2idx(housing.GarageQual);
housing.GarageCond_Numeric=grp2idx(housing.GarageCond);
housing.PoolQC_Numeric=grp2idx(housing.PoolQC);
housing.Fence_Numeric=grp2idx(housing.Fence);
housing.SaleCondition_Numeric=grp2idx(housing.SaleCondition);

%Deleting unessacary rows
housing_new=housing;
housing_new(:,3)=[];
housing_new(:,5:7)=[];
housing_new(:,7:14)=[];
housing_new(:,8:13)=[];

%Looking for missing values
TF=ismissing(housing_new.LotFrontage);
TF_index=find(~TF);
A=housing_new.LotFrontage(TF_index);
LotFrontage_median=median(A);
housing_new.LotFrontage=fillmissing(housing_new.LotFrontage,'constant',LotFrontage_median);

%Categorizing the SalePrice
nSalePrice=size(housing_new.SalePrice);
nSalePrice=nSalePrice(1);
nSalePrice=uint16(nSalePrice);

SalePrice_Categorical=[nSalePrice];
for c=1:nSalePrice
    if housing_new.SalePrice(c)<200000
        SalePrice_Categorical(c,1)=1;
    elseif housing_new.SalePrice(c)<400000
        SalePrice_Categorical(c,1)=2;
    elseif housing_new.SalePrice(c)<600000
        SalePrice_Categorical(c,1)=3;
    elseif housing_new.SalePrice(c)<700000
        SalePrice_Categorical(c,1)=4;
    else
        SalePrice_Categorical(c,1)=5;
    end   
end

%Drop SalePrice
housing_new.SalePrice=[];




%Drop ID
housing_new.Id=[];

MSE_Values=[];
for perc=1:1:100
    %Spliting Training and Validation Datasets

    split_perc=80;
    split_amnt=(split_perc/100)*nSalePrice;
    
    %Training Dataset
    training_features=housing_new(1:split_amnt,:);
    nTraining_features=size(training_features);
    nTraining_features=nTraining_features(1)*(perc/100);
    training_features=training_features(1:nTraining_features,:);

    training_labels=SalePrice_Categorical(1:split_amnt,:);
    nTraining_labels=size(training_labels);
    nTraining_labels=nTraining_labels(1)*(perc/100);
    training_labels=training_labels(1:nTraining_labels,:);

    training_labels_regression=housing.SalePrice(1:split_amnt,:);
    nTraining_labels_regression=size(training_labels_regression);
    nTraining_labels_regression=nTraining_labels_regression(1)*(perc/100);
    training_labels_regression=training_labels_regression(1:nTraining_labels_regression,:);
    training_labels_regression=normalize(training_labels_regression);

    %Validation Dataset
    testing_features=housing_new(split_amnt:nSalePrice,:);
    nTesting_features=size(testing_features);
    nTesting_features=nTesting_features(1)*(perc/100);
    testing_features=testing_features(1:nTesting_features,:);

    testing_labels=SalePrice_Categorical(split_amnt:nSalePrice,:);
    nTesting_labels=size(testing_labels);
    nTesting_labels=nTesting_labels(1)*(perc/100);
    testing_labels=testing_labels(1:nTesting_labels,:);

    testing_labels_regression=housing.SalePrice(split_amnt:nSalePrice,:);
    nTesting_labels_regression=size(testing_labels_regression);
    nTesting_labels_regression=nTesting_labels_regression(1)*(perc/100);
    testing_labels_regression=testing_labels_regression(1:nTesting_labels_regression,:);
    testing_labels_regression=normalize(testing_labels_regression);
    %====================
    %Decision Tree
    %====================
    ctree=fitctree(training_features,training_labels,"MaxNumSplits",13);
    %view(ctree,'mode','graph');
    predictions_dTree=predict(ctree,training_features);
    
    %cm=confusionchart(predictions_dTree,testing_labels);

   
    predictions_dTree_test=predict(ctree,testing_features);
    
    
    %Finding R2
    RMSE_decisiontree=sqrt(mean((training_labels-predictions_dTree).^2));
    RMSE_decisiontree_test=sqrt(mean((testing_labels-predictions_dTree_test).^2));

    %==================================
    %Multivariate Regression Analysis
    %==================================
    nTrainingData=size(training_features);
    nTrainingData=nTrainingData(1,1);
    
    Psi=[ones(nTrainingData(1,1),1),table2array(training_features)]; %Psi
    
    %Calculate theta_hat
    theta_hat=inv(Psi'*Psi)*Psi'*double(training_labels_regression);
    
    %Prediction
    nTestDataSize=size(testing_features);
    nTestDataSize=nTestDataSize(1,1);
    
    Psi_Star=[ones(nTestDataSize,1),table2array(testing_features)];
    predictions_regression=Psi_Star*theta_hat;

    MSE_Values(end+1,:)=[perc,RMSE_decisiontree,RMSE_decisiontree_test];
end



plot(MSE_Values(:,1),MSE_Values(:,2));
hold on
plot(MSE_Values(:,1),MSE_Values(:,3));
legend("Training Data","Testing Data")
xlabel("Data %");
ylabel("RMSE");
title("Learning Curve");






