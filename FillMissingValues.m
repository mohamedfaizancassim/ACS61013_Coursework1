function [retArray,missing_perc]=FillMissingValues(inputArray)
nInputArray=size(inputArray);

missing_perc=[];
    for i=1:34
        TF=ismissing(inputArray(:,i));
        nMissingValues=sum(TF==1);
        missing_perc(end+1,:)=[(nMissingValues/nInputArray(1))*100];

        inputArray(:,i)=fillmissing(inputArray(:,i),"movmedian",10);
        inputArray(:,i)=array2table(round(table2array(inputArray(:,i))));
    end
    retArray=inputArray;
end