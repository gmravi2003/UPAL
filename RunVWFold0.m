if(strcmp(lossstr,'logistic'))
    lossstrvw=lossstr;
end
if(strcmp(lossstr,'sqd'))
    lossstrvw='classic';
end

datapath='~/matlab_codes/iwal/whitewine/';
% Get ytst

% Note this piece of code is called only by ExptsFold0, and not by anyone
% else

cd ~/vowpal_wabbit;

%% First train on this fold
    
trndatapath=[datapath,'whitewine_train_0_vw.txt'];
%display(trndatapath);

vwtrncmd=['./vw --active_simulation --active_mellowness=',...
                 num2str(C),' -d ',' ',trndatapath,' --initial_t ',num2str(initial_t_val),' -l ',num2str(l_val),...
                    '  --quiet --loss_function=',lossstrvw];
display(vwtrncmd);
system(vwtrncmd);

%% The next thing to do is test

% Before that make a folder called fold and inside it should be vw_test_raw_predictions.
path=['~/matlab_codes/iwal/whitewine/'];
foldpath=[path,lossstr,'_loss/','fold_0'];
%display(foldpath);
mkdircmd1=['mkdir',' ',foldpath];
system(mkdircmd1);

rawpredictionspath=[path,lossstr,'_loss/','fold_0','/vw_test_raw_predictions'];
mkdircmd2=['mkdir',' ',rawpredictionspath];
system(mkdircmd2);


[~,out]=system(['ls -l',' ',path,lossstr,'_loss/vw_weights_*.txt','|wc -l']);
numqueries=str2num(out);
% Write the number of queries made to a file
dlmwrite([path,lossstr,'_loss/fold_0','/numqueries.txt'],numqueries);

testdatapath=[datapath,'whitewine_test_0_vw.txt'];
weightpath=[path,lossstr,'_loss/vw_weights_'];

error_vec=zeros(numqueries,1);
datatst=dlmread([datapath,'whitewine_test_0','.txt']);
ytst=datatst(1:1,:)';
numtst=size(ytst,1);
for num=1:numqueries
    testcmd=['./vw -t --quiet -d',' ',testdatapath,' -i ',...
                    weightpath,num2str(num),'.txt',' -r ',...
                    rawpredictionspath,'/vw_test_raw_predictions',...
                    '.txt'];
    
    system(testcmd);
    % Read the rawtest predictions.
    rtp=dlmread([rawpredictionspath,...
                                    '/vw_test_raw_predictions.txt']);
    error=sum(sign(rtp)~=sign(ytst))/numtst;
    error_vec(num)=error;
    % Remove the rawtestpredictions file that we generated.....
    rmcmd0=['rm -rf ',[rawpredictionspath,...
                        '/vw_test_raw_predictions.txt']];
    system(rmcmd0);
end
% Finally remove all the temporary weights that were formed.
rmcmd2=['rm -rf',' ',path,lossstr,'_loss','/vw_weights_*.txt'];
system(rmcmd2);

% Write the error rates
errorfilepath=[rawpredictionspath,'/error_rate.txt'];
dlmwrite(errorfilepath,error_vec);
