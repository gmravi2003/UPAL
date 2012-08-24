
%% Code assumes that VW files have already been generated but not processed
%   First task is to find out max number of queries over all folds
maxnumqueries=0;
totalnumqueries=0;
minnumqueries=1000000;
numfolds=5;
reg_coeff=0.1;
reg_coeff_pass=0.0001;
C=0.4*10^-4;
numqueries_vec=zeros(numfolds,1);

lossstr='logistic';
path1=['~/matlab_codes/iwal/whitewine/',lossstr,'_loss/fold_'];
display(numfolds)
for fold=1:numfolds
    filename=[path1,num2str(fold),'/numqueries.txt'];
    numqueries_vec(fold)=dlmread(filename);
    totalnumqueries=totalnumqueries+numqueries_vec(fold);
    if(maxnumqueries<numqueries_vec(fold))
        maxnumqueries=numqueries_vec(fold);
    end
    if(minnumqueries>numqueries_vec(fold))
        minnumqueries=numqueries_vec(fold);
    end
end
avgnumqueries=totalnumqueries/numfolds;

%avgtrnerror_vw=zeros(maxnumqueries,1);
avgtsterror_vw=zeros(maxnumqueries,1);

avgtrnerror_poolal_queries=zeros(maxnumqueries,1);
avgtsterror_poolal_queries=zeros(maxnumqueries,1);

avgtrnerror_pass_queries=zeros(maxnumqueries,1);
avgtsterror_pass_queries=zeros(maxnumqueries,1);

avgtrnerror_rand_unbiased_queries=zeros(maxnumqueries,1);
avgtrnerror_rand_queries=zeros(maxnumqueries,1);

avgtsterror_rand_unbiased_queries=zeros(maxnumqueries,1);
avgtsterror_rand_queries=zeros(maxnumqueries,1);


display(avgnumqueries);
display(minnumqueries);
display(maxnumqueries);

temp1=zeros(maxnumqueries,1);
% Now go over each fold.

if(strcmp(lossstr,'logistic'))
    avgtrnerror_bmal_queries=zeros(maxnumqueries,1);
    avgtsterror_bmal_queries=zeros(maxnumqueries,1);  
end
for fold=1:numfolds
    %% Process VW
    
    datapath='~/matlab_codes/iwal/whitewine/whitewine_';
    datatrn=dlmread([datapath,'train_',num2str(fold),'.txt']);
    datatst=dlmread([datapath,'test_',num2str(fold),'.txt']);
   
    ytrn=datatrn(1:1,:)';
    ytst=datatst(1:1,:)';

    xtrn=datatrn(2:end,:);
    xtst=datatst(2:end,:);
    
    numqueries=numqueries_vec(fold);  
    
    
    path1=['~/matlab_codes/iwal/whitewine/',lossstr,'_loss/fold_',num2str(fold)];           
    tsterror_vw=dlmread([path1,'/vw_test_raw_predictions/error_rate.txt']);
    temp1(1:numqueries)=tsterror_vw;
    temp1(numqueries+1:maxnumqueries)=tsterror_vw(end);
    avgtsterror_vw=avgtsterror_vw+temp1;
    options=optimset('Display','off','GradObj','on',...
                'LargeScale','off','TolFun',10^-5);
    %% Now Run PoolAL
    BUDGET=numqueries;
    PoolAL;
    %display(['Finished running PoolAL for fold ',num2str(fold)]);
    % Update quantities for PoolAL. We shall use a temporary variable
    % called temp1.
    
    
    
    temp1(1:numqueries)=trnerror_act_queries;
    temp1(numqueries+1:maxnumqueries)=trnerror_act_queries(end);
    avgtrnerror_poolal_queries=avgtrnerror_poolal_queries+temp1;
    
    
    temp1(1:numqueries)=tsterror_act_queries;
    temp1(numqueries+1:maxnumqueries)=tsterror_act_queries(end);
    avgtsterror_poolal_queries=avgtsterror_poolal_queries+temp1;   
    
    
    % Update quantities for Passive Learner
    
    temp1(1:numqueries)=trnerror_pass_queries;
    temp1(numqueries+1:maxnumqueries)=trnerror_pass_queries(end);
    avgtrnerror_pass_queries=avgtrnerror_pass_queries+temp1;
    
    
    temp1(1:numqueries)=tsterror_pass_queries;
    temp1(numqueries+1:maxnumqueries)=tsterror_pass_queries(end);
    avgtsterror_pass_queries=avgtsterror_pass_queries+temp1;
    
    % Clear variables
    clearvars -except avg* 'lossstr' 'numdims' 'numtrn' 'numtst' 'options'...
                'reg_coeff' 'reg_coeff_pass' 'xtrn' 'xtst' 'ytrn' 'ytst' 'fold' 'BUDGET' 'numqueries'....
                    'maxnumqueries' 'temp1' 'trn*' 'tst*' 'C' 'delta_bmal' 'path1' 'numqueries_vec' 'numfolds' 'err';

    
    %% Run Random Sampling code
    RandomSampling;
    
    % update quantities for unbiased random sampler
    temp1(1:numqueries)=trnerror_rand_unbiased_queries;
    temp1(numqueries+1:maxnumqueries)=trnerror_rand_unbiased_queries(end);
    avgtrnerror_rand_unbiased_queries=...
        avgtrnerror_rand_unbiased_queries+temp1;
    
    
    temp1(1:numqueries)=tsterror_rand_unbiased_queries;
    temp1(numqueries+1:maxnumqueries)=tsterror_rand_unbiased_queries(end);
    avgtsterror_rand_unbiased_queries=...
        avgtsterror_rand_unbiased_queries+temp1;
    
    % Update quantitites for random sampler
    temp1(1:numqueries)=trnerror_rand_queries;
    temp1(numqueries+1:maxnumqueries)=trnerror_rand_queries(end);
    avgtrnerror_rand_queries=...
        avgtrnerror_rand_queries+temp1;
    
    temp1(1:numqueries)=tsterror_rand_queries;
    temp1(numqueries+1:maxnumqueries)=tsterror_rand_queries(end);
    avgtsterror_rand_queries=...
        avgtsterror_rand_queries+temp1;
    
    display('Finished Random Sampling code...');
    clearvars -except avg* 'lossstr' 'numdims' 'numtrn' 'numtst' 'options'...
                'reg_coeff' 'reg_coeff_pass' 'xtrn' 'xtst' 'ytrn' 'ytst' 'fold' 'BUDGET' 'numqueries'...
    'maxnumqueries' 'temp1' 'trn*' 'tst*' 'C' 'delta_bmal' 'path1' 'numqueries_vec' 'numfolds' 'err';
    
    
    %% RUN BMAL if required
    if(strcmp(lossstr,'logistic'))
        delta_bmal=reg_coeff;
        xtrn_bmal=xtrn*diag(1./sqrt(sum(xtrn.^2)));
        BMAL;
        %display(['Finished running BMAL for fold ',num2str(fold)]);
      
        % Update the train error for BMAL
        
        temp1(1:numqueries)=trnerror_bmal_queries;
        temp1(numqueries+1:maxnumqueries)=trnerror_bmal_queries(end);
        avgtrnerror_bmal_queries=avgtrnerror_bmal_queries+temp1;
        
        % Update test error for BMAL
       
        temp1(1:numqueries)=tsterror_bmal_queries;
        temp1(numqueries+1:maxnumqueries)=tsterror_bmal_queries(end);
        avgtsterror_bmal_queries=avgtsterror_bmal_queries+temp1;
        clearvars -except avg* 'lossstr' 'numdims' 'numtrn' 'numtst' 'options'...
                'reg_coeff' 'reg_coeff_pass' 'xtrn' 'xtst' 'ytrn' 'ytst' 'fold'  'BUDGET'...
                    'numqueries' 'maxnumqueries' 'temp1' 'trn*' 'tst*' 'C'...
                        'delta_bmal' 'path1' 'numqueries_vec' 'numfolds' 'err';
    end
    
    
   %% Lets make some relevant plots
   
   if(strcmp(lossstr,'sqd'))
    
    plot([smooth(tsterror_act_queries),smooth(tsterror_vw),...
            smooth(tsterror_rand_queries),...
                smooth(tsterror_rand_unbiased_queries)...
                    smooth(tsterror_pass_queries)]);
    
    ylim([0.0,0.8])
    legend('AL: Tsterror', 'VW:Tsterror','Random:Tsterror','Random-unbiased:Tsterror',...
            'PL: Tsterror');
    title(['fold=',num2str(fold),' C=',num2str(C),' ','loss=',lossstr,' ' ...
                             'lambda=',num2str(reg_coeff)]);
    saveas(gcf,strcat(path1,'/fold_',num2str(fold),'_all','.fig'));

    % Save a plot comparing only the active learners.

    plot([smooth(tsterror_act_queries),smooth(tsterror_vw),...
            smooth(tsterror_rand_queries),...
                smooth(tsterror_rand_unbiased_queries)]);
    
    ylim([0.0,0.8])
    legend('AL: Tsterror', 'VW:Tsterror','Random:Tsterror',...
              'Random-unbiased:Tsterror','PL: Tsterror');
    title(['fold=',num2str(fold),' C=',num2str(C),' ','loss=',lossstr,' ' ...
                             'lambda=',num2str(reg_coeff)]);
    
    saveas(gcf,strcat(path1,'/fold_',num2str(fold),'_activelearners','.fig'));
   else
       % WE have results for BMAL too. Include them in your plots
        plot([smooth(tsterror_act_queries),smooth(tsterror_vw),...
                smooth(tsterror_bmal_queries),smooth(tsterror_rand_queries),...
                    smooth(tsterror_rand_unbiased_queries),...
                        smooth(tsterror_pass_queries)]);
    
        ylim([0.0,0.8])
        legend('AL: Tsterror', 'VW:Tsterror','Random:Tsterror',...
                    'Random-unbiased:Tsterror','BMAL:Tsterror','PL: Tsterror');
        title(['fold=',num2str(fold),' C=',num2str(C),' ','loss=',lossstr,' ' ...
                             'lambda=',num2str(reg_coeff)]);
        saveas(gcf,strcat(path1,'/fold_',num2str(fold),'_all','.fig'));

        % Save a plot comparing only the active learners.

         plot([smooth(tsterror_act_queries),smooth(tsterror_vw),...
                smooth(tsterror_bmal_queries),smooth(tsterror_rand_queries),...
                    smooth(tsterror_rand_unbiased_queries),...
                        smooth(tsterror_pass_queries)]);
    
        ylim([0.0,0.8])
        legend('AL: Tsterror', 'VW:Tsterror','Random:Tsterror',...
                    'Random-unbiased:Tsterror','BMAL:Tsterror','PL: Tsterror');
        title(['fold=',num2str(fold),' C=',num2str(C),' ','loss=',lossstr,' ' ...
                             'lambda=',num2str(reg_coeff)]);
        saveas(gcf,strcat(path1,'/fold_',num2str(fold),'_activelearners','.fig'));
       
        display(strcat('Made the plot for fold',num2str(fold)));
        display(strcat('Print finished fold ',num2str(fold)));
   end
end

% Finally normalize all quantities to get true averages
avgtrnerror_poolal_queries=avgtrnerror_poolal_queries/numfolds;
avgtsterror_poolal_queries=avgtsterror_poolal_queries/numfolds;

avgtsterror_vw=avgtsterror_vw/numfolds;

avgtrnerror_pass_queries=avgtrnerror_pass_queries/numfolds;
avgtsterror_pass_queries=avgtsterror_pass_queries/numfolds;

avgtsterror_rand_queries=...
    avgtsterror_rand_queries/numfolds;

avgtsterror_rand_unbiased_queries=...
    avgtsterror_rand_unbiased_queries/numfolds;

if(strcmp(lossstr,'logistic'))
    avgtsterror_bmal_queries=avgtsterror_bmal_queries/numfolds;
    avgtrnerror_bmal_queries=avgtrnerror_bmal_queries/numfolds;
end

%%%%%%%%%%%

figpath=['~/matlab_codes/iwal/whitewine/',lossstr,'_loss/'];
% 1a) Plot of average test error w.r.t number of unique queries.

if(strcmp(lossstr,'sqd'))
    plot([smooth(avgtsterror_poolal_queries),smooth(avgtsterror_vw),...
                smooth(avgtsterror_rand_queries),...
                    smooth(avgtsterror_rand_unbiased_queries),...
                            smooth(avgtsterror_pass_queries)]);
    ylim([0,0.5]);            
    legend('AL: Tsterror', 'VW:Tsterror','Random:Tsterror',...
                'Random-unbiased:Tsterror''PL: Tsterror');
    title(['Average performance: C=',num2str(C),' loss=',lossstr, ...
                                ' lambda=',num2str(reg_coeff)]);

    saveas(gcf,[figpath,'averaged_testerror_queries','.fig']);
else
    plot([smooth(avgtsterror_poolal_queries),smooth(avgtsterror_vw),...
           smooth(avgtsterror_rand_queries),...
                    smooth(avgtsterror_rand_unbiased_queries),...
                        smooth(avgtsterror_bmal_queries),...
                            smooth(avgtsterror_pass_queries)]);
    ylim([0.05,0.30]);            
    legend('AL: Tsterror', 'VW:Tsterror','Random:Tsterror',...
                'Random-unbiased:Tsterror','BMAL:Tsterror','PL: Tsterror');
    title(['Average performance: C=',num2str(C),' loss=',lossstr, ...
                                ' lambda=',num2str(reg_coeff)]);

    saveas(gcf,[figpath,'averaged_testerror_queries','.fig']);
    
end

% 1b) Make a plot of only the active learners. This is merely for clarity.

if(strcmp(lossstr,'sqd'))
    plot([smooth(avgtsterror_poolal_queries),smooth(avgtsterror_vw),...
                smooth(avgtsterror_rand_queries),...
                    smooth(avgtsterror_rand_unbiased_queries)]);
    ylim([0,0.5]);            
    legend('AL: Tsterror', 'VW:Tsterror','Random:Tsterror',...
                'Random-unbiased:Tsterror');
    title(['Average performance: C=',num2str(C),' loss=',lossstr, ...
                                ' lambda=',num2str(reg_coeff)]);
    saveas(gcf,[figpath,'averaged_testerror_activelearners_queries','.fig']);
else
    plot([smooth(avgtsterror_poolal_queries),smooth(avgtsterror_vw),...
           smooth(avgtsterror_rand_queries),...
                    smooth(avgtsterror_rand_unbiased_queries),...
                        smooth(avgtsterror_bmal_queries)]);
    legend('AL: Tsterror', 'VW:Tsterror','Random:Tsterror',...
                'Random-unbiased:Tsterror','BMAL:Tsterror');
    title(['Average performance: C=',num2str(C),' loss=',lossstr, ...
                                ' lambda=',num2str(reg_coeff)]);
    ylim([0.05,0.30]);
    saveas(gcf,[figpath,'averaged_testerror_activelearners_queries','.fig']);
end   

%-----

% 2) Make a plot for the train error and test error of PoolAL w.r.t number of rounds


plotmat=[smooth(fold0tsterror_poolal_rounds),smooth(fold0trnerror_poolal_rounds)];
plot(numpoints_queried_rounds',plotmat);
ylim([0,0.5]);            
legend('AL: Tsterror', 'AL:Trnerror');
title(['Performance of AL on entire dataset',' loss=',lossstr, ...
                            'lambda=',num2str(reg_coeff)]);
saveas(gcf,[figpath,'poolal','.fig']);
                
%-------

% 3) Make a plot of the test error w.r.t number of unique queries for all
% algorithms on the fold0 (i.e. the entire dataset)

if(strcmp(lossstr,'sqd'))

    plot([smooth(fold0tsterror_poolal_queries),smooth(fold0tsterror_vw),...
                    smooth(fold0tsterror_rand_queries),...
                        smooth(fold0tsterror_rand_unbiased_queries),... 
                            smooth(fold0tsterror_pass_queries)]);
    ylim([0,0.5]);            
    legend('AL: Tsterror', 'VW:Tsterror','Random:Tsterror',...
                'Random-unbiased:Tsterror','PL: Tsterror');
    title(['Fold0 performance: C=',num2str(C),' loss=',lossstr, ...
                                ' lambda=',num2str(reg_coeff)]);
    saveas(gcf,[figpath,'fold0_testerror_queries','.fig']);
else
    
    plot([smooth(fold0tsterror_poolal_queries),smooth(fold0tsterror_vw),...
            smooth(fold0tsterror_rand_queries),...
                smooth(fold0tsterror_rand_unbiased_queries),...
                    smooth(fold0tsterror_bmal_queries),...
                                smooth(fold0tsterror_pass_queries)]);
    ylim([0.05,0.30]);            
    legend('AL: Tsterror', 'VW:Tsterror','Random:Tsterror',...
                'Random-unbiased:Tsterror','BMAL:Tsterror','PL: Tsterror');
    title(['Fold0 performance: C=',num2str(C),' loss=',lossstr, ...
                                ' lambda=',num2str(reg_coeff)]);
    saveas(gcf,[figpath,'fold0_testerror_queries','.fig']);
    
end

% Save stuff to a .mat file

save([figpath,'results.mat']);

%%%%%%%%%%%%%%%%%%%%%%% THE END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
