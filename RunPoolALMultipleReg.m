% This script will run PoolAL code on different regularization
reg_coeff_vec=[10000];
num_repeats=2;
num_folds=3;
lossstr='logistic';
BUDGET=700;
reg_coeff_pass=0.1;
finaltrnerroract_mat=zeros(num_folds,length(reg_coeff_vec));
finaltsterroract_mat=zeros(num_folds,length(reg_coeff_vec));
%finaltrnerrorpass_mat=zeros(num_folds,length(reg_coeff_vec));
%finaltsterrorpass_mat=zeros(num_folds,length(reg_coeff_vec));
display(reg_coeff_vec);

for f=1:num_folds %For each fold
     
    trnfile=strcat('~/matlab_codes/iwal/magic/magic_train_',num2str(f),'.txt');
    tstfile=strcat('~/matlab_codes/iwal/magic/magic_test_',num2str(f),'.txt');
    % The data is arranged column wise. Hence the data is d x n
    % d= num of features, n=num of points.

    datatrn=dlmread(trnfile);
    datatst=dlmread(tstfile);
    % Now remove the first row as these have the labels
    ytrn=datatrn(1:1,:)';
    ytst=datatst(1:1,:)';

    xtrn=datatrn(2:end,:);
    xtst=datatst(2:end,:);

    for i=1:length(reg_coeff_vec) % For each regularization value
        round=1;
        finaltrnerroract=0;
        finaltsterroract=0;
        %finaltrnerrorpass=0;
        %finaltsterrorpass=0;
        totalpasses=0;
        passes_vec=zeros(num_repeats,1);
        reg_coeff=reg_coeff_vec(i);
        
        for repeats=1:num_repeats
            
            PoolAL;
            finaltsterroract=finaltsterroract+final_tsterror_act;
            finaltrnerroract=finaltrnerroract+final_trnerror_act;

            %finaltsterrorpass=finaltsterrorpass+final_tsterror_pass;
            %finaltrnerrorpass=finaltrnerrorpass+final_trnerror_pass;
            passes_vec(repeats)=t; % CAUTION: Make sure to see that the variable 't'
                                   % hasn't been used before.
            
        end
        finaltsterroract_mat(f,i)=finaltsterroract/num_repeats;
        finaltrnerroract_mat(f,i)=finaltrnerroract/num_repeats;
        %finaltrnerrorpass_mat(f,i)=finaltrnerrorpass/num_repeats;
        %finaltsterrorpass_mat(f,i)=finaltsterrorpass/num_repeats;
        mean_passes=mean(passes_vec);
        std_passes=std(passes_vec);
        
        display(strcat('fold:',num2str(f),' reg=',num2str(reg_coeff_vec(i)),...
                        ' avgpasses=',num2str(mean_passes),' std passes=',num2str(std_passes)));
    end
end

% Average over all folds.
avgtsterroract_mat=ones(1,num_folds)*finaltsterroract_mat/num_folds;
avgtrnerroract_mat=ones(1,num_folds)*finaltrnerroract_mat/num_folds;

%avgtsterrorpass_mat=ones(1,num_folds)*finaltsterrorpass_mat/num_folds;
display(avgtsterroract_mat);
%display(avgtsterrorpass_mat);
display(avgtrnerroract_mat);
[minval,index]=min(avgtsterroract_mat);
reg_opt=reg_coeff_vec(index);
display(reg_opt);
display('Optimal test error of AL is');
display(minval);

display('train error of AL at optimal settings is');
display(avgtrnerroract_mat(index));

display(reg_coeff_vec);

%%% Finally run the passive learner to know what is the best reg_coeff for
%%% passive learning.


% NOTE this piece of code is being run only for 1 fold which is the last
% fold. That should be enough

reg_coeff_pass_vec=[0.001,0.01,0.1,1.0,10.0,100];

minval=+Inf;
decay_pass=1/sqrt(numtrn);

for counter=1:length(reg_coeff_pass_vec)
    reg_coeff_pass=reg_coeff_pass_vec(counter);
    if(strcmp(lossstr,'sqd'))
        wpass=((1/numtrn)*(xtrn*xtrn')+...
            (reg_coeff_pass*decay_pass/2)*eye(numdims))\((1.0/numtrn)*(xtrn*ytrn));
    
        % Now calculate the labels at all the test points and calculate the
        % missclassification error.
        final_tsterror_pass=sum(sign(wpass'*xtst)'~=ytst)/numtst;
        final_trnerror_pass=sum(sign(wpass'*xtrn)'~=ytrn)/numtrn;
    else
        
       PASSOBJ=@(w) EMPRISKPASS(w,numtrn,xtrn,ytrn)+(reg_coeff*decay_pass/2)*norm(w,2)^2;
       
       PASSGRAD=@(w) reg_coeff_pass*decay_pass*w+(1/numtrn)*(xtrn*(ytrn.*...
                        LossGradient(w'*xtrn*diag(ytrn))')); 
       PASSOBJGRAD=@(w) deal(...
                    EMPRISKPASS(w,numtrn,xtrn,ytrn)+(reg_coeff_pass*decay_pass/2)*norm(w,2)^2,...
                    reg_coeff_pass*decay_pass*w+(1/numtrn)*(xtrn*(ytrn.*...
                        LossGradient(w'*xtrn*diag(ytrn))')));   
                                                
        w0=zeros(numdims,1);    
        wpass=fminunc(PASSOBJGRAD,w0,options);
       %passgradopt=PASSGRAD(wpass);
        %normpassgradopt=norm(passgradopt);                     
        
        % Calculate train and test error.
        final_tsterror_pass=sum(sign(wpass'*xtst)'~=ytst)/numtst;
        final_trnerror_pass=sum(sign(wpass'*xtrn)'~=ytrn)/numtrn;
    end
   
    if(minval>final_tsterror_pass)
        minval=final_tsterror_pass;
        
        opt_reg_coeff_pass=reg_coeff_pass_vec(counter);
    end
end
display('For passive learning....');
display(opt_reg_coeff_pass);
display('Optimal test error for passive learning....');
display(minval);


