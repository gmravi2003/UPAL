% Active Learning with random sampling w/o replacement

%% Read data
wrkspcinitflag=exist('xtrn');
if(~ wrkspcinitflag)
    display('Training and test data NOT provided');
    trnfile=...
        '~/matlab_codes/iwal/whitewine/whitewine_train_0.txt';
    tstfile=...
        '~/matlab_codes/iwal/whitewine/whitewine_test_0.txt';

    % The data is arranged column wise. Hence the data is d x n
    % d= num of features, n=num of points.

    datatrn=dlmread(trnfile);
    datatst=dlmread(tstfile);

    
    % Now remove the first row as these have the labels

    ytrn=datatrn(1:1,:)';
    ytst=datatst(1:1,:)';

    xtrn=datatrn(2:end,:);
    xtst=datatst(2:end,:);
    numtrn=size(xtrn,2);
    numtst=size(xtst,2);
    numdims=size(xtrn,1);
    BUDGET=400;
    options=optimset('Display','off','GradObj','on','LargeScale','off','TolFun',10^-5);
    lambda_random=0.1;
    lossstr='logistic';
else
    %display('Training and test data provided');
    % We already have data
    numtrn=size(xtrn,2);
    numtst=size(xtst,2);
    numdims=size(xtrn,1);
end

%% Setup necessary variables


% Train and test error

trnerrrandqrs=zeros(BUDGET,1);
tsterrrandqrs=zeros(BUDGET,1);

if(strcmp(lossstr,'logistic'))
    LOSS =@(arg) log1p(exp(-arg)); % LOGISTIC LOSS
else
    if(strcmp(lossstr,'exponential'))
        LOSS=@(arg) exp(-arg);
    else
        if(strcmp(lossstr,'sqd'))
            % For squared losses we have closed form updates.
            % Hence we won't use any matlab optimizers to solve the
            % problem. However we need these variables for our calculations
            % Initial parameter settings
            Mrand=(lambda_random/2)*eye(numdims);
            vrand=zeros(numdims,1);
        end 
    end
end

%%%%% The gradients%%%%%%%

if(strcmp(lossstr,'logistic'))
    LossGradient= @(arg) -1.0./(1+exp(arg));
else
    if(strcmp(lossstr,'exponential'))
        LossGradient= @(arg) -exp(-arg);
    else
        if(strcmp(lossstr,'sqd'))
            % For squared losses, we have closed form solutions. Hence we
            % won't use fminunc.
        end
    end
end

sampling_prob_rand=zeros(numtrn,1);
queries_bool_rand=false(numtrn,1);

w_rand=zeros(numdims,1);

for counter=1:BUDGET
    sampling_prob_rand=zeros(numtrn,1);   
    sampling_prob_rand(~queries_bool_rand)=1/(numtrn-counter+1);
    newpoint_index_rand=randp(sampling_prob_rand,1);
    queries_bool_rand(newpoint_index_rand)=true;
    decay_rand=1/sqrt(counter);
    if(strcmp(lossstr,'sqd'))
         M=(lambda_random*decay_rand/2)*eye(numdims)+...
             xtrn(:,queries_bool_rand)*...
                xtrn(:,queries_bool_rand)'/counter;
         v=xtrn(:,queries_bool_rand)*ytrn(queries_bool_rand);
         tsterrrandqrs(counter)=sum(sign((M\v)'*xtst)'~=ytst)/numtst;
    else
        EMPRAND=@(w,n,x,y) (1/n)*sum(LOSS(w'*(x*diag(y))));
        xqueried=xtrn(:,queries_bool_rand);
        yqueried=ytrn(queries_bool_rand);
       % RANDGRAD =@(w) lambda_random*w+...
        %        (1/counter)*(xqueried*diag(yqueried.*...
         %                                   LossGradient(w'*xqueried*...
          %                                      diag(yqueried))'))*...
           %                                         ones(counter,1);
        RANDOBJGRAD=@(w) deal((lambda_random*decay_rand/2)*norm(w,2)^2+...
                                EMPRAND(w,counter,xqueried,yqueried),... 
                                    lambda_random*decay_rand*w+(1/counter)*...
                                        (xqueried*(yqueried.*...
                                            LossGradient(w'*xqueried*...
                                                diag(yqueried))')));
                                    
         w_rand=fminunc(RANDOBJGRAD,w_rand,options);
         
         %normgrad=norm(RANDGRAD(w_rand));
         %if(normgrad>10^-5)
          %  display(normgrad);
         %end
         tsterrrandqrs(counter)=...
             sum(sign(w_rand'*xtst)'~=ytst)/numtst;      
    end
end
% Clear off the junk



%% Now perform unbiased active learning with random sampling
numpoints_queried=0;
importance_weights=zeros(numtrn,1);
queries_bool=false(numtrn,1);

trnerrrandunbiasedqrs=zeros(BUDGET,1);
tsterrrandunbiasedqrs=zeros(BUDGET,1);

trnerror_rand_unbiased_rounds=zeros(2*BUDGET,1);
tsterror_rand_unbiased_rounds=zeros(2*BUDGET,1);


counter=1;
w_rand_unbiased=zeros(numdims,1);

while numpoints_queried< BUDGET
    % We have U_{t-1}, L_{t-1}, v_{t-1}, M_{t-1}, w_AL(t-1) 
       
    %Sample a single point
    new_query_index=unidrnd(numtrn);
   
    % Can be non-zero if it was queried earlier
    
    importance_weights(new_query_index)=...
       importance_weights(new_query_index)+numtrn;
   
    queries_bool(new_query_index)=true;
    numpoints_queried=sum(queries_bool); 
    
    % Now we have the new point perform the following updates.
    
    if(strcmp(lossstr,'sqd'))
        % Update M. 
        constant2=1/(numtrn*counter);
        alpha_vec=importance_weights*constant2;
        decay_rand_unbiased=max(importance_weights)/sqrt(numpoints_queried);
        M=(lambda_random*decay_rand_unbiased/2)*eye(numdims)+xtrn*diag(alpha_vec)*xtrn';
        v=xtrn*(ytrn.*alpha_vec);
        tsterrrandunbiasedqrs(numpoints_queried)=...
            sum(sign(v'*(M\xtst))~=ytst');
        
        val=sum(sign(v'*(M\xtrn(:,queries_bool)))~=...
                        ytrn(queries_bool)')/numpoints_queried;
                    
        tsterrrandunbiasedqrs(numpoints_queried)=val;
        tsterror_rand_unbiased_rounds(counter)=val;
            
    else   % The loss function is NOT squared loss.
        
     
        xtemp=xtrn(:,queries_bool);
        ytemp=ytrn(queries_bool);
        EMPRANDUNBIASED=@(w,n,t,imp,x,y) (1/(n*t))*imp'*LOSS(w'*(x*diag(y)))';
        %RANDUNBIASEDGRAD=@(w) lambda_random*w+(1/(numtrn*counter))*(xtemp*diag(ytemp.*...
         %           importance_weights(queries_bool).*...
          %              LossGradient(w'*xtemp*diag(ytemp))')*ones(length(ytemp),1));
 
        decay_rand_unbiased=...
              norm(importance_weights)/(numtrn*counter*counter);
        RANDUNBIASEDOBJGRAD=@(w) deal(...
                EMPRANDUNBIASED(w,numtrn,counter,importance_weights(queries_bool),...
                    xtemp,ytemp)+(lambda_random*decay_rand_unbiased/2)*norm(w,2)^2,... 
            lambda_random*decay_rand_unbiased*w+1/(numtrn*counter)*(xtemp*(ytemp.*...
                        importance_weights(queries_bool).*...
                            LossGradient(w'*xtemp*diag(ytemp))')));
        % Find out active learning vector.                        
        w_rand_unbiased=fminunc(RANDUNBIASEDOBJGRAD,w_rand_unbiased,options);
        %normgrad=norm(RANDUNBIASEDGRAD(w_rand_unbiased));
        %if(normgrad>10^-5)
         %   display(normgrad);
        %end
       
        % Get the train and test errors of the active learner.
        temp2=sum(sign(w_rand_unbiased'*xtemp)~=ytemp')/numpoints_queried;
        trnerrrandunbiasedqrs(numpoints_queried)=temp2;
        trnerror_rand_unbiased_rounds(counter)=temp2;
        
        temp2=sum(sign(w_rand_unbiased'*xtst)~=ytst')/numtst;
        tsterrrandunbiasedqrs(numpoints_queried)=temp2;
        tsterror_rand_unbiased_rounds(counter)=temp2;
        
        % Find out the passive learning vector. 
    end
    counter=counter+1;
end

trnerror_rand_unbiased_rounds(counter:2*BUDGET)=[];
tsterror_rand_unbiased_rounds(counter:2*BUDGET)=[];
% Clear all the junk that we created