% Unbiased Pool based Active Learning (UPAL.)
% This code does unbiased pool based active learning.

%% --- Step 1: Read train and test data-----
wrkspcinitflag=exist('xtrn');
if(~ wrkspcinitflag)

    N_TO_USE=1;
    LASTN=maxNumCompThreads(N_TO_USE);

    display('Training and test data NOT provided in PoolAL');
    trnfile=...
        '~/matlab_codes/iwal/abalone/abalone_train_0.txt';
    tstfile=...
        '~/matlab_codes/iwal/abalone/abalone_test_0.txt';

    % The data is arranged column wise. Hence the data is d x n
    % d= num of features, n=num of points.
    
    datatrn=dlmread(trnfile);
    datatst=dlmread(tstfile);

    
    % Now remove the first row as these have the labels

    ytrn=datatrn(1:1,:)';
    ytst=datatst(1:1,:)';

    xtrn=datatrn(2:end,:);
    xtrn=xtrn*diag(1./sqrt(sum(xtrn.^2)));
    xtst=datatst(2:end,:);

    %%%% THIS IS ONLY FOR SCALABILITY%%%

    SIZE=1200;
    xtrn=xtrn(:,1:SIZE);
    ytrn=ytrn(1:SIZE);
    display(SIZE);
    %%%%%%%%%%%%%%%%%%%% 

    numtrn=size(xtrn,2);
    numtst=size(xtst,2);
    numdims=size(xtrn,1);
    BUDGET=300;
    explrexpupal=1/4;
    strategy_upal='old';
    display(BUDGET);   
    % Create an optimization structure of options
    options=optimset('Display','off','GradObj','on',...
                'LargeScale','off','TolFun',10^-5);
    
    lambda_upal=0.001;
    lossstr='logistic';
    stream1 = RandStream('mt19937ar','Seed',1);
    RandStream.setDefaultStream(stream1);
else
    numtrn=size(xtrn,2);
    numtst=size(xtst,2);
    numdims=size(xtrn,1);
    options=optimset('Display','off','GradObj','on',...
                     'LargeScale','off',); 
    options.MaxFunEvals=1000;
    options.MaxIter=500;
end
    

%% ---- Step 2: Algorithm begins -----

% The importance weights on the points. For points that were not queried
% these weights are simply 0.
importance_weights=zeros(numtrn,1);

% A boolean vector indicating what points were queried
queries_bool=false(numtrn,1);


% Vectors measuring test/train error of AL,PL

trnerrupalqrs=zeros(BUDGET,1);
tsterrupalqrs=zeros(BUDGET,1);

numpntsqrd=0;

t=1;

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
            %M=(lambda_upal/2)*eye(numdims);
            %v=zeros(numdims,1);
            %Mpass=M;
            %vpass=v;
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

%%%%%%%%%%%%%%%%%

sampling_prob=zeros(numtrn,1);
w_al=zeros(numdims,1);

while numpntsqrd < BUDGET
    % We have U_{t-1}, L_{t-1}, v_{t-1}, M_{t-1}, w_AL(t-1) 
    
    if(t==1)
        % In the first round simply random sample
        sampling_prob=(1.0/numtrn)*ones(numtrn,1);
        
    else % Calculate the sampling probability                           
        temp_weight=zeros(1,numtrn);
        p_min=1.0/(numtrn*t^explrexpupal);
        
        if(strcmp(lossstr,'logistic'))
            if(strcmp(strategy_upal,'old'))
                condprob_vec=1./(1+exp(-w_al'*xtrn));
                entropy_vec=-condprob_vec.*log(condprob_vec)-...
                    (1-condprob_vec).*log(1-condprob_vec);
                entropy_vec(isnan(entropy_vec))=0;
                sumentropy=sum(entropy_vec);
                if(sumentropy~=0)
                    sampling_prob=p_min+...
                        (entropy_vec/sumentropy)*(1-numtrn*p_min);      
                else
                    display(['All conditional probabilities are either 0 or ' ...
                             '1']);
                    break;
                end
            else
                % The strategy is new 
                condprob_vec=(1./(1+exp(-w_al'*xtrn(:,~queries_bool))))';
                entropy_vec=-condprob_vec.*log(condprob_vec)-...
                    (1-condprob_vec).*log(1-condprob_vec);
                entropy_vec(isnan(entropy_vec))=0;
                sumentropy=sum(entropy_vec);
                if(sumentropy~=0)
                    % For queried points just give p_min probability.    
                    sampling_prob(queries_bool)=p_min;
                    
                    % For the unqueried points do the old thing
                    sampling_prob(~queries_bool)=p_min+...
                        (entropy_vec/sumentropy)*(1-numtrn*p_min);
                else
                    display(['All conditional probabilities are either 0 or ' ...
                             '1']);
                    break;
                end
                
            end    
        end
        
        if(strcmp(lossstr,'sqd'))
            % Then do some simple calculations
            pred1=v'*(M\xtrn);
            p_plus=max(0,min((1+pred1)/2,1));
            p_minus=1-p_plus;
            temp_weight=p_plus*log(1./p_plus)+...
                p_minus*log(1./p_minus);
            sampling_prob=p_min+(1-(numtrn*p_min))*...
                (temp_weight/sum(temp_weight));
        end
    end
    %Sample a single point
    
    new_query_index=randp(sampling_prob,1,1);
    
    % Can be non-zero if it was queried earlier
    oldimportanceweight=importance_weights(new_query_index);
    importance_weights(new_query_index)=...
        oldimportanceweight+(1.0/(sampling_prob(new_query_index)));
    
    isnewquery=~queries_bool(new_query_index); % Set if we queried a new point
    queries_bool(new_query_index)=true;
    numpntsqrd=sum(queries_bool);
    
    % Now we have the new point perform the following updates.
    
    if(strcmp(lossstr,'sqd'))
        % Update M. 
        
        constant1=1/(numtrn*t);
        alpha_vec=importance_weights*constant1;
        decay_poolal=max(alpha_vec)/sqrt(numpntsqrd);
        M=(lambda_upal*decay_poolal/2)*eye(numdims)+xtrn*diag(alpha_vec)*xtrn';
        v=xtrn*(ytrn.*alpha_vec);
        temp=sum(sign(v'*(M\xtst))~=ytst');
        tsterrupalqrs(numpntsqrd)=temp;
        
        % Note we are calculating the trainerror on the dataset seen till
        % now, and NOT on the entire training dataset.
        temp=sum(sign(v'*(M\xtrn(:,queries_bool)))~=...
                        ytrn(queries_bool)')/numpntsqrd;
                    
        trnerrupalqrs(numpntsqrd)=temp;
        
        %Now calculate the test error of the current vector.
        temp=sum(sign(v'*(M\xtst))~=ytst')/numtst;
        tsterrupalqrs(numpntsqrd)=temp;     
        % Get the passive learning vector if a new query was indeed made.
    end
    
    if(strcmp(lossstr,'logistic')) 
        % The loss function is NOT squared loss.
        %ACTOBJ= @(w) EMPRISKACT(w,numtrn,t,...
        %                    importance_weights(queries_bool),...
        %                       xtrn(:,queries_bool),...
        %                          ytrn(queries_bool))+...
        %                             (lambda_upal/2)*norm(w,2)^2; 
        
        xy=xtrn(:,queries_bool)*diag(ytrn(queries_bool));
        imp=importance_weights(queries_bool);
        EMPRISKACT=@(w) (1/(numtrn*t))*imp'*LOSS(w'*(xy))';
        
        ststc=norm(importance_weights);
        %decay_poolal=ststc/(numtrn*t*t);
        new_ststc=sum(importance_weights);
        decay_poolal=1/new_ststc^(1/3);
        ACTOBJGRAD=@(w) deal(...
            EMPRISKACT(w)+(lambda_upal*decay_poolal/2)*norm(w,2)^2,... 
            lambda_upal*decay_poolal*w+...
            (1/(numtrn*t))*(xy*(LossGradient(w'*xy)'.*imp)));
        % Find out active learning vector.
        try                
          w_al=minFunc(ACTOBJGRAD,w_al,options);
        catch err
        end

        %ACTGRAD=@(w) lambda_upal*decay_poolal*w+(1/(numtrn*t))*(xtrn*diag(ytrn.*importance_weights.*...
         %            LossGradient(w'*xtrn*diag(ytrn))'))*ones(length(ytrn),1);
 
         %if(norm(ACTGRAD(w_al))>10^-4)
          %  display(norm(ACTGRAD(w_al)));
           % display([num2str(t),' :Possible error...']);
        %end
       
        % Get the train and test errors of the active learner.
        temp=sum(sign(w_al'*xtrn(:,queries_bool))~=...
                        ytrn(queries_bool)')/numpntsqrd;
        trnerrupalqrs(numpntsqrd)=temp;
        
        temp=sum(sign(w_al'*xtst)~=ytst')/numtst;
        tsterrupalqrs(numpntsqrd)=temp;
    end
    % DONT FORGET TO INCREMENT t
    
    %display(numpntsqrd);
    t=t+1;
end
finaltsterract=tsterrupalqrs(end);
finaltrnerrupal=trnerrupalqrs(end);
%display(['Number of rounds in PoolAL are ',num2str(t)]);
%% Code ends here..