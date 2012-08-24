%% This code simply does passive learning.

wrkspcinitflag=exist('xtrn');
if(~ wrkspcinitflag)

    N_TO_USE=1;
    LASTN=maxNumCompThreads(N_TO_USE);

    display('Training and test data NOT provided in PoolAL');
    trnfile=...
        '~/matlab_codes/iwal/mnist/mnist_train_0.txt';
    tstfile=...
        '~/matlab_codes/iwal/mnist/mnist_test_0.txt';

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
    
    %%%%%%%%%%%%%%%%%%%% 

    numtrn=size(xtrn,2);
    numtst=size(xtst,2);
    numdims=size(xtrn,1);
    BUDGET=300;

    display(BUDGET);    
    performfullpassive=false;
    % Create an optimization structure of options
    options=optimset('Display','off','GradObj','on',...
                'LargeScale','off','TolFun',10^-5);
    
    regcoeff_pass=0.0001;
    lossstr='logistic';

    stream1 = RandStream('mt19937ar','Seed',1);
    RandStream.setDefaultStream(stream1);    
else
    
    % We already have data
    numtrn=size(xtrn,2);
    numtst=size(xtst,2);
    numdims=size(xtrn,1);
end

check=exist('performfullpassive');
if(~check)
    performfullpassive=false;
end

%% ---- Step 2: Algorithm begins -----

tic;
% Vectors measuring test/train error of PL

trnerrpassqrs=zeros(BUDGET,1);
tsterrpassqrs=zeros(BUDGET,1);
numpntsqrd=0;

% Define various loss functions
if(strcmp(lossstr,'logistic'))
    LOSS =@(arg) log1p(exp(-arg)); % LOGISTIC LOSS
end
if(strcmp(lossstr,'exponential'))
    LOSS=@(arg) exp(-arg);
end

EMPRISKPASS=@(w,n,x,y) (1/n)*sum(LOSS(w'*(x*diag(y))));


%%%%% The gradients%%%%%%%

if(strcmp(lossstr,'logistic'))
    LossGradient= @(arg) -1.0./(1+exp(arg));
end

if(strcmp(lossstr,'exponential'))
    LossGradient= @(arg) -exp(-arg);
end

if(strcmp(lossstr,'sqd'))
    % For squared losses, we have closed form solutions. Hence we
    % won't use fminunc.
end

%%%%%%%%%%%%%%%%%

w_pass=zeros(numdims,1);

for t=1:BUDGET
    decay_pass=0.5/(sqrt(t));
    if(strcmp(lossstr,'sqd'))
        % Update M. 
        % Get the passive learning vector if a new query was indeed made.
        
        Mpass=(regcoeff_pass*decay_pass/2)*eye(numdims)+...
              (1/t)*xtrn(:,1:t)*xtrn(:,1:t)';
        
        vpass=(1/t)*(xtrn(:,1:numpntsqrd)*...
                     ytrn(1:numpntsqrd));
        
        w_pass=Mpass\vpass;
        
        temp=sum(sign(w_pass'*xtst)~=ytst')/numtst;
        tsterrpassqrs(t)=temp;
        
        temp=sum(sign(w_pass'*xtrn(:,1:t))~=...
                 ytrn(1:t)')/t;
        trnerrpassqrs(t)=temp;
    end
    
    if(strcmp(lossstr,'logistic'))
        
        PASSOBJ= @(w) EMPRISKPASS(w,t,xtrn(:,1:t),ytrn(1:t))+...
                 (regcoeff_pass*decay_pass/2)*norm(w,2)^2;
        
        xtemp=xtrn(:,1:t);
        ytemp=ytrn(1:t);
        PASSGRAD=@(w) regcoeff_pass*decay_pass*w+...
                 (1/t)*(xtemp*(ytemp.*LossGradient(w'*xtemp*diag(ytemp))'));   
        
        PASSOBJGRAD=@(w) deal(...
            EMPRISKPASS(w,t,xtemp,ytemp)+(regcoeff_pass*decay_pass/2)*norm(w,2)^2,...
            regcoeff_pass*decay_pass*w+(1/t)*(xtemp*(ytemp.*...
                    LossGradient(w'*xtemp*diag(ytemp))')));   
                                             
        w_pass=fminunc(PASSOBJGRAD,w_pass,options);
        
        temp=sum(sign(w_pass'*xtrn(:,1:t))~=ytrn(1:t)')/t;
        trnerrpassqrs(t)=temp;
        
        temp=sum(sign(w_pass'*xtst)~=ytst')/numtst;
        tsterrpassqrs(t)=temp;
        
    end
end

%% Run it on the entire dataset xtrn.
if(performfullpassive)
    decay_pass=1/sqrt(numtrn);
    if(strcmp(lossstr,'sqd'))
        
        w_pass=((1/numtrn)*(xtrn*xtrn')+...
               (lambda_pass*decay_pass/2)*eye(numdims))\((1.0/numtrn)*(xtrn*ytrn));
        
        % Now calculate the labels at all the test points and calculate the
        % missclassification error.
        finaltsterrpass=sum(sign(w_pass'*xtst)'~=ytst)/numtst;
        finaltrnerrpass=sum(sign(w_pass'*xtrn)'~=ytrn)/numtrn;
    end
    if(strcmp(lossstr,'logistic'))
        
       PASSOBJ=@(w) EMPRISKPASS(w,numtrn,xtrn,ytrn)+(lambda_pass*decay_pass/2)*norm(w,2)^2;
       
       PASSGRAD=@(w) regcoeff_pass*decay_pass*w+(1/numtrn)*(xtrn*(ytrn.*...
                                                         LossGradient(w'*xtrn*diag(ytrn))')); 
       PASSOBJGRAD=@(w) deal(...
                    EMPRISKPASS(w,numtrn,xtrn,ytrn)+(regcoeff_pass*decay_pass/2)*norm(w,2)^2,...
                    regcoeff_pass*decay_pass*w+(1/numtrn)*(xtrn*(ytrn.*...
                        LossGradient(w'*xtrn*diag(ytrn))')));   
                                                
        w0=zeros(numdims,1);    
        w_pass=fminunc(PASSOBJGRAD,w0,options);
        %passgradopt=PASSGRAD(w_pass);
        %normpassgradopt=norm(passgradopt);                   
        
        %  Calculate train and test error.
        finaltsterrpass=sum(sign(w_pass'*xtst)'~=ytst)/numtst;
        finaltrnerrpass=sum(sign(w_pass'*xtrn)'~=ytrn)/numtrn;
    end
end
toc;