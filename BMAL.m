
% This code implements the algorithm from the paper 
% "Batch Mode Active Learning and Its Applications to Medical Image 
% Classification by Hoi et al. [ICML 2006]"
%@inproceedings{hoi2006batch,
 % title={Batch mode active learning and its application to medical image classification},
 % author={Hoi, S.C.H. and Jin, R. and Zhu, J. and Lyu, M.R.},
 % booktitle={Proceedings of the 23rd international conference on Machine learning},
 % pages={417--424},
 % year={2006},
 % organization={ACM}
%}


%% First step is to read the data
wrkspcflagb=exist('xtrn_bmal');
display(wrkspcflagb);
if(~wrkspcflagb)
    % Read data
    display('Training and test data NOT provided for BMAL');
    display('Hence reading data');
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
   
    % Normalize the data points. It is enough to normalize the train set.
    
    xtrn_bmal=xtrn*diag(1./sqrt(sum(xtrn.^2)));
    %BUDGET=min(5*ceil(sqrt(numtrn)),numtrn);
    BUDGET=150;
    display(BUDGET);
    options=optimset('Display','off','GradObj','on',...
                     'LargeScale','off','TolFun',10^-5,...
                     'MaxFunEvals',50000);
    delta_bmal=0.0001;
else
    display('Training data provided for BMAL');
    numtrn=size(xtrn_bmal,2);
    numtst=size(xtst,2);
    numdims=size(xtrn_bmal,1);
end


%% Second step is the actual algorithm

% The entire set of unqueried points
unqrdpnts_bmal=true(numtrn,1);

% The prelimiaries. Since BMAL works with logistic loss
LOSS=@(arg) log1p(exp(-arg));
BMALGradient= @(arg) -1.0./(1+exp(arg));
EMPRISKBMAL= @(w,n,x,y) sum((1/n)*LOSS(w'*(x*diag(y)))');


w_bmal=zeros(numdims,1);
trnerror_bmal_queries=zeros(BUDGET,1);
tsterror_bmal_queries=zeros(BUDGET,1);

numpntsqrd_bmal=0;


% First calculate X^TX
X_trans_X_esqd=(xtrn_bmal'*xtrn_bmal).^2;
%display('Finished forming the matrix X_trans_X_esqd');
%display(size(X_trans_X_esqd));


subsample_size=300;
for iter=1:BUDGET
    
    % Get pi_vec
    pi_vec=1./(1+exp(w_bmal'*xtrn_bmal)); 
    
    % We shall subsample from the set of unqueried points in order to speed
    % up the algorithm. Let this subsample size be approximately equal to
    % BUDGET
    
    indices=find(unqrdpnts_bmal);
    randompermvec=randperm(length(indices));
    indices=indices(randompermvec(1:subsample_size));
    
    maxval=-Inf;
    pi_one_minus_pi=pi_vec.*1-pi_vec;
    for uq=1:length(indices) % This is a maximization loop
        % Remove from \bar{S}
        unqrdpnts_bmal(indices(uq))=false;
        temp_mat=X_trans_X_esqd(unqrdpnts_bmal,~unqrdpnts_bmal);
   
        denom_vec=temp_mat*(pi_one_minus_pi(~unqrdpnts_bmal))'+delta_bmal;
        
        sumval=-sum(pi_one_minus_pi(unqrdpnts_bmal)'./denom_vec); %This calculates f(S_N).
        
        if(sumval>maxval)
            maxval=sumval;
            selectedindx=indices(uq);
        end
        % Revert back
        unqrdpnts_bmal(indices(uq))=true;
    end
            
    numpntsqrd_bmal=iter;
    unqrdpnts_bmal(selectedindx)=false;
    
    % Get w_bmal which corresponds to the weight vector obtained with the
    % new set of points
    
    % The optimization problem.
    xtrnqrd=xtrn_bmal(:,~unqrdpnts_bmal);
    ytrnqrd=ytrn(~unqrdpnts_bmal);
  
                                     
    BMALGRAD=@(w) 2*delta_bmal*w+1/(numpntsqrd_bmal)*(xtrnqrd*(ytrnqrd.*...
                            BMALGradient(w'*xtrnqrd*diag(ytrnqrd))'));
                               
    BMALOBJGRAD=@(w) deal(...
            EMPRISKBMAL(w,numpntsqrd_bmal,xtrnqrd,ytrnqrd)+...
                                         delta_bmal*norm(w,2)^2,... 
            2*delta_bmal*w+1/(numpntsqrd_bmal)*(xtrnqrd*...
                        (ytrnqrd.*BMALGradient(w'*xtrnqrd*diag(ytrnqrd))')));
        
    % Find out active learning vector.                        
    w_bmal=fminunc(BMALOBJGRAD,w_bmal,options);
    
    % Now test and train
    
    trnerror_bmal_queries(numpntsqrd_bmal)=sum(sign(w_bmal'*xtrnqrd)~=ytrnqrd')/(numpntsqrd_bmal);
    
    tsterror_bmal_queries(numpntsqrd_bmal)=sum(sign(w_bmal'*xtst)~=ytst')/numtst;
end

