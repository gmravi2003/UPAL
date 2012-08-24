% Lets create a simple 2-d dataset, which is linearly separable by
% a vector  in 2-d

wstar=[1,1];
numtrn=100;
numtst=100;
generator=1;

if(generator==1)
    % Generate a point from 2-d gaussian.
    % mu=[0,0], sigma^2=I
    mu=[0,0];
    sigma=eye(2);
    % generate numtrn number of points and 
    % transpose it to get a dXn matrix.
    
    xtrn=(mvnrnd(mu,sigma,numtrn))';
    xtst=(mvnrnd(mu,sigma,numtst))';
    
    dlmwrite(numtrn,"train.txt");
    dlmwrite(numtst,"test.txt");
end

    
    
    
