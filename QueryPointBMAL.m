% This code queries a point for BMAL, by calculating f(S_N)-f(S), where 
% S_N= S\union \{x\}.

% Get a vector of probabilities of the point being labeled -1.
pi_vec=1./(1+exp(w_bmal'*xtrn_bmal)); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For each currently unqueried point
    % add to currently queried points
    % calculate the value
% end


    





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for uq=1:length(unqrdpnts_bmal) % For each unqueried point
    % get the unqueried point
    unqindx=unqrdpnts_bmal(uq);
    % Form the set S_N=qrdpnts_bmal\union \{x\}, NotS_N=unqrdpnts_bmal-\{x}
    
    % Add all points in qrdpnts_bmal to S_N and the point 
    S_N(numpntsqrd_bmal+1)=unqindx;

    % Remove the point at uq
    NotS_N(uq)=[];
    % Calculations begin now
    temp_mat=original_mat;   
    
    % Remove a row
    temp_mat(uq:uq,:)=[];
    % The last column needs to be filled with appropriate values.
    temp_mat(:,end+1:end+1)=...
        (xtrn_bmal(:,unqindx)'*xtrn_bmal(:,NotS_N))';
    
    denom_vec=temp_mat.^2*(pi_vec(S_N).*1-pi_vec(S_N))'+delta_bmal;
   
    numer_vec=pi_vec(NotS_N).*(1-pi_vec(NotS_N));
    
    sumval=-sum(numer_vec'./denom_vec);
    
    if(sumval>maxval)
        selectedindx_data=unqindx;
        maxval=sumval;
        selectedindx_in_unqrdpnts_vec=uq;
    end  
    %Restore back 
    NotS_N=unqrdpnts_bmal;
end