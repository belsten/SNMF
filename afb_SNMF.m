function [W,H,newerr] = afb_SNMF(V, rdim, lamb)
    % non negative matrix factorization with sparse coding
    % [W,H] = afb_NNSC(V, rdim);
    %
    % INPUTS:
    % V          - data matrix (N x T)
    % rdim       - (M) number of inner components 
    % lamb       - sparsity constraint, larger -> more weight put on sparsity
    % 
    % OUTPUTS:
    % W           - first non-negative factor in V~WH decomposition (N x rdim)
    % H           - second non-negative factor in V~WH decomposition (rdim x T)
    % newerr      - latest estimate of error of in estimation
    %
    % based on "Sparse coding and NMF" by Julian Eggert and Edgar Korner ( 2004 )
    
    %% defaults and data check

    % Check that we have non-negative data
    if min(V(:))<0, error('Negative values in data!'); end

    % Globally rescale data to avoid potential overflow/underflow
    V = V/max(V(:));
    rng(1997)
    % strictly positive initial initialization
    W = abs(randn(size(V,1),rdim)); 
    H = abs(randn(rdim,size(V,2)));
    
    %% do the learning of the sparse matrices, initilize errors 
    n_itr = 500;
    itr = 0;
    
    while (itr < n_itr)
        itr = itr + 1;
        
        % resecale columns of W such that they are unit norm
        W = W./(ones(size(V,1),1)*sqrt(sum(W.^2,1)));
        
        % reconstruction of V
        R = W*H;
        
        % update sparse activations
        H = H.*((W'*V)./(W'*R + lamb));
        
        % recompute reconstruction
        R = W*H;

        % update W (non-parametrically)
        for j=1:size(W,2)  % columns 
%             % verbatim algo described in paper, for verification of
%             % optimized 
%             num = zeros(size(V,1),1); den = zeros(size(V,1),1);
%             for i=1:size(V,1)
%                num = num + H(j,i)*(V(i,:)' + (R(i,:)*W(:,j))*W(:,j));
%                den = den + H(j,i)*(R(i,:)' + (V(i,:)*W(:,j))*W(:,j));
%             end
            
            num = (V' + ((R*W(:,j))*ones(1,size(V,1)))'.*W(:,j))*H(j,:)';
            den = (R' + ((V*W(:,j))*ones(1,size(V,1)))'.*W(:,j))*H(j,:)';
            W(:,j) = W(:,j).*(num./den);
        end
    end
    
    [~,pp]=sort(sum(W.^2,1),'descend');
    W=W(:,pp);H=H(pp,:);
    newerr = norm(abs(V-W*H));
end
