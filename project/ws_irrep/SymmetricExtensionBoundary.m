%%  SYMMETRICEXTENSION    Determines whether or not an operator has a symmetric extension
%   This function has one required argument:
%     X: a positive semidefinite matrix
%
%   EX = SymmetricExtension(X) is either 1 or 0, indicating that X does or
%   does not have a 2-copy symmetric extension. The extension is always
%   taken on the second subsystem of X.
%
%   This function has five optional arguments:
%     K (default 2)
%     DIM (default has both subsystems of equal dimension)
%     PPT (default 0)
%     BOS (default 0)
%     TOL (default eps^(1/4))
%
%   [EX,WIT] = SymmetricExtension(X,K,DIM,PPT,BOS,TOL) determines whether
%   or not X has a symmetric extension and provides a witness WIT that
%   verifies the answer. If a symmetric extension of X exists
%   (i.e., EX = 1) then WIT is such a symmetric extension. If no symmetric
%   extension exists (i.e., EX = 0) then WIT is an entanglement witness
%   with trace(WIT*X) = -1 but trace(WIT*Y) >= 0 for all symmetrically
%   extendable Y.
%
%   K is the desired number of copies of the second subsystem. DIM is a
%   1-by-2 vector containing the dimensions of the subsystems on which X
%   acts. PPT is a flag (either 0 or 1) indicating whether the desired
%   symmetric extension must have positive partial transpose. BOS is a flag
%   (either 0 or 1) indicating whether the desired symmetric extension must
%   be Bosonic (i.e., be supported on the symmetric subspace). TOL is the
%   numerical tolerance used when determining whether or not a symmetric
%   extension exists.
%
%   URL: http://www.qetlab.com/SymmetricExtension

%   requires: cvx (http://cvxr.com/cvx/), IsPPT.m, IsPSD.m, opt_args.m,
%             PartialTrace.m, PartialTranspose.m, PermutationOperator.m,
%             PermuteSystems.m, sporth.m, SymmetricProjection.m
%
%   author: Nathaniel Johnston (nathaniel@njohnston.ca), with improvements
%           by Mateus Ara�jo
%   package: QETLAB
%   last updated: January 5, 2023

function ret = SymmetricExtensionBoundary(X,varargin)

lX = length(X);

% set optional argument defaults: k=2, dim=sqrt(length(X)), ppt=0, bos=0, tol=eps^(1/4)
[k,dim,ppt,bos,tol] = opt_args({ 2, round(sqrt(lX)), 0, 0, eps^(1/4) },varargin{:});

% allow the user to enter a single number for dim
if(length(dim) == 1)
    dim = [dim,lX/dim];
    if abs(dim(2) - round(dim(2))) >= 2*lX*eps
        error('SymmetricExtension:InvalidDim','If DIM is a scalar, it must evenly divide length(X); please provide the DIM array containing the dimensions of the subsystems.');
    end
    dim(2) = round(dim(2));
end
if (k<=1)
    error('SymmetricExtensionBoundary:InvalidK','K must be greater than 1.');
end

sdp_dim = [dim(1),dim(2)*ones(1,k)];
% For Bosonic extensions, it suffices to optimize over the symmetric
% subspace, which is smaller.
if(bos)
    sdp_prod_dim = dim(1)*nchoosek(k+dim(2)-1, dim(2)-1);
else
    sdp_prod_dim = dim(1)*dim(2)^k;
end
rho0 = eye(dim(1)*dim(2))/(dim(1)*dim(2));
cvx_begin sdp quiet
    cvx_precision(tol);
    if(bos)
        V = kron(speye(dim(1)),SymmetricProjection(dim(2),k,1)); %this is an isometry
        variable sig(sdp_prod_dim,sdp_prod_dim) hermitian
        rho = V*sig*V';
    else
        variable rho(sdp_prod_dim,sdp_prod_dim) hermitian
    end
    variable alpha_i
    maximize(alpha_i)
    if(nargout > 1) % don't want to compute the dual solution in general (especially not if this is called within CVX)
        dual variable W
    end
    subject to
        if(nargout > 1)
            W : PartialTrace(rho,3:k+1,sdp_dim) == rho0*(1-alpha_i)+alpha_i*X;
        else
            PartialTrace(rho,3:k+1,sdp_dim) == rho0*(1-alpha_i)+alpha_i*X;
        end
        if(ppt)
            for j=2:k+1
                PartialTranspose(rho,2:j,sdp_dim) >= 0;
            end
        end
        if(bos)
            sig >= 0;
        else
            rho >= 0;
            for j = 3:k+1% in the permutation invariant case we need to explicitly enforce the symmetry
                PartialTrace(rho,setdiff(2:k+1,j),sdp_dim) == rho0*(1-alpha_i)+alpha_i*X;
            end
        end
cvx_end
ret = cvx_optval;
end
