% dim = 3;
% kext_list = [2,3,4];

dim = 2;
kext_list = [6,8,10];

use_ppt = 0;
tol = 1e-7; %precision comparable to that in mosek

dm0 = WernerState(dim,1);
ret = zeros(size(kext_list));
for ind0=1:size(kext_list,2)
    kext = kext_list(ind0);
    if dim==2
        use_bos=1;
    else
        use_bos=0;
    end
    t0 = tic;
    tmp0 = SymmetricExtensionBoundary(dm0, kext, [dim,dim], use_ppt, use_bos, tol);
    t1 = toc(t0);
    alpha_sdp = (tmp0*dim)/(tmp0+dim-1);
    % alpha_sdp = (tmp0 * dim*dim) / (1+tmp0*dim);
    alpha_analytical = (kext+dim*dim-dim)/(kext*dim+dim-1);
    disp([kext, alpha_sdp, abs(alpha_sdp-alpha_analytical), t1])
end

%% gtx3060, tol=1e-7
%% kext alpha abs(error) time
%% d=2
% 6   0.615384607623860   0.000000007760756   0.143685
% 8   0.588235285558861   0.000000008558786   0.18626
% 10   0.571428562566841   0.000000008861730  12.601661
%% d=3
% 2   0.999999994383922   0.000000005616078   0.185774
% 3   0.818181815857862   0.000000002323956   0.615825
% 4   0.714285710060162   0.000000004225552   7.965962
