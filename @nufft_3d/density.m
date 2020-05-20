%% density estimation
function [d sd dwsd] = density(obj,ok)

% Pipe's method (sort of)
maxit = 20;

% initial estimate (preserve zeros)
d = reshape(ok,[],1);

% iterative refinement: d = d ./ H*H*d
for j = 1:maxit

    q = obj.spmv(obj.spmv_t(d));
    d = d ./ max(q,eps(max(q)));

end

% this "density" is very sensitive to oversampling factor
% e.g. u=1 acts like a highpass filter and u=4 a lowpass.
% this is probably correct behaviour... the "density" of
% samples is calculated on the fine grid, so for fixed J
% the density flattens off (i.e. spokes stop interacting) 
% as we move away from the center of k-space. this makes
% it challenging to interpret d as "the" sample density.
% following Sedarat we will interpret d as a diagonal
% approximation to the full least squares problem. in a
% sane world, gridding and least squares should give the
% same ballpark answer.
%
%      Qx = k [for DFT matrix Q]
%       x = Q'*D*k [gridding D=diag(d)]
% Q'*D*Qx = Q'*D*k [least squares weighted by D^0.5]
% 
% since it's plain rude for the two recons to give very
% different answers, the modest goal is to scale d such
% that diag(Q'*D*D) is 1.
%
% we can get the k-th diagonal of (Q'*D*Q) by letting e =
% [0...0 1 0...0]' (1 at the k-th position) then computing
% v = Q'*D*Q*e using our current D. the k-th element of v 
% is k-th diagonal element (Q'*D*Q). E.g. if k=2 then
%
% v = [a1 a2 a3] [0] = [a2]
%     [a4 a5 a6] [1]   [a5] <---the k-th element
%     [a7 a8 a9] [0]   [a8]
%
% in Cartesian the diagonal of (Q'*D*Q) is the mean of the
% sample weighting squared (typically an average of 1s and
% 0s for present/absent). using that same idea we can get
% diag(Q'*D*Q) quickly from the center of aNUFT(d) aka the
% mean of FFT(aNUFT(d)).

% diagonal of Q'*D*Q with current d
q = obj.aNUFT(d);
if obj.N(3)==1
    dwsd = real(q(obj.N(1)/2+1,obj.N(2)/2+1));
else
    dwsd = real(q(obj.N(1)/2+1,obj.N(2)/2+1,obj.N(3)/2+1));
end

% the dwsd is hard to interpret other than diag(Q'*D*Q) but
% we can normalize for weighting to get the mean sampling
% density which is the diagonal of Q'*Q. an expensive way
% of calculating no. data points / no. matrix points!
sd = dwsd / mean(d);

% measure true diagonals of Q'*Q and Q'*D*Q (V. SLOW)
if false
    N = 100; % how many to test
    T = zeros(N,2);
    for j = 1:N
        tmp = zeros(size(obj.U)); tmp(j) = 1;
        tmp = obj.aNUFT(obj.fNUFT(tmp)); % Q'*Q: does not vary with J or u
        T(j,1) = tmp(j);
        tmp = zeros(size(obj.U)); tmp(j) = 1;
        tmp = obj.aNUFT(d.*obj.fNUFT(tmp)); % Q'*D*Q: varies with everything
        T(j,2) = tmp(j);       
        %fprintf('%i/%i\n',j,N);
    end
    T = cat(2,real(T),max(abs(imag(T)),[],2));
    plot(T); legend({'diag(Q''*Q','diag(Q=Q''*D*Q)','imag'});
    fprintf('  sd=%f dwsd=%f (measured)\n',mean(T(:,1)),mean(T(:,2)));
end

% normalize for dwsd, as promised, so that diag(Q'*D*Q) = 1. there
% remains a "mystery" factor of u so let's just remove that for now
% until maybe we figure out where is came from
d = d / dwsd / obj.u;

% whatever we did to d we must do to dwsd
dwsd = dwsd / dwsd / obj.u;

% report metrics
R = numel(d) / prod(obj.N);
fprintf(' Density: sd=%f dwsd=%f 1/R=%f\n',sd,dwsd,R);
