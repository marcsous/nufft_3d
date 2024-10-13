%% density estimation
function d = density(obj,ok)

% Pipe's method (sort of)
maxit = 20;

% initial estimate (preserve zeros)
d = reshape(ok,[],1);

% iterative refinement: d = d ./ H*H'*d
for j = 1:maxit
    HHd = obj.spmv(obj.spmv_t(d));
    d = d ./ max(HHd,eps); % protection for div by 0
end

% this "density" is very sensitive to oversampling factor
% e.g. u=1 acts like a highpass filter and u=4 a lowpass.
% this is probably correct behaviour. the density is 
% calculated on the oversampled grid, so for a fixed J
% the density levels off (i.e. spokes stop interacting) 
% as u gets larger. this makes it challenging to interpret
% d as "the" sample density. following Sedarat we interpret
% d as a diagonal approximation to least squares.
%
%      Qx = k [for DFT matrix Q]
%       x = Q'*D*k [gridding D=diag(d)]
% Q'*D*Qx = Q'*D*k [least squares (weighted by D)]
% 
% Sedarat H, Nishimura DG. On the optimality of the gridding
% reconstruction algorithm. IEEE Trans Med Imaging 2000;19:306
%
% in the Cartesian case Q is unitary (Q'*Q=I) so we design
% fNUFT/aNUFT similarly so that Q'Q and Q'DQ are both ~I.
%
%     Q = fft(eye(8))/sqrt(8);
%     diag(Q'*Q) % [1.000 1.000 1.000 ...]
%     D = diag(rand(8,1));
%     diag(Q'*D*Q) % [0.6289 0.6289 0.6289 ...]
%     mean(diag(D)) % 0.6289
%
% so normalize by the mean density to make diag(Q'DQ)=1
d = d / (sum(d)/obj.nnz);

% check true diagonals (T) of Q'*Q and Q'*D*Q (V. SLOW)
if false
    % we can get the k-th diagonal of Q'*D*Q by letting e =
    % [0...0 1 0...0]' (1 at the k-th position) and computing
    % v = Q'*D*Q*e where D=diag(d). the k-th element of v
    % is k-th diagonal element of Q'*D*Q. E.g. if k=2 then
    %
    % v = [q1 q2 q3] [0] = [q2]
    %     [q4 q5 q6] [1]   [q5] <-- the k-th element
    %     [q7 q8 q9] [0]   [q8]
    %
    QQ  = @(x)obj.iprojection(x,0,1);
    QDQ = @(x)obj.iprojection(x,0,d);
    
    N = 10; % how many to test
    T = zeros(N,2); % diag(Q'Q) and diag(Q'DQ)
    for j = 1:N
        e = zeros(obj.N,'like',d);
        e(j) = 1;
        v = QQ(e); % Q'*Q: does not vary with J or u
        T(j,1) = v(j);
        v = QDQ(e); % Q'*D*Q: varies with everything
        T(j,2) = v(j);       
    end
    subplot(1,2,1);cplot(T(:,1));legend({'diag(Q''*Q)','imag'});title('should be 1');
    subplot(1,2,2);cplot(T(:,2));legend({'diag(Q''*D*Q)','imag'});title('should be 1');
    
    % power method (largest eigenvalue of QQ and QDQ)
    tmp1 = randn(obj.N)+i*randn(obj.N);
    tmp2 = randn(obj.N)+i*randn(obj.N);
    for k = 1:30
        tmp1 = QQ (tmp1/norm(tmp1(:)));
        tmp2 = QDQ(tmp2/norm(tmp2(:)));
        fprintf('%.3e %.3e\n',norm(tmp1(:)),norm(tmp2(:)));
    end
    s1 = norm(tmp1(:));
    v1 = tmp1(:)/s1;
    u1 = QQ(v1)/s1;
   
end
