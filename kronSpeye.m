function [K] = kronSpeye(A,n)

C = cell(n,1);
[C{:}] = deal(sparse(A));
K = blkdiag(C{:});
% Rearrange rows and columns
I = reshape(reshape(1:(size(A,1)*n),size(A,1),n)',size(A,1)*n,1);
J = reshape(reshape(1:(size(A,2)*n),size(A,2),n)',size(A,2)*n,1);
K = K(I,J);