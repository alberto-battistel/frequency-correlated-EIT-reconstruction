function [x,n_singular_values] = inv_solve_truncated_SVD(J, b, n_singular_values)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

[U,S,V] = svd(J, 'econ');


picard_coeffs = U'*b;
diag_S = diag(S);


if n_singular_values == 0 || n_singular_values == inf
    n_singular_values = find(abs(picard_coeffs) > diag_S, 1, 'first');
    
    figure(215883)
    clf
    hold on
    plot([abs(picard_coeffs), diag_S])
    plot(n_singular_values, abs(picard_coeffs(n_singular_values)), 'o')
    hold off
    set(gca, 'yscale', 'log')
end
truncSVD = @(k) V(:,1:k)*diag(1./diag(S(1:k,1:k)))*U(:,1:k)';
x = truncSVD(n_singular_values)*b;

end