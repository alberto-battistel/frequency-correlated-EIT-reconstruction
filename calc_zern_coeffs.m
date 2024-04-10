function [coeffs_matrix] = calc_zern_coeffs(max_radial_order)
% dirty way to make a matrix with all the zernike coefficients up to the
% max_radial_order
coeffs_matrix = [];
for n = 0:max_radial_order
    mk = -n:2:n;
    buf = n.*ones(length(mk),2);
    buf(:,2) = mk;
    coeffs_matrix = [coeffs_matrix; buf];
end