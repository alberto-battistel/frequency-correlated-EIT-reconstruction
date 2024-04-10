function [J,A] = perturbation_fwd_problem(mdl, pert_amplitude, n_freqs, freqs_order, zern_order)
% given a model mdl calculate the discrete jacobian J and associated 
% reconstruction matrix Afor a pert_amplitude
% with n_freqs and a tensor product of freqs_order and zern_order 

un_perturbed_value = 1;

elem_centers = interp_mesh(mdl.fwd_model, 0); % center of elements
cyl_elem_centers = cylindrical_elem_centers(elem_centers);

[coefficients_matrix] = calc_zern_coeffs(zern_order);
n_space_pert = size(coefficients_matrix,1);
space_basis_set = zernfun(coefficients_matrix(:,1), coefficients_matrix(:,2), cyl_elem_centers(:,1), cyl_elem_centers(:,2), 'norm');

freqs = linspace(0,1,n_freqs);
freqs_basis = zeros(n_freqs, freqs_order); 
for ii = 1:freqs_order+1 
    freqs_basis(:,ii) = freqs.^(ii-1);
end

a = cell(length(freqs), freqs_order);
for ii = 1:n_freqs
    for order = 1:freqs_order+1
        a{ii,order} = space_basis_set.*freqs_basis(ii, order);
    end
end
A = cell2mat(a);
n_combs = size(A,2);

%%

homo_img = mk_image(mdl, un_perturbed_value);
% homo_img.show_slices.do_colourbar = true;
homo_elem_data = homo_img.elem_data;

vh = fwd_solve(homo_img);
v0 = vh.meas;
n_voltages = length(vh.meas);

pert_img = cell(n_space_pert ,1);
v_pert = zeros(n_freqs*n_voltages, n_combs);

for i_combs = 1:n_combs
    aa_ = reshape(A(:,i_combs), [], n_freqs);
    v_pert_comb = zeros(n_voltages,n_freqs);
    for i_freqs = 1:n_freqs
        pert_img{i_freqs, i_combs} = homo_img;
        pert_img{i_freqs, i_combs}.elem_data = homo_elem_data + pert_amplitude*aa_(:,i_freqs);
        v_pert_ = fwd_solve(pert_img{i_freqs, i_combs});
        v_pert_comb(:, i_freqs) = v_pert_.meas;
    end
    v_pert(:, i_combs) = v_pert_comb(:);
end
        

%%

v_diff = v_pert-repmat(v0, n_freqs, n_combs);

J = v_diff/pert_amplitude;

