
clear
close all
rng(sum(double('BMT2024_Battistel'))); % for reproducibility
init_eidors()
%%

para.n_freqs = 7;
para.freqs_order = 3;
para.pert_amplitude = 1e-6;
para.zern_order = 20;
para.n_elec = 16;
para.noise_ampl = 1e-3;
para.lambda = 5e-1;

para

targets = struct('center', [], 'radius', [], 'cond', []);
targets(1).center = [-.4,0];
targets(1).radius = 0.35;

targets(2).center = [.4,0.2];
targets(2).radius = 0.4;

freqs = logspace(5,6,para.n_freqs);
freqs = round(freqs/1e4)*1e4; % nice values

[muscle, lung_inflated, lung_deflated] = interp_tissues(freqs);

base_cond = muscle;
targets(1).cond = lung_deflated;
targets(2).cond = lung_inflated;

%% Figure 1 of the article 
figure(1)
clf
t = tiledlayout(2,1);
t.TileSpacing = 'compact';
t.Padding = 'compact';

h1 = nexttile;
plot_line_o(freqs, base_cond, targets(1).cond, targets(2).cond)
legend('Muscle', 'Defl. Lung', 'Infl. Lung', 'Location', 'northwest')
ylabel('\sigma / S m^{-1}')

h2 = nexttile;
plot_line_o(freqs, base_cond-base_cond, targets(1).cond-base_cond, targets(2).cond-base_cond)
ylabel('\Delta\sigma / S m^{-1}')
xlabel('F / Hz')

annotation('textbox', h1.Position, 'String', 'a)', 'EdgeColor', 'none', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontSize', 12, 'FontWeight', 'bold');
annotation('textbox', h2.Position, 'String', 'b)', 'EdgeColor', 'none', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontSize', 12, 'FontWeight', 'bold');

%% make models and calculate new jacobian
[imgs_base, imgs_target, img_rec, imgs_reference, imgs_diff, mdl_rec] = mk_models(para.n_elec, base_cond, targets);
% to speed up you can comment next line if you are repeating without changing the model
[J,A] = perturbation_fwd_problem(img_rec, para.pert_amplitude, para.n_freqs, para.freqs_order, para.zern_order);
n_elems = size(A,1)/para.n_freqs;

%% calculate the voltages with and without noise
[vh, vi, vi_noise, delta_volt, delta_volt_noise] = calc_voltages(imgs_base, imgs_target, para.noise_ampl);

%% normal reconstruction
imgs = inv_solve(mdl_rec, vh, vi_noise);
imgs.show_slices.do_colourbar = false;

imgs_tikhonov = cell(para.n_freqs,1);
for ii = 1:para.n_freqs
    imgs_tikhonov{ii} = imgs;
    imgs_tikhonov{ii}.elem_data = imgs.elem_data(:,ii);
end

mm = cellfun(@(c) c.elem_data, imgs_tikhonov, 'UniformOutput', false);
tikhonov_elem_data = reshape(cat(1,mm{:}), n_elems, para.n_freqs);

%% solve with frequency correlation
b = delta_volt_noise(:);

R = eye(size(J,2));
x = (J'*J + para.lambda.^2*R)\(J'*b);
y = J*x;

%% plot reconstructed values vs original
figure(2) 
clf
hold on
plot(b)
plot(y)
hold off

%% Get original data and frequency correlated ones

elem_values_ = A*x;

multifreqs_elem_data = reshape(elem_values_, n_elems, para.n_freqs);

mm = cellfun(@(c) c.elem_data, imgs_reference, 'UniformOutput', false);
orig_elem_data = reshape(cat(1,mm{:}), n_elems, para.n_freqs);

res_elem_data = orig_elem_data - multifreqs_elem_data;


%% Plot reconstructed  images
imgs_multifreqs = imgs_reference;
imgs_res = imgs_reference;

out_imgs_multifreqs = zeros(64*64, para.n_freqs);
out_imgs_tikhonov = zeros(64*64, para.n_freqs);
out_imgs_reference = zeros(64*64, para.n_freqs);

for ii = 1:para.n_freqs
    imgs_multifreqs{ii}.elem_data = multifreqs_elem_data(:,ii);
    imgs_res{ii}.elem_data = res_elem_data(:,ii);
end

pause(2) % otherwise it plots one in the previous figure (a bug?)
figure(3)
clf
t = tiledlayout(4, para.n_freqs);
t.TileSpacing = 'compact';
t.Padding = 'compact';

% reference images
for ii = 1:para.n_freqs
    nexttile
    out_img = show_slices(imgs_reference{ii});
    out_imgs_reference(:,ii) = out_img(:);
end 

% frequency correlated images
for ii = 1:para.n_freqs
    nexttile
    out_img = show_slices(imgs_multifreqs{ii});
    out_imgs_multifreqs(:,ii) = out_img(:);
end 

% standard reconstruction
for ii = 1:para.n_freqs
    nexttile
    out_img = show_slices(imgs_tikhonov{ii});
    out_imgs_tikhonov(:,ii) = out_img(:);
end 

% difference image 
for ii = 1:para.n_freqs
    nexttile
    show_slices(imgs_diff{ii});
end 

%% Figure 2 of the article

% three points to highlight
points_2_see = [0, -0.5; targets(1).center; targets(2).center];

pause(2) % otherwise it plots one in the previous figure (a bug?)
figure(4)
clf
t = tiledlayout(2, 4);
t.TileSpacing = 'compact';
t.Padding = 'compact';

freqs_to_print = {'b) 100 kHz', 'c) 150 kHz', 'd) 220 kHz', 'e) 320 kHz', 'f) 460 kHz', 'g) 680 kHz', 'h) 1 MHz'};

% ground truth
nexttile
hold on
img = flip(reshape(out_imgs_reference(:,1),64,64),1); %eidors flips the images
image(img);
axis image
axis off
title('a) Ground Truth')
% add highlighted points
ax = gca;
ax.TitleHorizontalAlignment = 'left';
for ii = 1:size(points_2_see,1)
    x = points_2_see(ii,1)*32+32;
    y = points_2_see(ii,2)*32+32;
    plot(x,y, 'or')
end
axis equal
hold off

% frequency correlated images
for ii = 1:para.n_freqs
    nexttile
    show_slices(imgs_multifreqs{ii})
    title(freqs_to_print{ii})
    ax = gca;
    ax.TitleHorizontalAlignment = 'left';
    axis equal
end 



%% figure of merit
% L2 on elem_data
L2_tikhonov_elem_data = vecnorm(orig_elem_data - tikhonov_elem_data);
L2_multifreqs_elem_data = vecnorm(orig_elem_data - multifreqs_elem_data);

sum(L2_tikhonov_elem_data)
sum(L2_multifreqs_elem_data)

% L2 on image data (pixel color)
L2_tikhonov_pixels = vecnorm(out_imgs_reference - out_imgs_tikhonov);
L2_multifreqs_pixels = vecnorm(out_imgs_reference - out_imgs_multifreqs);

sum(L2_tikhonov_pixels)
sum(L2_multifreqs_pixels)

%% Figure 3 of the article

l = zeros(size(points_2_see,1),1);
elem_centers = interp_mesh(imgs_reference{1}.fwd_model, 0); % center of elements
for ii = 1:size(points_2_see,1)
    l(ii) = find(sum((elem_centers - points_2_see(ii,:)).^2,2) < 1e-3, 1, 'first');
    elem_centers(l(ii),:);
end

figure(5)
clf
t = tiledlayout(2,1);
t.TileSpacing = 'compact';
t.Padding = 'compact';
h1 = nexttile;
plot_line_o(freqs, multifreqs_elem_data(l(1),:), multifreqs_elem_data(l(2),:), multifreqs_elem_data(l(3),:));
legend('Base', 'Target_1', 'Target_2')
ylabel('\Delta\sigma / S m^{-1}')

h2 = nexttile;
plot_line_o(freqs, tikhonov_elem_data(l(1),:), tikhonov_elem_data(l(2),:), tikhonov_elem_data(l(3),:));
ylabel('\Delta\sigma / S m^{-1}')
xlabel('F / Hz')

annotation('textbox', h1.Position, 'String', 'a)', 'EdgeColor', 'none', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontSize', 12, 'FontWeight', 'bold');
annotation('textbox', h2.Position, 'String', 'b)', 'EdgeColor', 'none', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontSize', 12, 'FontWeight', 'bold');

%% Helper functions declarations

function [imgs_base, imgs_target, img_rec, imgs_reference, imgs_diff, mdl_rec] ...
    = mk_models(n_elec, base_cond, targets)
% make all models

n_freqs = length(base_cond);
model_str_base = 'f2c';
model_str_rec = 'g2c';  % more elements for the reconstruction

% models for measurement data
mdl = mk_common_model(model_str_base, n_elec);

imgs_base = cell(n_freqs,1);
imgs_target = cell(n_freqs,1);

elem_centers = interp_mesh(mdl.fwd_model, 0); % center of elements
idx_target = zeros(length(elem_centers),length(targets));
for it = 1:length(targets)
    idx_target(:,it) = sum((elem_centers-targets(it).center).^2, 2) <= targets(it).radius.^2;
end
idx_target = logical(idx_target);

for ii = 1:n_freqs
    imgs_base{ii} = mk_image(mdl, base_cond(ii));
    imgs_base{ii}.show_slices.do_colourbar = false;

    imgs_target{ii} = imgs_base{ii};
    for it = 1:length(targets)     
        imgs_target{ii}.elem_data(idx_target(:,it)) = targets(it).cond(ii);
    end
end

% models for reconstruction reference and difference
mdl = mk_common_model(model_str_rec, n_elec);

elem_centers = interp_mesh(mdl.fwd_model, 0); % center of elements
idx_target = zeros(length(elem_centers),length(targets));
for it = 1:length(targets)
    idx_target(:,it) = sum((elem_centers-targets(it).center).^2, 2) <= targets(it).radius.^2;
end
idx_target = logical(idx_target);

imgs_reference = cell(n_freqs,1);
imgs_diff = cell(n_freqs,1);

for ii = 1:n_freqs

    imgs_reference{ii} = mk_image(mdl, base_cond(ii));
    imgs_reference{ii}.show_slices.do_colourbar = false;
        
    for it = 1:length(targets)     
        imgs_reference{ii}.elem_data(idx_target(:,it)) = targets(it).cond(ii);
    end

    imgs_diff{ii} = imgs_reference{ii};
    imgs_diff{ii}.elem_data = imgs_diff{ii}.elem_data-base_cond(ii);
end

% model for inverse problem
mdl_rec = mdl;
img_rec = mk_image(mdl_rec);
img_rec.show_slices.do_colourbar = false;
end


function [vh, vi, vi_noise, delta_volt, delta_volt_noise] = calc_voltages(imgs_base, imgs_target, noise_ampl)
% solve the models and calculate the voltages differences

n_freqs = length(imgs_base);
vh_cell = cell(n_freqs,1);
vi_cell = cell(n_freqs,1);

for ii = 1:n_freqs
    vh_cell{ii} = fwd_solve(imgs_base{ii});
    vi_cell{ii} = fwd_solve(imgs_target{ii});
end

mm = @(v) reshape(cell2mat(cellfun(@(c) c.meas, v, 'UniformOutput', false)), [], n_freqs);
vh = mm(vh_cell);
vi = mm(vi_cell);

noise = noise_ampl*std(vi,0,'all')*randn(size(vi));
vi_noise = vi + noise;

% normalize by the norm of vh
delta_volt = (vi-vh)./vecnorm(vh,2,1);
delta_volt_noise = (vi_noise-vh)./vecnorm(vh,2,1);
end


function plot_line_o(x, y1, y2, y3)
% easy plot conductivities
y1 = y1(:);
y2 = y2(:);
y3 = y3(:);

hold on
plot(x, y1, '-o')
plot(x, y2, '-o')
plot(x, y3, '-o')
hold off
set(gca, 'xscale', 'log')
end
