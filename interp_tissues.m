function [muscle,lung_inflated,lung_deflated] = interp_tissues(freqs)
% interpolate the tissues data

freqs = freqs(:);
tissues = load('tissues.mat');

interpolate_fun = @(tissue) makima(tissue(:,1), tissue(:,3), freqs);

muscle = interpolate_fun(tissues.muscle);
lung_inflated = interpolate_fun(tissues.lung_inflated);
lung_deflated = interpolate_fun(tissues.lung_deflated);

end