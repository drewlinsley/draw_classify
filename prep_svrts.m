wd = '/Users/drewlinsley/Desktop/results_problem_4';
od = '/Users/drewlinsley/Desktop/res_results_problem_4';
wd = '/Users/drewlinsley/Desktop/res_results_problem_4';
od = '/Users/drewlinsley/Desktop/new_res_results_problem_4';

if ~exist(od,'dir'),
    mkdir(od);
end
ns = [28,28];
files = dir(fullfile(wd,'*.jpg'));
num_files = numel(files);
for idx = 1:num_files,
    if mod(idx,1e4)==0,fprintf('%i/%i\r',idx,num_files);end
    nf = im2bw(imresize(imread(fullfile(wd,files(idx).name)),ns),.9);
    imwrite(nf,fullfile(od,files(idx).name),'Quality',100);
end