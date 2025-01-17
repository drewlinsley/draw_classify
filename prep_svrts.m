stem_dir = '/Users/drewlinsley/Documents/ubuntu_shared';
dir_list = dir(fullfile(stem_dir,'results*'));
old_ft = '.png';
new_ft = '.jpg';
ns = [64,64];

for ol = 1:numel(dir_list),
    fprintf('Working directory %i/%i\r',ol,numel(dir_list))
    %name old dir
    od =  fullfile(stem_dir,dir_list(ol).name);
    %make new dir
    nd = fullfile(stem_dir,strcat('res_',dir_list(ol).name));
    if ~exist(nd,'dir'),
        mkdir(nd);
    end
    %list files in new dir
    flist = dir(fullfile(od,strcat('*',old_ft)));
    for il = 1:numel(flist),
        nf = im2bw(imresize(imread(fullfile(od,flist(il).name)),ns),.9);
        new_name = strrep(flist(il).name,old_ft,new_ft);
        imwrite(nf,fullfile(nd,new_name),'Quality',100);
    end
end

% return
% wd = '/Users/drewlinsley/Desktop/results_problem_4';
% od = '/Users/drewlinsley/Desktop/res_results_problem_4';
% wd = '/Users/drewlinsley/Desktop/res_results_problem_4';
% od = '/Users/drewlinsley/Desktop/new_res_results_problem_4';
% 
% if ~exist(od,'dir'),
%     mkdir(od);
% end
% ns = [28,28];
% files = dir(fullfile(wd,'*.jpg'));
% num_files = numel(files);
% for idx = 1:num_files,
%     if mod(idx,1e4)==0,fprintf('%i/%i\r',idx,num_files);end
% end