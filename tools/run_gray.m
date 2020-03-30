%{
run_gray.m

%}
clear variables;

INPUT_ROOT = '/mnt/UCF50_videos/UCF50/';
OUTPUT_ROOT = '/mnt/ehpc-hz-JdjdHW8IZe/UCF50_gray/';

% Get class names
all_dirs = dir(INPUT_ROOT);
dir_names = {all_dirs.name};

% Throw out '.' and '..'
classNames = {};
for name = dir_names
    if name ~= "." && name ~= ".."
        classNames = cat(1, classNames, name);
    end
end

class_num = length(classNames);
disp(classNames);
fprintf("%d classes.\n", class_num);

% Create new directory
for i = 1:class_num
    if exist(fullfile(OUTPUT_ROOT, classNames{i}), 'dir') ~= 7
        mkdir(fullfile(OUTPUT_ROOT, classNames{i}));
    end
end

for i = 1:10
    %class_num
    % Load videos from the folder
    class_name = classNames{i};
    input_dir = fullfile(INPUT_ROOT, class_name);
    dirData = dir(fullfile(input_dir, '*.avi'));
    fileList = {dirData.name}';

    for j = 1:length(fileList)
        file_name = fileList{j}; % video file name
        video_path = fullfile(input_dir, file_name);
        save_dir = fullfile(OUTPUT_ROOT, class_name, file_name);

        if exist(save_dir, 'dir') ~= 7
            mkdir(save_dir);
        end

        im = saveDiff(video_path, save_dir, 0);
        fprintf("%s 's diff is saved.\n", video_path);
    end
end
