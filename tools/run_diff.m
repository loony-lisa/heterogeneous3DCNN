%{
run_diff.m

Generate difference image between two adjacant frames
%}
function result = run_diff(input_root, output_root)

  % Get class names
  all_dirs = dir(input_root);
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
  
  % Create new folders named by class name for saving images
  for i = 1:class_num
    if exist(fullfile(output_root, classNames{i}), 'dir') ~= 7
      mkdir(fullfile(output_root, classNames{i}));
    end
  end
  
  for i = 1:class_num
    % Load videos from the folder
    class_name = classNames{i};
    input_dir = fullfile(input_root, class_name);
    dirData = dir(fullfile(input_dir, '*.avi'));      
    fileList = {dirData.name}';
     
    for j = 1:length(fileList)
      file_name = fileList{j}; % video file name
      video_path = fullfile(input_dir, file_name);
      save_dir = fullfile(output_root, class_name, file_name);
      
      if exist(save_dir, 'dir') ~= 7
        mkdir(save_dir);
      end
        
      save_features(video_path, save_dir); 
    end
  end
  result = 0;
end 



   

