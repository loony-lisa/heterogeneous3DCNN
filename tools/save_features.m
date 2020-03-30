%{
Preprocessing the video data.

Transform video data into mbh, diff and grayscale images.
%}

function result=save_features(input_path)
  VIDEO_ROOT = 'D:\2019_graduation_work\data';
  DIFF_ROOT = 'D:\2019_graduation_work\diff';
  GRAY_ROOT = 'D:\2019_graduation_work\gray';
  MBH_ROOT = 'D:\2019_graduation_work\mbh';
  FRAME_MAX = 600; 
  
  v = VideoReader(input_path);
  step = 1;
  count = 1;
  
  for i=1:MAX_FRAME
    % To check frame exists or not 
    if ~hasFrame(v)
      break
    end
    
    frameRGB = readFrame(v);
    
    % Always descade the first image 
    if i == 1 || mod(i, step) ~= 0
      continue
    end
    
    count = count + 1;
    frameGray = rgb2gray(frameRGB);
    
    [~, video_name, ~] = fileparts(input_path);
    gray_file_name = [video_name '_gray_' int2str(count) '.jpg'];
    save_path = fullfile(output_dir, gray_file_name);
    % Save grayscale image
    imwrite(frameGray, save_path);
    
    x_kernel = [1 0 -1];
    y_kernel = x_kernel';
    
    x_grad = conv2(frameGray, x_kernel,'same');
    y_grad = conv2(frameGray, y_kernel,'same');
                
    padded_x_mbhs = zeros(v.Height, v.Width, 'uint8');
    padded_y_mbhs = zeros(v.Height, v.Width, 'uint8');
    padded_x_mbhs(2:end-1, 2:end-1) = x_grad(2:end-1, 2:end-1);
    padded_y_mbhs(2:end-1, 2:end-1) = y_grad(2:end-1, 2:end-1);
    
    % Save 2 images into the directory
    [~, video_name, ~] = fileparts(input_path);
    x_mbh_file_name = [video_name '_diff_' int2str(count) '_x.jpg'];
    y_mbh_file_name = [video_name '_diff_' int2str(count) '_y.jpg'];
    save_path = fullfile(output_dir, x_mbh_file_name);
    imwrite(padded_x_mbhs, save_path);
    save_path = fullfile(output_dir, y_mbh_file_name);
    imwrite(padded_y_mbhs, save_path);
    end
end 

 
% Normalize the picture grayscale color.
% 2-d array input
function image = normalized(input, floor)
    input = abs(double(input));
    minV = min(min(input));
    image = (-minV + input) * 255/(max(max(input)) - minV);
    image = uint8(image.*(image > floor));
end




