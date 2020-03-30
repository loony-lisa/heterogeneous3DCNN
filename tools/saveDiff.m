%{
save 
%}
function image = saveDiff(input_path, output_dir)
    % The maximum number of the video frames.
    % This parameter is setted for matric memory allocation.
    FRAME_MAX = 600; 
    
    v = VideoReader(input_path);
    n = FRAME_MAX;
    step = 1;
    
    count = 1;
    
    for i=1:n
        % To check frame exists or not 
        if ~hasFrame(v)
            break
        end
        
        frameRGB = readFrame(v);
        % The function estimateFlow sets the previous image for 
        % the first run to a black image. So the first optical
        % flow image need to be discarded.
        if i == 1
            rgb2gray(frameRGB);
            continue
        end
        
        if mod(i, step) == 0
            frameGray = rgb2gray(frameRGB);
            
            x_kernel = [1 0 -1];
            y_kernel = x_kernel';
            
            x_grad = conv2(frameGray, x_kernel,'same');
            y_grad = conv2(frameGray, y_kernel,'same');
           
            count = count + 1;
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
    
    image = zeros(1,1);
    clear v;
end

% Normalize the picture grayscale color.
% 2-d array input
function image = normalized(input, floor)
    input = abs(double(input));
    minV = min(min(input));
    image = (-minV + input) * 255/(max(max(input)) - minV);
    image = uint8(image.*(image > floor));
end




