%{
mbh.m

This file is used to calculate mbh(motion boundary histogram)
between 2 frames. If the input video has n frames, then n-1
files will be saved in the output directory.

By: Yijie Xu
Date: 2019.3.15
%}
function image = saveMBH(input_path, output_dir, current_time)
    % The maximum number of the video frames.
    % This parameter is setted for matric memory allocation.
    FRAME_MAX = 500;

    v = VideoReader(input_path);
    v.CurrentTime = current_time;

    n = FRAME_MAX;
    step = 1;

    floor_thredhold = 30;

    % x_mbhs = zeros(v.Height, v.Width, uint32(n / step)); % To store mbhs(x direction)
    % y_mbhs = zeros(v.Height, v.Width, uint32(n / step));

    opticFlow = opticalFlowHS;
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
            frameGray = rgb2gray(frameRGB);
            estimateFlow(opticFlow, frameGray);
            continue
        end

        if mod(i, step) == 0

            frameGray = rgb2gray(frameRGB);
            flow = estimateFlow(opticFlow, frameGray);

            x_op = normalized(flow.Vx, 0);
            y_op = normalized(flow.Vy, 0);

            x_kernel = [1 0 -1];
            y_kernel = x_kernel';

            x_result1 = conv2(x_op, x_kernel,'same');
            x_result2 = conv2(x_op, y_kernel,'same');
            x_result = normalized((x_result1(2:end-1, 2:end-1).^2 +...
                                   x_result2(2:end-1, 2:end-1).^2), floor_thredhold);

            y_result1 = conv2(y_op, x_kernel,'same');
            y_result2 = conv2(y_op, y_kernel,'same');
            y_result = normalized((y_result1(2:end-1, 2:end-1).^2 +...
                                   y_result2(2:end-1, 2:end-1).^2), floor_thredhold);


            x_result = uint8(double(frameGray(2:end-1, 2:end-1)).*double((x_result>0)));
            y_result = uint8(double(frameGray(2:end-1, 2:end-1)).*double((y_result>0)));

            count = count + 1;
            padded_x_mbhs = zeros(v.Height, v.Width, 'uint8');
            padded_y_mbhs = zeros(v.Height, v.Width, 'uint8');
            padded_x_mbhs(2:end-1, 2:end-1) = x_result;
            padded_y_mbhs(2:end-1, 2:end-1) = y_result;

            % Save 2 images into the directory
            [~, video_name, ~] = fileparts(input_path);
            x_mbh_file_name = [video_name '_mbh_' int2str(count) '_01.jpg'];
            y_mbh_file_name = [video_name '_mbh_' int2str(count) '_02.jpg'];
            save_path = fullfile(output_dir, x_mbh_file_name);
            imwrite(padded_x_mbhs, save_path);
            save_path = fullfile(output_dir, y_mbh_file_name);
            imwrite(padded_y_mbhs, save_path);

        end
    end

    image = zeros(1,1);
    clear v opticFlow;
end

% Normalize the picture grayscale color.
% 2-d array input
function image = normalized(input, floor)
    input = abs(double(input));
    minV = min(min(input));
    image = (-minV + input) * 255/(max(max(input)) - minV);
    image = uint8(image.*(image > floor));
end
