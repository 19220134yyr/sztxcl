function createGUI()
    % 创建图形窗口
    hFig = figure('Name', '大作业', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none', 'Position', [100, 100, 600, 500]);
    
    % 创建坐标轴用于显示图像
    ax = axes('Parent', hFig, 'Units', 'pixels', 'Position', [50, 250, 500, 200]);
    title(ax, '图像显示区域');
    
    % 添加按钮和菜单等控件
    uicontrol('Style', 'pushbutton', 'String', '打开图像', 'Position', [50, 210, 100, 30], 'Callback', @openImage);
    uicontrol('Style', 'pushbutton', 'String', '显示直方图', 'Position', [160, 210, 100, 30], 'Callback', @showHistogram);
    uicontrol('Style', 'pushbutton', 'String', '对比度增强', 'Position', [270, 210, 100, 30], 'Callback', @enhanceContrast);
    uicontrol('Style', 'pushbutton', 'String', '图像变换', 'Position', [380, 210, 100, 30], 'Callback', @imageTransform);
    uicontrol('Style', 'pushbutton', 'String', '加噪和滤波', 'Position', [490, 210, 100, 30], 'Callback', @noiseAndFilter);
    uicontrol('Style', 'pushbutton', 'String', '边缘提取', 'Position', [50, 170, 100, 30], 'Callback', @edgeDetection);
    uicontrol('Style', 'pushbutton', 'String', '目标提取', 'Position', [160, 170, 100, 30], 'Callback', @objectExtraction);
    uicontrol('Style', 'pushbutton', 'String', '特征提取', 'Position', [270, 170, 100, 30], 'Callback', @featureExtraction);
    
    % 初始化数据存储
    setappdata(hFig, 'imageData', []);
end


function openImage(hObject, eventdata)
    
    % 选择图像文件
    [filename, pathname] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif', 'Image Files (*.jpg, *.jpeg, *.png, *.bmp, *.tif)'}, '选择图像文件');
    
    % 如果用户选择了文件
    if isequal(filename, 0)
        disp('用户取消了打开图像文件。');
        return; % 用户取消了操作，直接返回
    else
        % 构建完整的文件路径
        fullpath = fullfile(pathname, filename);
        
        % 读取图像
        imageData = imread(fullpath);
        
        % 获取当前图形窗口句柄
        hFig = gcf;
        
        % 获取坐标轴句柄
        ax = findobj(hFig, 'Type', 'axes');
        
        % 显示图像
        imshow(imageData, 'Parent', ax);
        
        % 保存图像数据到GUI句柄
        setappdata(hFig, 'imageData', imageData);
        
        % 更新坐标轴标题
        title(ax, ['显示图像: ', filename]);
    end
end


function showHistogram(hObject, eventdata)
    
    % 获取当前图形窗口句柄
    hFig = gcf;
    
    % 从GUI句柄中获取图像数据
    imageData = getappdata(hFig, 'imageData');
    
    % 如果没有图像数据，则返回
    if isempty(imageData)
        disp('没有图像数据。请先打开一个图像文件。');
        return;
    end
    
    % 如果图像是彩色的，则转换为灰度图像
    if size(imageData, 3) == 3
        grayImage = rgb2gray(imageData);
    else
        grayImage = imageData;
    end
    
    % 计算灰度直方图
    [counts, binLocations] = imhist(grayImage);
    
    % 显示原始灰度直方图
    figure('Name', '原始灰度直方图', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
    bar(binLocations, counts);
    xlabel('灰度值');
    ylabel('像素数量');
    title('原始图像灰度直方图');
    axis tight;
    
    % 计算累积分布函数（CDF）
    cdf = cumsum(counts) / sum(counts);
    
    % 使用累积分布函数进行直方图均衡化
    equalizedImage = im2uint8(mat2gray(cdf(grayImage + 1)));
    
    % 显示均衡化后的直方图
    figure('Name', '均衡化后的灰度直方图', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
    [equalizedCounts, ~] = imhist(equalizedImage);
    bar(binLocations, equalizedCounts);
    xlabel('灰度值');
    ylabel('像素数量');
    title('均衡化后图像灰度直方图');
    axis tight;
end


function imageTransform(hObject, eventdata)
    % 获取当前图形窗口句柄
    hFig = gcf;
    
    % 从GUI句柄中获取图像数据
    imageData = getappdata(hFig, 'imageData');
    
    % 如果没有图像数据，则返回
    if isempty(imageData)
        disp('没有图像数据。请先打开一个图像文件。');
        return;
    end
    
    % 如果图像是彩色的，则转换为灰度图像
    if size(imageData, 3) == 3
        grayImage = rgb2gray(imageData);
    else
        grayImage = imageData;
    end
    
    % 创建输入对话框
    prompt = {'请输入水平缩小系数：', '请输入垂直缩小系数：', '请输入旋转角度（度数）：'};
    dlgtitle = '输入图像变换参数';
    numlines = 1;
    definput = {'0.66', '0.77', '45'}; % 默认输入值
    answer = inputdlg(prompt, dlgtitle, numlines, definput);
    
    % 检查用户是否点击了取消或关闭对话框
    if isempty(answer)
        disp('用户取消了输入。');
        return;
    end
    
    % 获取输入值
    kx = str2double(answer{1});
    ky = str2double(answer{2});
    theta = str2double(answer{3});
    
    % 缩小图像
    g_scaled = bilinearInterpolationScale(grayImage, kx, ky);
    
    % 将角度转换为弧度
    theta = -theta * pi / 180;
    
    % 旋转图像
    g_rotated = bilinearInterpolationRotate(g_scaled, theta);
    
    % 保存变换后的图像数据到GUI句柄
    setappdata(hFig, 'scaledImageData', g_scaled);
    setappdata(hFig, 'rotatedImageData', g_rotated);
    
    % 显示变换后的图像
    figure('Name', '缩小后的图像', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
    imshow(g_scaled);
    title('缩小后的图像');
    
    figure('Name', '缩小和旋转后的图像', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
    imshow(g_rotated);
    title('缩小和旋转后的图像');
end

function g_scaled = bilinearInterpolationScale(f, kx, ky)
    % 双线性插值缩小图像
    [height, width] = size(f);
    newHeight = floor(ky * height); % 使用floor确保新尺寸是原始尺寸的整数倍
    newWidth = floor(kx * width);
    g_scaled = uint8(zeros(newHeight, newWidth)); % 输出图像初始化为uint8
    
    % 将输入图像转换为double类型
    f_double = double(f);
    
    for i = 1:newHeight
        for j = 1:newWidth
            x = (j - 1) / kx + 1;
            y = (i - 1) / ky + 1;
            x1 = floor(x);
            x2 = ceil(x);
            y1 = floor(y);
            y2 = ceil(y);
            
            % 确保x1, x2, y1, y2在图像尺寸范围内
            x1 = max(1, min(x1, width));
            x2 = max(1, min(x2, width));
            y1 = max(1, min(y1, height));
            y2 = max(1, min(y2, height));
            
            % 计算权重
            r1 = f_double(y1, x1) * (x2 - x) + f_double(y1, x2) * (x - x1);
            r2 = f_double(y2, x1) * (x2 - x) + f_double(y2, x2) * (x - x1);
            
            % 应用权重
            g_scaled(i, j) = uint8(r1 * (y2 - y) + r2 * (y - y1));
        end
    end
end



function g_rotated = bilinearInterpolationRotate(f, theta)
    % 双线性插值旋转图像
    [height, width] = size(f);
    cos_theta = cos(theta);
    sin_theta = sin(theta);
    t = [cos_theta, -sin_theta; sin_theta, cos_theta];
    
    % 计算旋转后的四个角点
    corners = [-1 -1; width -1; width height; -1 height]';
    rotated_corners = t * corners;
    
    % 找出旋转后图像的边界
    min_x = min(rotated_corners(1, :));
    max_x = max(rotated_corners(1, :));
    min_y = min(rotated_corners(2, :));
    max_y = max(rotated_corners(2, :));
    
    % 计算新图像的大小
    newWidth = round(max_x - min_x) + 1;
    newHeight = round(max_y - min_y) + 1;
    g_rotated = uint8(zeros(newHeight, newWidth)); % 初始化为uint8类型
    
    % 旋转矩阵的逆
    t_inv = [cos_theta, sin_theta; -sin_theta, cos_theta];
    
    % 将输入图像转换为double类型
    f_double = double(f);
    
    for i = 1:newHeight
        for j = 1:newWidth
            % 将旋转后的坐标映射回原始图像
            x = t_inv * [(j - 1 + min_x); (i - 1 + min_y)];
            
            % 找到原始图像中最近的四个点
            x1 = floor(x(1));
            x2 = ceil(x(1));
            y1 = floor(x(2));
            y2 = ceil(x(2));

            % 确保坐标不超出原始图像的边界
            x1 = max(1, min(x1, width));
            x2 = max(1, min(x2, width));
            y1 = max(1, min(y1, height));
            y2 = max(1, min(y2, height));
            
            % 进行双线性插值
            if x1 == x2 && y1 == y2
                g_rotated(i, j) = f_double(y1, x1);
            else
                r1 = f_double(y1, x1) * (x2 - x(1)) + f_double(y1, x2) * (x(1) - x1);
                r2 = f_double(y2, x1) * (x2 - x(1)) + f_double(y2, x2) * (x(1) - x1);
                g_rotated(i, j) = uint8(r1 * (y2 - x(2)) + r2 * (x(2) - y1));
            end
        end
    end
end



function enhanceContrast(hObject, eventdata)
    % 获取当前图形窗口句柄
    hFig = gcf;
    
    % 从GUI句柄中获取图像数据
    imageData = getappdata(hFig, 'imageData');
    
    % 如果没有图像数据，则返回
    if isempty(imageData)
        disp('没有图像数据。请先打开一个图像文件。');
        return;
    end
    
    % 如果图像是彩色的，则转换为灰度图像
    if size(imageData, 3) == 3
        grayImg = rgb2gray(imageData);
    else
        grayImg = imageData;
    end
    
    % 显示原始灰度图像
    figure('Name', '原始灰度图像', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
    imshow(grayImg);
    title('原始灰度图像');
    
    % 线性变换增强对比度
    c = 1.5; % 对比度增强系数
    b = 0; % 偏移量
    linearEnhanced = uint8(c * double(grayImg) + b);
    figure('Name', '线性变换增强对比度', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
    imshow(linearEnhanced);
    title('线性变换增强对比度');
    
    % 保存线性变换增强的图像
    setappdata(hFig, 'linearEnhancedImage', linearEnhanced);
        
    % 对数变换增强对比度
    base = 0.01; % 基数
    logEnhanced = uint8(255 * log(1 + base * double(grayImg)));
    logEnhanced = max(min(logEnhanced, 255), 0); % 裁剪到[0, 255]范围
    
    figure('Name', '对数变换增强对比度', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
    imshow(logEnhanced);
    title('对数变换增强对比度');
    
    % 保存对数变换增强的图像
    setappdata(hFig, 'logEnhancedImage', logEnhanced);

    
    % 指数变换增强对比度
    gamma = 0.4; % 指数变换的gamma值
    expEnhanced = uint8(255 * (double(grayImg) / 255) .^ gamma);
    figure('Name', '指数变换增强对比度', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
    imshow(expEnhanced);
    title(['指数变换增强对比度 (gamma=' num2str(gamma) ')']);
    
    % 保存指数变换增强的图像
    setappdata(hFig, 'expEnhancedImage', expEnhanced);
end




function noiseAndFilter(hObject, eventdata)
    % 获取当前图形窗口句柄
    hFig = gcf;
    
    % 从GUI句柄中获取图像数据
    imageData = getappdata(hFig, 'imageData');
    
    % 如果是彩色图像，则转换为灰度图像
    if size(imageData, 3) == 3
        imageData = rgb2gray(imageData);
    end
    
    % 创建对话框，提示用户输入噪声水平
    prompt = {'输入噪声水平:'};
    dlgtitle = 'Input Noise Level';
    numlines = 1;
    definput = {'0.2'}; % 默认噪声水平
    answer = inputdlg(prompt, dlgtitle, numlines, definput);
    
    % 检查用户是否输入了值
    if isempty(answer)
        disp('没有输入');
        return;
    else
        noiseLevel = str2double(answer{1});
        if isnan(noiseLevel)
            disp('无效输入');
            return;
        end
    end
    
     % 添加高斯噪声
    noisyImage = imnoise(imageData, 'gaussian', 0, noiseLevel^2);
    
    % 显示噪声图像
    figure('Name', '添加高斯噪声', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
    imshow(noisyImage);
    title('添加高斯噪声后的图像');
    
    % 均值滤波 
    filterSize = 3; % 滤波器大小
    meanFilter = ones(filterSize) / (filterSize^2);
    filteredImageMean = filter2(meanFilter, noisyImage);
    filteredImageMean = uint8(filteredImageMean);
    
    % 显示均值滤波后的图像
    figure('Name', '空域-均值滤波', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
    imshow(filteredImageMean);
    title('均值滤波后的图像');
    
    % 理想低通滤波
    [M, N] = size(noisyImage);
    D0 = max(M, N) ; % 截止频率
    [u, v] = meshgrid(-N/2:N/2-1, -M/2:M/2-1);
    D = sqrt(u.^2 + v.^2);
    H_ideal = double(D <= D0);
    
    % 频域滤波 - 理想低通
    F = fft2(double(noisyImage));
    G_ideal = F .* H_ideal;
    filteredImageIdeal = ifft2(G_ideal);
    filteredImageIdeal = real(filteredImageIdeal);
    
    % 归一化到[0, 255]并转换为uint8
    filteredImageIdeal = filteredImageIdeal - min(filteredImageIdeal(:));
    filteredImageIdeal = filteredImageIdeal / max(filteredImageIdeal(:)) * 255;
    filteredImageIdeal = uint8(filteredImageIdeal);
    
    % 显示理想低通滤波后的图像
    figure('Name', '频域-理想低通滤波', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
    imshow(filteredImageIdeal);
    title('理想低通滤波后的图像');
end





function edgeDetection(hObject, eventdata)
    % 获取当前图形窗口句柄
    hFig = gcf;
    
    % 从GUI句柄中获取图像数据
    imageData = getappdata(hFig, 'imageData');
    
    % 如果没有图像数据，则返回
    if isempty(imageData)
        disp('没有图像数据。请先打开一个图像文件。');
        return;
    end
    
    % 如果图像是彩色的，则转换为灰度图像
    if size(imageData, 3) == 3
        grayImage = rgb2gray(imageData);
    else
        grayImage = imageData;
    end
    
    % 创建窗口
    figure('Name', '边缘检测', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
    
    % Robert算子
    subplot(2, 2, 1); 
    robertsEdge = edge(grayImage, 'roberts');
    imshow(robertsEdge);
    title('Robert算子边缘提取');
    
    % Prewitt算子
    subplot(2, 2, 2); 
    prewittEdge = edge(grayImage, 'prewitt');
    imshow(prewittEdge);
    title('Prewitt算子边缘提取');
    
    % Sobel算子
    subplot(2, 2, 3); 
    sobelEdge = edge(grayImage, 'sobel');
    imshow(sobelEdge);
    title('Sobel算子边缘提取');
    
    % 拉普拉斯算子
    subplot(2, 2, 4); 
    laplacianEdge = edge(grayImage, 'log'); % 'log' 使用了拉普拉斯高斯算子
    imshow(laplacianEdge);
    title('拉普拉斯算子边缘提取');
end



function objectExtraction(hObject, eventdata)
    % 获取当前图形窗口句柄
    hFig = gcf;
    
    % 从GUI句柄中获取图像数据
    imageData = getappdata(hFig, 'imageData');
    
    % 如果没有图像数据，则返回
    if isempty(imageData)
        disp('没有图像数据。请先打开一个图像文件。');
        return;
    end
    
    % 如果图像是彩色的，则转换为灰度图像
    if size(imageData, 3) == 3
        grayImage = rgb2gray(imageData);
    else
        grayImage = imageData;
    end
    
    % 使用Otsu方法进行阈值分割
    level = graythresh(grayImage);
    binaryImage = imbinarize(grayImage, level);
    
    % 标记连通区域
    [L, numObjects] = bwlabel(binaryImage);
    
    % 显示二值图像和标记的连通区域
    figure;
    imshow(binaryImage);
    title('处理后的二值图像');
        
    figure;
    imshow(L);
    title(['找到了', num2str(numObjects), '个标记的物体']);

    
end


function featureExtraction(hObject, eventdata)
    % 特征提取的回调函数
    hFig = gcf;
    
    % 从GUI句柄中获取图像数据
    imageData = getappdata(hFig, 'imageData');
    
    % 如果没有图像数据，则返回
    if isempty(imageData) 
        disp('没有图像数据。请先打开一个图像文件。');
       return;
    end
    
    % 检查图像是否为灰度图像，如果不是则转换为灰度图像
    if size(imageData, 3) == 3
        grayImageData = rgb2gray(imageData);
    else
        grayImageData = imageData;
    end

    calculateLBP(grayImageData);
    calculateHOG(grayImageData);

    %LBP特征提取
    function lbpImage = calculateLBP(grayImage)
        [N,M]=size(grayImage);
        lbp=zeros(N,M);
        for j=2:N-1
            for i=2:M-1
                neighbor=[j-1 i-1;j-1 i;j-1 i+1;j i+1;j+1 i+1;j+1 i;j+1 i-1;j i-1];
                count=0;
                for k=1:8
                    if grayImageData(neighbor(k,1),neighbor(k,2))>grayImage(j,i)
                        count=count+2^(8-k);
                    end
                end
                lbp(j,i)=count;
            end
        end
        lbp=uint8(lbp);
        figure('Name', 'LBP特征图', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
        imshow(lbp);
        title('LBP特征图');
        subim=lbp(1:8,1:8);
        figure('Name', 'LBP直方图', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
        imhist(subim);
        title('第一个8*8子区域的LBP直方图');
        end

    %HOG特征提取
    function hogFeatures = calculateHOG(grayImage)
    % 计算图像的梯度
    [Gx, Gy] = imgradientxy(grayImage);
    % 计算梯度的幅度和方向
    GradientMag = sqrt(Gx.^2 + Gy.^2);
    GradientAngle = atan2(Gy, Gx) * (180/pi); % 将弧度转换为度

    % 将角度限制在0到180度之间
    GradientAngle(GradientAngle < 0) = GradientAngle(GradientAngle < 0) + 180;
    
    % 设置HOG参数
    cellSize = [8 8]; % 单元格大小
    blockSize = [16 16]; % 块大小（包含多个单元格）
    bins = 9; % 直方图的方向数量

    % 初始化HOG特征
    hogFeatures = zeros(floor(size(grayImage, 1)/cellSize(1)), floor(size(grayImage, 2)/cellSize(2)), bins);

    % 计算每个单元格的直方图
    for y = 1:cellSize(1):size(grayImage, 1)-cellSize(1)+1
        for x = 1:cellSize(2):size(grayImage, 2)-cellSize(2)+1
            cellMag = GradientMag(y:y+cellSize(1)-1, x:x+cellSize(2)-1);
            cellAngle = GradientAngle(y:y+cellSize(1)-1, x:x+cellSize(2)-1);
            cellHOG = zeros(1, bins);
            for i = 1:cellSize(1)
                for j = 1:cellSize(2)
                    angleBin = floor(cellAngle(i, j) / 20) + 1; % 将角度映射到直方图的bin
                    if angleBin == bins+1
                        angleBin = 1; % 如果角度是180度，则将其映射到第一个bin
                    end
                    cellHOG(angleBin) = cellHOG(angleBin) + cellMag(i, j);
                end
            end
            hogFeatures(floor((y-1)/cellSize(1)+1), floor((x-1)/cellSize(2)+1), :) = cellHOG;
        end
    end
    
    % 可视化HOG特征
    figure('Name', 'HOG特征', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
    for i = 1:size(hogFeatures, 3)
        subplot(3, 3, i);
        imshow(hogFeatures(:, :, i), []);
        title(['HOG特征', num2str(i)]);
    end
    end

end