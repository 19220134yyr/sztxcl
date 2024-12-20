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
    logEnhanced = uint8(255 * log(1 + double(grayImg)));
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
    
    % 放大系数
    kx = input('请输入水平放大系数：');
    ky = input('请输入垂直放大系数：');
    
    % 放大图像
    g_scaled = bilinearInterpolationScale(grayImage, kx, ky);
    
    % 旋转角度（逆时针）
    theta = input('请输入旋转角度（度数）：');
    theta = -theta * pi / 180; % 将角度转换为弧度
    
    % 旋转图像
    g_rotated = bilinearInterpolationRotate(g_scaled, theta);
    
    % 保存变换后的图像数据到GUI句柄
    setappdata(hFig, 'scaledImageData', g_scaled);
    setappdata(hFig, 'rotatedImageData', g_rotated);
    
    % 显示变换后的图像
    figure('Name', '放大和旋转后的图像', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
    imshow(g_rotated);
    title('放大和旋转后的图像');
end

function g_scaled = bilinearInterpolationScale(f, kx, ky)
    % 双线性插值放大图像
    [height, width] = size(f);
    newHeight = round(ky * height);
    newWidth = round(kx * width);
    g_scaled = zeros(newHeight, newWidth);
    
    for i = 1:newHeight
        for j = 1:newWidth
            x = (j - 1) / kx + 1;
            y = (i - 1) / ky + 1;
            x1 = floor(x);
            x2 = ceil(x);
            y1 = floor(y);
            y2 = ceil(y);
            x1 = max(1, min(x1, width));
            x2 = max(1, min(x2, width));
            y1 = max(1, min(y1, height));
            y2 = max(1, min(y2, height));
            r1 = f(y1, x1) * (x2 - x) + f(y1, x2) * (x - x1);
            r2 = f(y2, x1) * (x2 - x) + f(y2, x2) * (x - x1);
            g_scaled(i, j) = r1 * (y2 - y) + r2 * (y - y1);
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
    g_rotated = zeros(newHeight, newWidth);
    
    % 旋转矩阵的逆
    t_inv = [cos_theta, sin_theta; -sin_theta, cos_theta];
    
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
                g_rotated(i, j) = f(y1, x1);
            else
                r1 = f(y1, x1) * (x2 - x(1)) + f(y1, x2) * (x(1) - x1);
                r2 = f(y2, x1) * (x2 - x(1)) + f(y2, x2) * (x(1) - x1);
                g_rotated(i, j) = r1 * (y2 - x(2)) + r2 * (x(2) - y1);
            end
        end
    end
end




function noisyImage = addGaussianNoise(image, noiseSigma)
    noisyImage = double(image) + noiseSigma * randn(size(image));
    noisyImage = min(max(noisyImage, 0), 255); % 确保像素值在0-255范围内
end


function filteredImage = meanFilter(image, filterSize)
    [height, width] = size(image);
    filteredImage = zeros(height, width);
    padSize = floor(filterSize / 2);
    paddedImage = padarray(image, [padSize, padSize], 'replicate', 'both');
    
    for i = 1:height
        for j = 1:width
            window = paddedImage(i:i+2*padSize, j:j+2*padSize);
            filteredImage(i, j) = mean(window(:));
        end
    end
end

function H = createGaussianFilter(height, width, filterSize, sigma)
    centerH = ceil(height / 2);
    centerW = ceil(width / 2);
    H = zeros(height, width);
    
    for u = 1:height
        for v = 1:width
            distance = sqrt((u - centerH)^2 + (v - centerW)^2);
            H(u, v) = exp(-(distance^2) / (2 * sigma^2));
        end
    end
end

function filteredImage = frequencyDomainFilter(image, filterSize, sigma)
    % 转换图像到频域
    F = fft2(double(image));
    F = fftshift(F); % 将零频率分量移到频谱中心
    
    % 获取图像尺寸
    [rows, cols] = size(image);
    
    % 创建高斯低通滤波器
    [x, y] = meshgrid(-floor(cols/2):floor(cols/2)-1, -floor(rows/2):floor(rows/2)-1);
    H = exp(-(x.^2 + y.^2) / (2 * sigma^2));
    H = H / sum(H(:)); % 归一化滤波器
    
    % 扩展滤波器大小以匹配图像尺寸
    H = padarray(H, [(rows - filterSize) / 2, (cols - filterSize) / 2], 'both');
    
    % 执行频域乘法
    G = F .* H;
    
    % 反变换回空域
    g = ifftshift(G);
    filteredImage = ifft2(g);
    filteredImage = real(filteredImage); % 只取实部
    filteredImage = min(max(filteredImage, 0), 255); % 确保像素值在0-255范围内
end


function noiseAndFilter(hObject, eventdata)
    hFig = gcf;
    imageData = getappdata(hFig, 'imageData');
    
    if isempty(imageData)
        disp('没有图像数据。请先打开一个图像文件。');
        return;
    end
    
    % 设置噪声参数
    noiseSigma = 20; % 可以根据需要调整噪声强度
    
    % 添加高斯噪声
    noisyImage = addGaussianNoise(imageData, noiseSigma);
    
    % 空域滤波
    filterSize = 3; % 均值滤波器的大小
    spatialFiltered = meanFilter(noisyImage, filterSize);
    
    % 频域滤波
    filterSize = 15; % 高斯低通滤波器的大小
    sigma = 5; % 高斯低通滤波器的标准差
    frequencyFiltered = frequencyDomainFilter(noisyImage, filterSize, sigma);
    
    % 显示结果
    figure;
    subplot(1, 3, 1);
    imshow(uint8(noisyImage), []);
    title('Noisy Image');
    
    subplot(1, 3, 2);
    imshow(uint8(spatialFiltered), []);
    title('Spatial Filtered Image');
    
    subplot(1, 3, 3);
    imshow(uint8(frequencyFiltered), []);
    title('Frequency Filtered Image');
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
    
    % 初始化图形窗口
    figure('Name', 'Edge Detection', 'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none');
    
    % Robert算子
    robertsEdge = edge(grayImage, 'roberts');
    subplot(2, 2, 1);
    imshow(robertsEdge);
    title('Robert Edge Detection');
    axis on;
    
    % Prewitt算子
    prewittEdge = edge(grayImage, 'prewitt');
    subplot(2, 2, 2);
    imshow(prewittEdge);
    title('Prewitt Edge Detection');
    axis on;
    
    % Sobel算子
    sobelEdge = edge(grayImage, 'sobel');
    subplot(2, 2, 3);
    imshow(sobelEdge);
    title('Sobel Edge Detection');
    axis on;
    
    % 拉普拉斯算子
    laplacianEdge = edge(grayImage, 'log'); % 'log' 使用了拉普拉斯高斯算子
    subplot(2, 2, 4);
    imshow(laplacianEdge);
    title('Laplacian Edge Detection');
    axis on;
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
    subplot(1, 2, 1);
    imshow(binaryImage);
    title('Binary Image');
    axis on;
    
    subplot(1, 2, 2);
    imshow(L);
    title(['Labeled Objects - ', num2str(numObjects), ' objects found']);
    axis on;
    
    % 将标记的连通区域图像存储到GUI句柄的 'targetImage' 属性中
    setappdata(hFig, 'targetImage', L);
    disp('目标图像提取并存储成功。');
end


function featureExtraction(hObject, eventdata)
    % 特征提取的回调函数
    hFig = gcf;
    
    % 从GUI句柄中获取图像数据
    imageData = getappdata(hFig, 'imageData');
    targetImage = getappdata(hFig, 'targetImage');
    
    % 如果没有图像数据，则返回
    if isempty(imageData) || isempty(targetImage)
        disp('没有图像数据或目标图像。请先打开一个图像文件并提取目标。');
        return;
    end
    
    % 提取原始图像的LBP特征
    [lbpFeaturesOrig, lbpImageOrig] = extractLBP(imageData);
    disp('原始图像的LBP特征提取完成。');
    
    % 提取原始图像的HOG特征
    hogFeaturesOrig = extractHOG(imageData);
    disp('原始图像的HOG特征提取完成。');
    
    % 提取目标图像的LBP特征
    [lbpFeaturesTarget, lbpImageTarget] = extractLBP(targetImage);
    disp('目标图像的LBP特征提取完成。');
    
    % 提取目标图像的HOG特征
    hogFeaturesTarget = extractHOG(targetImage);
    disp('目标图像的HOG特征提取完成。');
    
    % 可以在这里添加代码来显示特征或进行进一步处理
end

function [lbpFeatures, lbpImage] = extractLBP(image)
    P = 8; % 假设使用8邻域LBP
    R = 1; % 圆形邻域半径
    lbpFeatures = zeros(1, P + 2); % 初始化LBP特征数组，大小为P + 2
    lbpImage = zeros(size(image)); % 初始化LBP图像
    
    [rows, cols] = size(image);
    
    for i = 1:rows
        for j = 1:cols
            % 提取中心像素周围的像素值
            centerPixel = image(i, j);
            pixelValues = zeros(1, P);
            for k = 0:P-1
                theta = k * 2 * pi / P;
                x = round(i + R * cos(theta));
                y = round(j + R * sin(theta));
                % 确保索引在图像范围内
                x = max(1, min(x, rows));
                y = max(1, min(y, cols));
                pixelValues(k + 1) = image(x, y);
            end
            % 计算LBP值
            lbpValue = 0;
            for k = 1:P
                lbpValue = lbpValue + (pixelValues(k) >= centerPixel) * 2^(k-1);
            end
            % 确保lbpValue在特征数组大小范围内
            if lbpValue < length(lbpFeatures)
                lbpFeatures(lbpValue + 1) = lbpFeatures(lbpValue + 1) + 1;
            end
            lbpImage(i, j) = lbpValue;
        end
    end
end

function hogFeatures = extractHOG(image)
    % 简单的HOG特征提取
    % 这个实现非常基础，仅用于示例
    % 实际应用中应使用更完善的实现
    [rows, cols] = size(image);
    cellSize = [8 8]; % 每个单元的大小
    blockSize = [16 16]; % 块的大小
    bins = 9; % 直方图箱数
    
    % 计算图像的梯度
    Gx = [-1 0 1; -2 0 2; -1 0 1];
    Gy = [1 2 1; 0 0 0; -1 -2 -1];
    Ix = conv2(double(image), Gx, 'same');
    Iy = conv2(double(image), Gy, 'same');
    
    % 计算梯度的幅度和方向
    gradientMag = sqrt(Ix.^2 + Iy.^2);
    gradientAngle = atan2(Iy, Ix) * (180/pi); % 转换为度
    
    % 直方图规范化
    hogFeatures = zeros(1, (rows/blockSize(1)-1) * (cols/blockSize(2)-1) * bins);
    idx = 1;
    for i = 1:blockSize(1):rows-cellSize(1)
        for j = 1:blockSize(2):cols-cellSize(2)
            blockMag = gradientMag(i:i+cellSize(1)-1, j:j+cellSize(2)-1);
            blockAngle = gradientAngle(i:i+cellSize(1)-1, j:j+cellSize(2)-1);
            hist = zeros(1, bins);
            for m = 1:cellSize(1)
                for n = 1:cellSize(2)
                    angle = blockAngle(m, n);
                    mag = blockMag(m, n);
                    bin = floor((angle + 180) / 360 * bins);
                    if bin == bins
                        bin = 0;
                    end
                    hist(bin+1) = hist(bin+1) + mag;
                end
            end
            hogFeatures(idx:idx+bins-1) = hist;
            idx = idx + bins;
        end
    end
end


