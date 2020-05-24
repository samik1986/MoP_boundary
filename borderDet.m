function mop_border = borderDet(org, ctxmask, nisslimg)
%%  Inputs :
%   org : Original Nissl Image
%   

close all;
%% Initializations

p = 0.00009; % epsilon
pt_step = 1000; % sampling size along normal
ft = [];
flatIndex = []; % storage for index
delta = 1000; % smoothing parameter
lineLen = 1500; % Length of Normal

sampling = 1; % use 1 for full size
%% Use for downsizing
orgL = org(1:sampling:end,1:sampling:end,:);
ctxmaskL = ctxmask(1:sampling:end,1:sampling:end);


sigma = 51;
wsize = 101;
h = fspecial('gaussian', [wsize wsize], sigma);
nisslimgG = imfilter(single(nisslimg), h, 'replicate');
nisslimgL = nisslimgG(1:sampling:end,1:sampling:end) .* ctxmaskL; % mask along cortex boundary

%% Get medial axis and angle from centroid
[shiftedX, shiftedY, shiftedTheta] = get_medial_axis(ctxmaskL);

%% Smoothing spline fit and the angle of the normal 
smooth_shiftedY = runline1(shiftedY, 1000, 1);
smooth_shiftedX = runline1(shiftedX, 1000, 1);
d_smooth_shiftedX = d_runline(shiftedX, 1000, 1);
d_smooth_shiftedY = d_runline(shiftedY, 1000, 1);

m_smooth = atan2(-d_smooth_shiftedY,d_smooth_shiftedX);

for lTheta = 2 : 8 : length(m_smooth)
    flatIndex = [flatIndex lTheta];
%     disp(lTheta);
    [xx1,yy1] =  calculate_cortical_normal(m_smooth, ...
        smooth_shiftedX, smooth_shiftedY, ...
        lineLen, lTheta, pt_step, ctxmaskL);

    %% Along the normal calcuate the profile
    ftN = [];
    for j = 1 : length(xx1)
        dnstyMat = nisslimgL(int16(xx1(j)), int16(yy1(j)));
        ftN = [ftN; dnstyMat];
    end

    %% Flattening of the cortical density map    
    ft = [ft ftN];
%     plot(yy1, xx1, 'r')
end

%% FInd Boundary in profile

ftCrop = ft(200:800, floor(0.2*length(ft)):floor(0.8*length(ft)));


ftCropS = imgaussfilt(single(ftCrop),11);
ftCropB = imcomplement(ftCrop>0.2);
ftSG = imgaussfilt(single(ftCropB), 21);
ftG = ftSG>0.2;
ftGrefined = bwareaopen(ftG, 10000);

%% Remove medial region
R = regionprops(ftGrefined,'Orientation', 'PixelIdxList');
for idxP = 1 : length(R)
    if abs(R(idxP).Orientation) > 85
        ftGrefined(R(idxP).PixelIdxList)= 0;
    end
end

%% divide into hemispheres
midPoint = floor(length(flatIndex)/2);
ftGL = ftGrefined(:, 1:midPoint);
ftGR = ftGrefined(:, midPoint:end);

%% Find the index of the boundary
[x, y] = find(ftGL);
vLTheta = flatIndex(max(y));
[x, y] = find(ftGR);
vRTheta = flatIndex(min(y)+midPoint);

%% Find the boundary
[xxL, yyL] = calculate_cortical_normal(m_smooth, ...
    smooth_shiftedX, smooth_shiftedY, ...
    lineLen, vLTheta, pt_step, ctxmaskL);

[xxR, yyR] = calculate_cortical_normal(m_smooth, ...
    smooth_shiftedX, smooth_shiftedY, ...
    lineLen, vRTheta, pt_step, ctxmaskL);

mop_border.yyL = yyL;
mop_border.xxL = xxL;
mop_border.yyR = yyR;
mop_border.xxR = xxR;
mop_border.ftCrop = ftCrop;
%% Show boundary
% h = figure;
% imshow(org); hold on;
% plot(yyL, xxL, 'r');
% plot(yyR, xxR, 'r');
% savefig(h, ['results/' name(1:end-3) 'fig']);
end

