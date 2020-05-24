warning off;
clear all;

direc = dir('/home/samik/mnt/gpu5b_1/nfs/data/main/M32/Cell_Detection/CellDetPass1_Nissl/res/data2/*.jp2');
load('ctxIds_V3.mat');
for file = 11 : 18%length(direc)

    disp(direc(file).name);
% org = imread(['data1/' direc(i).name]);
%     tic
    org = imread(['/home/samik/mnt/gpu5b_1/nfs/data/main/M32/RegistrationData/Data/DK5_630/Transformation_OUTPUT/DK5_630_img/' direc(file).name]);
    load(['/home/samik/mnt/gpu5b_1/nfs/data/main/M32/RegistrationData/Data/DK5_630/Transformation_OUTPUT/reg_high_seg_pad_V3/' ...
    direc(file).name(1:end-3) 'mat']);
%     toc
    
    %% Create Cortes Mask
    ctxmask = false(size(seg));
    for i = 1 : length(ctxIds)
        ctxmask(find(seg==ctxIds(i))) = true;
    end
    
    %% Create Nisll segmentation
    
    mask = ones([size(org,1), size(org,2)]);
    
    for i = 1:size(org,1)
        for j = 1: size(org,2)
            if org(i,j,1)>150 || org(i,j,2)>150
                mask(i,j) = 0;
            end
        end
    end
    maskB = imbinarize(mask);
    nisslimg = bwfill(maskB, 'holes');
    nisslimg = bwareaopen(nisslimg, 50);
    %% Store borders in structure
    mop_border = borderDet(org, ctxmask, nisslimg);
    mop(file).name = direc(file).name;
    mop(file).border = mop_border;
    mop(file).nisslimg = nisslimg;
    
end