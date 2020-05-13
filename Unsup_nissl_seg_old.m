direc = dir('data2/*.jp2');

%% 249-, -253, -255, -256, -258, -259, -260, -262-, -264, -265, -266, -267,
for file = 19 : length(direc)
    org = imread(['data1/' direc(i).name]);
%     org = imread(['/home/samik/mnt/gpu5b_1/nfs/data/main/M32/RegistrationData/Data/DK5_630/Transformation_OUTPUT/DK5_630_img/' direc(file).name]);
    load(['/home/samik/mnt/gpu5b_1/nfs/data/main/M32/RegistrationData/Data/DK5/Transformation_OUTPUT/reg_high_seg_pad/' ...
        direc(file).name(1:end-3) 'mat']);
    
    %% Generation of Mask %%
    
    mask = ones([size(org,1), size(org,2)]);
    
    for i = 1:size(org,1)
        for j = 1: size(org,2)
            if org(i,j,1)>150 || org(i,j,2)>150
                mask(i,j) = 0;
            end
        end
    end
    maskB = imbinarize(mask);
    maskB = bwfill(maskB, 'holes');
    %% Smooth Boundary
    N = 255;
    
    kernel = ones(N, N, N) / N^3;
    blurryImage = convn(double(seg), kernel, 'same');
    newBinaryImage = blurryImage > 0;
    edgeSeg = edge(newBinaryImage, 'canny');
    [y,x] = find(edgeSeg) ;
    B = [x y];
%     scatter(B(:,1),B(:,2),'r.') ;
    %%
    
    %% Right Half
    f41 = seg(1:size(mask,1)/2, 1 : size(mask,2)/2);
    [r, c] = find(f41);
    rm = min(r);
    cm = min(c);
    f41_c = maskB(rm:size(mask,1)/2, cm:size(mask,2)/2);
    f41_R = org(rm:size(mask,1)/2, cm:size(mask,2)/2, :);
    n41_c = 1-imbinarize(seg(rm:size(mask,1)/2, cm:size(mask,2)/2));
    %% Density
    rbox = zeros(size(f41_c));
    bb = 50;
    
    for i = bb+1 : bb/2: size(f41_c,1)-bb
%         disp(i)
        for j = bb+1 : bb/2: size(f41_c,2)-bb
            rbox(i-bb:i+bb,j-bb:j+bb) = (sum(sum(f41_c(i-bb: i+bb, j-bb:j+bb))));
        end
    end
    rbox = (rbox-min(min(rbox)))/(max(max(rbox))- min(min(rbox)));
    maskR = imgaussfilt(rbox, 51);
    
    edgeA = edge((1-maskR).^1.5, 'canny');
    edgeB = imbinarize(imgaussfilt(uint8(edgeA)*255, 3));
    r3 = bwareaopen(edgeB, 3000);
    R4 = r3 - bwareaopen(edgeB, 15000);
    statsL = regionprops(imbinarize(R4), 'Area', 'Circularity','PixelIdxList') ;
    
    BW3 = imbinarize(R4);
    statsL = regionprops(BW3, 'Area', 'Circularity','PixelIdxList') ;
    
    
    [nz, nzidx] = find(n41_c);
    
    for i = 1 : length(statsL)
%         disp(i);
        [xC, YC] = ind2sub(size(n41_c), statsL(i).PixelIdxList);
        [statsL(i).dist, statsL(i).I] = pdist2([nz nzidx],[xC YC],'euclidean','Smallest',1);
        statsL(i).std = std(statsL(i).dist);
        statsL(i).var = var(statsL(i).dist);
        statsL(i).mn = mean(statsL(i).dist);
    end
    
    R41 = false(size(r3));
    for i = 1 : length(statsL)
        if statsL(i).mn <1000 && statsL(i).std < 50 && statsL(i).mn > 300
            R41(statsL(i).PixelIdxList) = true;
        end
    end

    [idy, idx] = find(R41);
    p = polyfit(idx, idy, 3);
    resXL = max(idx);
    resYL = polyval(p, resXL);
    fullXL = resYL + rm;
    fullYL = resXL + cm;
    
    %% Left half
    
    f41 = seg(1:size(mask,1)/2, size(mask,2)/2:end);
    [r, c] = find(f41);
    rm = min(r);
    cm = max(c);
    f41_c = maskB(rm:size(mask,1)/2, size(mask,2)/2:size(mask,2)/2+cm);
    f41_R = org(rm:size(mask,1)/2, size(mask,2)/2:size(mask,2)/2+cm, :);
    n41_c = 1-imbinarize(seg(rm:size(mask,1)/2, size(mask,2)/2:size(mask,2)/2+cm));
    %% Density    
    rbox = zeros(size(f41_c));
    bb = 50;
    
    for i = bb+1 : bb/2: size(f41_c,1)-bb
%         disp(i)
        for j = bb+1 : bb/2: size(f41_c,2)-bb
            rbox(i-bb:i+bb,j-bb:j+bb) = (sum(sum(f41_c(i-bb: i+bb, j-bb:j+bb))));
        end
    end
    rbox = (rbox-min(min(rbox)))/(max(max(rbox))- min(min(rbox)));
    maskR = imgaussfilt(rbox, 51);
%     maskR = rbox; 
    
    edgeA = edge((1-maskR).^1.5, 'canny');
    edgeB = imbinarize(imgaussfilt(uint8(edgeA)*255, 3));
    r3 = bwareaopen(edgeB, 3000);
    L4 = r3 - bwareaopen(edgeB, 15000);
    statsR = regionprops(imbinarize(L4), 'Area', 'Circularity', 'PixelIdxList') ;
    

    BW4 = imbinarize(L4);
    statsR = regionprops(BW4, 'Area', 'Circularity','PixelIdxList', 'Eccentricity') ;
    
    
    [nz, nzidx] = find(n41_c);
    
    for i = 1 : length(statsR)
%         disp(i);
        [xC, YC] = ind2sub(size(n41_c), statsR(i).PixelIdxList);
        [statsR(i).dist, statsR(i).I] = pdist2([nz nzidx],[xC YC],'euclidean','Smallest',1);
        statsR(i).std = std(statsR(i).dist);
        statsR(i).var = var(statsR(i).dist);
        statsR(i).mn = mean(statsR(i).dist);
    end
    
    L41 = false(size(r3));
    for i = 1 : length(statsR)
        if statsR(i).mn <1000  && statsR(i).std < 50 && statsR(i).mn > 300
            L41(statsR(i).PixelIdxList) = true;
        end
    end

    [idy, idx] = find(L41);
    p = polyfit(idx, idy, 3);
    
    resXR = min(idx);
    resYR = polyval(p, resXR);
    fullXR = resXR + size(mask,2)/2;
    fullYR = resYR +  rm;

    
    %%
    %% line
    [DL, IL] =  pdist2(B,[fullXL fullYL],'euclidean','Smallest',1);
%     scatter(B(IL,1), B(IL,2), 'g')
    [DR, IR] =  pdist2(B,[fullXR fullYR],'euclidean','Smallest',1);
%     scatter(B(IR,1), B(IR,2), 'g')
    
    imshow(org); hold on;
    hLine = imline(gca,[B(IL,1), fullXL], [B(IL,2), fullYL]); 
    BL = hLine.createMask();
    hLine = imline(gca,[B(IR,1), fullXR], [B(IR,2), fullYR]); 
    BL = BL + hLine.createMask();
    
%     imshow(BL);
%     resC = imoverlay(org, BL, 'g');
%     imwrite(resC, ['results/' direc(file).name(1:end-4) '_mask.tif']);
    
%     line([B(IL,1), fullXL], [B(IL,2), fullYL], 'Color', 'green', 'Linewidth', 3);
%     line([B(IR,1), fullXR], [B(IR,2), fullYR], 'Color', 'green', 'Linewidth', 3);
%     
    save(['results3/new_' direc(file).name(1:end-3) 'mat'], 'BL', '-v7.3');
    
    hold off;
    close all;
    
%     saveas(gcf,[direc(i).name(1:end-4) '_border.tif']);
end