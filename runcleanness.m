close all
clear
clc

Iorig = imread('chip04/row3/10x_01.jpg');

I = rgb2gray(Iorig);
tmpl = rgb2gray(imread('template/tmpl.jpg'));

% resize the image
scale = 0.2;
Iorig = imresize(Iorig, scale);
I = imresize(I, scale);
tmpl = imresize(tmpl, scale);

figure, imshow(Iorig);

% calculate padding
bx = size(I, 2);
by = size(I, 1);
tx = size(tmpl, 2); % used for bbox placement
ty = size(tmpl, 1); 

% fft
Ga = fft2(I);
Gb = fft2(tmpl, by, bx);
c = real(ifft2((Ga.*conj(Gb)) ./ abs(Ga.*conj(Gb))));

% find peak correlation
[max_c, imax] = max(abs(c(:)));
[ypeak, xpeak] = find(c==max(c(:)));
% figure, surf(c), shading flat;

position = [xpeak(1), ypeak(1), tx-1, ty-1];

% crop the image
for i=1:3
    tmp = Iorig(:,:,i);
    Icrop(:,:,i) = imcrop(tmp,position);
end

load('trainresult.mat');
Igray = double((Icrop)) / 255;
Iclust = zeros(size(Icrop,1), size(Icrop,2));
Itmpl = zeros(size(Icrop,1), size(Icrop,2));

for i=1:size(Icrop,1)
    for j=1:size(Icrop,2)
        % find euclidean distance of each pixel to each cluster
        e = zeros(1, size(centers, 1));
        for cluster=1:size(centers,1)
            tmp = 0;
            for color=1:3
                tmp = tmp + (Igray(i,j,color) - centers(cluster,color))^2;
            end
            e(1, cluster) = sqrt(tmp);
        end
        
        % find the darkest
        t = 0.29*centers(:,1) + 0.59*centers(:,2) + 0.11*centers(:,3);
        [~,cdarkest] = min(t);
        
        % plot
        [~,cmember(i,j)] = max(e(:));
        if cmember(i,j) == cdarkest
            Iclust(i,j) = 0;
        else
            Iclust(i,j) = 1;
        end
    end
end

figure, imshow(Iclust);

tr = size(Iclust,1)*size(Iclust,2) / 4;
bw = bwareaopen(logical(Iclust), tr);

figure, imshow(bw,[]);

% noclean = Iclust .* double(bw);
noclean = xor(logical(Iclust),bw);

if max(noclean(:))==0
    disp('the pad is clean');
else
    disp('unclean area on the pad is detected');
end

figure, imshow(Icrop);
figure, imshow(noclean);