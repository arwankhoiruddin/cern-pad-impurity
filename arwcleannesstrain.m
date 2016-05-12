close all
clear
clc

Iorig = imread('chip01/row1/10x_11.jpg');

I = rgb2gray(Iorig);
tmpl = rgb2gray(imread('template/tmpl.jpg'));

% resize the image
scale = 0.2;
Iorig = imresize(Iorig, scale);
I = imresize(I, scale);
tmpl = imresize(tmpl, scale);

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

position = [xpeak(1), ypeak(1), tx, ty];

% crop the image
for i=1:3
    tmp = Iorig(:,:,i);
    Icrop(:,:,i) = imcrop(tmp,position);
end

% change input into (3 x mn)
input = zeros(size(Icrop,1)*size(Icrop,2), 3);
for i=1:3
    tmp = Icrop(:,:,i);
    input(:,i) = double(tmp(:))/255;
end

options = [NaN 300 0.001 0];
% [centers, U, objFun] = fcm(input, 3, options);
[centers, U, objFun] = fcm(input, 3);

save('trainresult.mat', 'centers');

% end of train here. The next lines is to test the algorithm

tmp = zeros(size(Icrop,1), size(Icrop, 2), 3);
for i=1:3
    tmp(:,:,i) = reshape(U(i,:), size(Icrop,1), size(Icrop,2));
end

% find the brightest cluster
% brightest cluster is the pad
t = centers*255;
t = 0.29*t(:,1) + 0.59*t(:,2) + 0.11*t(:,3);
[~,cbrightest] = max(t);

Iclust = zeros(size(Icrop,1), size(Icrop, 2));
for i=1:size(Icrop,1)
    for j=1:size(Icrop,2)
        [~,loc] = max(tmp(i,j,:));
        if loc==cbrightest
            Iclust(i,j) = 0;
        else
            Iclust(i,j) = 1;
        end
    end
end
bw = im2bw(Iclust, 0.6);
bw = bwareaopen(bw, 500);

kotoran = Iclust - double(bw);

figure, imshow(Icrop);
figure, imshow(bw,[]);
figure, imshow(kotoran,[]);