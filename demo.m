%% Demo file for the usage of SSBiEM

clear all;
close all;
clc;

%% generating data

data = zeros(30,40);
data(8:15,17:27) = 1;
data(17:25,2:15) = 1;
data = data + rand(size(data))*0.1;

figure;imagesc(data);

%% running SSBiEM
nbic = 2;
res = SSBiEM(data,nbic);

figure;
subplot(2,2,1); imagesc(data); title('Original Data');
subplot(2,2,2); imagesc(res.V * res.Z); title('Reconstructed Matrix');
for i = 1 : nbic
    subplot(2,2,i+2);
    imagesc(res.h(:,i) * res.g(i,:))
    title (['Bicluster #',int2str(i)]);
end