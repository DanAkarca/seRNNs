%% set pre-requisites and paths
% clear the workspace and command window
clear; clc;
% addpath of bct
addpath('/imaging/astle/users/da04/PhD/toolboxes/2019_03_03_BCT');
% addpath of iosr
addpath('/imaging/astle/users/da04/PhD/toolboxes/MatlabToolbox-master/');
% addpath of voronoi
addpath('/imaging/astle/users/da04/PhD/hd_gnm_generative_models/voronoi');
% addpath of iosr
addpath('/imaging/astle/users/da04/PhD/toolboxes/MatlabToolbox-master');
% addpath of stdshade
addpath('/imaging/astle/users/da04/PhD/toolboxes/stdshade');
% set directories of the data
project = '/imaging/shared/users/ja02/CBUActors/SpatialRNNs/Results/';
% load the decoding summary table
SummaryFrame_sWC = readtable('/imaging/shared/users/ja02/CBUActors/SpatialRNNs/Results/mazeGenI_SE1_sWc_mtIV/SummaryFrameI/SummaryFrameI.csv');
SummaryFrame_L1 = readtable('/imaging/shared/users/ja02/CBUActors/SpatialRNNs/Results/mazeGenI_L1_mtIII/SummaryFrameI/SummaryFrameI.csv');
% load the activity data
load('/imaging/astle/users/da04/PhD/ann_jascha/data/activity.mat'); % se_swc_4
load('/imaging/astle/users/da04/PhD/ann_jascha/data/l1_3_activity.mat');
% load the global structural statistics
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_4_global_statistics.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/l1_3_global_statistics.mat');
% go to the project
cd(project);
% set paths
se_swc_4  = '/mazeGenI_SE1_sWc_mtIV';
l1_3 = '/mazeGenI_L1_mtIII';
%% load structural se_swc_4 networks
% change directory
cd(strcat(project,se_swc_4));
% set the number of networks in here
nnet = 101;
% nnode
nnode = 100;
% set how many timepoints
tt = 11;
% list the directory files
k = dir('*.mat'); 
data      = string({k.name})';
% reorder 
reorder = [1 13 24 35 46 57 68 79 90 101 3:12 14:23 25:34 36:45 47:56 58:67 69:78 80:89 91:100 2];
% reorder the networks
data = data(reorder);
% set parameters
parameters_se_swc_4 = [1:101;linspace(0.001,0.3,101)]';
% initialise
net_se_swc_4 = zeros(nnet,tt,nnode,nnode);
wei_se_swc_4 = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_se_swc_4 = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_se_swc_4(i,t,:,:) = Training_History{t+1,1};
        wei_se_swc_4(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_se_swc_4(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('se_swc_4 network %g loaded',i-1));
end
% get the euclidean space
Dbox = Cost_Matrix;
C = Coordinates;
%% load strutural l1_3 networks
% change directory
cd(strcat(project,l1_3));
% set the number of networks in here
nnet = 101;
% set how many timepoints
tt = 11;
% list the directory files
k = dir('*.mat'); 
data      = string({k.name})';
% reorder the networks
data = data(reorder);
% initialise
net_l1_3 = zeros(nnet,tt,nnode,nnode);
wei_l1_3 = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_l1_3 = zeros(nnet,tt,2); % acc, val acc
parameters_l1_3 = [];
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_l1_3(i,t,:,:) = Training_History{t+1,1};
        wei_l1_3(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_l1_3(i,t,:) = Training_History{t+1,5:6};
        parameters_l1_3(i,:) = [i Regulariser_Strength];
    end
    % display
    disp(sprintf('l1_3 network %g loaded',i-1));
end
%% set activity data
% set activity data
activityset = activity;
% set the structural networks
structure = net_se_swc_4;
% set the accuracy data
accuracy = acc_se_swc_4;
% set the title label
title_label = 'sWC';
% set network range
network = [0:100];
% set epoch strings
epochset = string({...
    '01','02','03','04','05','06','07','08','09','10'});
%% visualise a network and activation
% set epoch
epoch = 6;
% set network
net = 53;
% set neuron
neuron = 2;
% set pre or post
pp = 2;
% take data
data = squeeze(activityset{net,epoch}(neuron,:,:,pp));
% visualise
figure; 
stdshade(data,.5,[.5 .5 .5]);
box off; 
b = gca; 
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 25;
xlim([0 50]);
ylabel('Activation'); xlabel('Step');
hold on;
xline(20); hold on; xline(30); 
% plot all
data_all = squeeze(activity{net,epoch}(:,:,:,pp));
meandata = squeeze(mean(data_all,2));
% sort by largest goal 
t = 10;
[~,ind] = sort(meandata(:,t),'descend');
h = figure; h.Position = [100 100 1200 1200];
col = jet(100);
for i = 1:100;
    u = squeeze(data_all(ind(i),:,:));
    subplot(10,10,i);
    stdshade(u,.25,col(i,:));
    box off; xticks([]); yticks([]);
    ylim([0 1]);
end
%% compute functional connectivity
% set the time windows
timewindows = [1:5;6:10;11:15;16:20;21:25;26:30;31:35;36:40;41:45;46:50];
timewindowsset = {1:30,31:50};
% simplify to average activity across units over time
av_func = []; av_fc = [];
for net = 1:length(network);
    for epoch = 1:length(epochset);
        for set = 1:length(timewindowsset);
            % average of trials and timewindow periods
            b = squeeze(mean(activityset{net,epoch}(:,:,timewindowsset{set},:),2));
            av_func(net,epoch,set,:,:) = squeeze(mean(b,2));
            % form functional connectivity whole window
            for prepost = 1:2;
                av_fc(net,epoch,set,:,:,prepost) = corr(squeeze(b(:,:,prepost))');
            end
            % display progress
            disp(sprintf('network %g epoch %g set %g computed',net,epoch,set));
        end
    end
end
%% visualise functional activity with Euclidean distance
% act label
actlab = {'Pre-act','Post-act'};
% dec label
declab = {'Pre-ons','Post-ons'};
% absolute label
abstlab = {'Abs','Non-Abs'};
% set if we filter by accuracy
acclim = 0;
% make a palette
nstep = 500;
lin = linspace(0,1,nstep)'; 
c = ones(nstep,3); c(:,1)=lin; c(:,2)=lin;
d = ones(nstep,3); d(:,2)=lin; d(:,3)=lin;
col = [d;flip(c)]; 
% initialise figure
h = figure; h.Position = [100 100 1500 650];
% initialise
r = zeros(2,2,2,100,9);
% loop
step = 1;
for act = 1:2;
    for dec = 1:2;
        for abst = 1:2;
            % loop over epochs
            for epoch = 1:9;
                % loop over networks
                for network = 1:100;
                    % get fc network
                    fc_w = squeeze(av_fc(network,epoch,act,:,:,dec));
                    % remove nans
                    fc_w(isnan(fc_w)) = 0;
                    % plot correlations upper 
                    ind = find(triu(fc_w,1));
                    if abst == 1;
                    % get data
                    x = abs(Dbox(ind));
                    y = abs(fc_w(ind));
                    else
                        x = Dbox(ind);
                        y = fc_w(ind);
                    end
                    % correlate
                    r(act,dec,abst,network,epoch) = corr(x,y);
                    % filter by accuracy
                    if accuracy(network,epoch)<acclim;
                        r(act,dec,abst,network,epoch) = 0;
                    end
                end
            end
            % visualise
            subplot(2,4,step);
            imagesc(squeeze(r(act,dec,abst,:,:))) 
            xlabel('Epoch'); ylabel('Network');
            c = colorbar; c.Label.String = 'r'; colormap(col); caxis([-.15 .15]);
            b = gca; b.TickLength = [0 0]; b.FontSize = 8;
            title(sprintf('%s | %s | %s',abstlab{abst},actlab{act},declab{dec}));
            step = step + 1;
        end
    end
end
sgtitle(sprintf('%s Euclidean-function correlations, val accuracy >%g',title_label,acclim));
% visualise
h = figure; h.Position = [100 100 1000 300];
step = 1;
for act = 1:2;
    for dec = 1:2;
        for abst = 1:2;
            u = subplot(2,4,step);
            a = squeeze(r(act,dec,abst,:,:));
            b = a(:);
            histogram(b,'edgecolor','w');
            box off;
            u.TickLength = [0 0];
            b = gca; b.TickLength = [0 0]; b.FontSize = 8;
            title(sprintf('%s | %s | %s',abstlab{abst},actlab{act},declab{dec}));
            xlabel('r');
            step = step + 1;
        end
    end
end
%% visualise a specific Euclidean-function
% set parameters
act = 1; dec = 2; abst = 1; network = 53; epoch = 8;
% get network
fc_w = squeeze(av_fc(network,epoch,act,:,:,dec));
% remove nans
fc_w(isnan(fc_w)) = 0;
% plot correlations upper 
ind = find(triu(fc_w,1));
if abst == 1;
    % get data
    x = abs(Dbox(ind));
    y = abs(fc_w(ind));
else
    x = Dbox(ind);
    y = fc_w(ind);
end
% correlate
[r p] = corr(x,y);
% plot the whole lot to show the step change
figure; scatterhist(x,y,'kernel','on','direction','out','marker','x');
xlabel('Euclidean'); ylabel('FC'); b = gca; b.TickDir = 'out'; b.FontName = 'Arial';
% form bins
bins = linspace(0,max(x),15); binned_dist = {}; bin_dist = [];
for nbin = 1:length(bins)-1;
    lim = [bins(nbin) bins(nbin+1)];
    u = find(x>=lim(1) & x<lim(2));
    binned_dist{nbin} = y(u);
    bin_dist(nbin,1) = mean(y(u));
    bin_dist(nbin,2) = std(y(u));
end
% plot
h = figure; h.Position = [100 100 500 400];
k = errorbar(bin_dist(:,1),bin_dist(:,2)); k.Marker = '.'; k.LineWidth = 3; k.Color = 'r';
k.XData = bins(1:end-1);
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 25;
box off;
sgtitle(sprintf('Network %g | Epoch %g | %s | %s | %s',network,epoch,abstlab{abst},actlab{act},declab{dec}));
xlabel('Euclidean'); ylabel('FC');
xlim([0 inf]);
%% visualise functional activity with structure
% act label
actlab = {'Pre-act','Post-act'};
% dec label
declab = {'Pre-dec','Post-dec'};
% absolute label
abstlab = {'Abs','non-Abs'};
% set if we filter by accuracy 
acclim = 0;
% initialise figure
h = figure; h.Position = [100 100 1500 650];
% initialise
r = zeros(2,2,2,100,9);
% loop
step = 1;
for act = 1:2;
    for dec = 1:2;
        for abst = 1:2;
            % loop over epochs
            for epoch = 1:9;
                % loop over networks
                for network = 1:100;
                    % get fc network
                    fc_w = squeeze(av_fc(network,epoch,act,:,:,dec));
                    % remove nans
                    fc_w(isnan(fc_w)) = 0;
                    % get corresponding strucutre
                    st_w = squeeze(structure(network,epoch,:,:));
                    % plot correlations upper 
                    ind = find(triu(fc_w,1));
                    if abst == 1;
                    % get data
                    x = abs(st_w(ind));
                    y = abs(fc_w(ind));
                    else
                        x = st_w(ind);
                        y = fc_w(ind);
                    end
                    % correlate
                    r(act,dec,abst,network,epoch) = corr(x,y);
                    % filter by accuracy
                    if accuracy(network,epoch)<acclim;
                        r(act,dec,abst,network,epoch) = 0;
                    end
                end
            end
            % visualise
            subplot(2,4,step);
            imagesc(squeeze(r(act,dec,abst,:,:)));
            xlabel('Epoch'); ylabel('Network');
            c = colorbar; c.Label.String = 'r'; colormap(flip(pink)); caxis([-.05 .35]);
            b = gca; b.TickLength = [0 0]; b.FontSize = 8;
            title(sprintf('%s | %s | %s',abstlab{abst},actlab{act},declab{dec}));
            step = step + 1;
        end
    end
end
sgtitle(sprintf('%ss SC-function correlations, val accuracy >%g',title_label,acclim));
% visualise
h = figure; h.Position = [100 100 1000 300];
step = 1;
for act = 1:2;
    for dec = 1:2;
        for abst = 1:2;
            u = subplot(2,4,step);
            a = squeeze(r(act,dec,abst,:,:));
            b = a(:);
            histogram(b,'edgecolor','w');
            box off;
            u.TickLength = [0 0];
            b = gca; b.TickLength = [0 0]; b.FontSize = 8;
            title(sprintf('%s | %s | %s',abstlab{abst},actlab{act},declab{dec}));
            xlabel('r');
            step = step + 1;
        end
    end
end
%% visualise a specific structure-function
% set parameters
act = 1; dec = 2; abst = 1; network = 27; epoch = 8;
% get network
fc_w = squeeze(av_fc(network,epoch,act,:,:,dec));
% remove nans
fc_w(isnan(fc_w)) = 0;
% get corresponding structure
st_w = squeeze(structure(network,epoch,:,:));
% plot correlations upper 
ind = find(triu(fc_w,1));
if abst == 1;
    % get data
    x = abs(st_w(ind));
    y = abs(fc_w(ind));
else
    x = st_w(ind);
    y = fc_w(ind);
end
% correlate
[r p] = corr(x,y);
% form bins
bins = linspace(0,max(x),15); binned_dist = {}; bin_dist = [];
for nbin = 1:length(bins)-1;
    lim = [bins(nbin) bins(nbin+1)];
    u = find(x>=lim(1) & x<lim(2));
    binned_dist{nbin} = y(u);
    bin_dist(nbin,1) = mean(y(u));
    bin_dist(nbin,2) = std(y(u));
end
% plot
h = figure; h.Position = [100 100 500 400];
k = errorbar(bin_dist(:,1),bin_dist(:,2)); k.Marker = '.'; k.LineWidth = 3; k.Color = 'r';
k.XData = bins(1:end-1);
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 25;
box off;
sgtitle(sprintf('Network %g | Epoch %g | %s | %s | %s',network,epoch,abstlab{abst},actlab{act},declab{dec}));
xlabel('SC'); ylabel('FC');
xlim([0 inf]);