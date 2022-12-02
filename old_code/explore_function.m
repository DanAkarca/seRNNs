%% explore function
% written dr danyal akarca
%% set pre-requisites and paths
% clear the workspace and command window
clear; clc;
% addpath of bct
addpath('/imaging/astle/users/da04/PhD/toolboxes/2019_03_03_BCT');
% addpath of iosr
addpath('/imaging/astle/users/da04/PhD/toolboxes/MatlabToolbox-master/');
% addpath of voronoi
addpath('/imaging/astle/users/da04/PhD/hd_gnm_generative_models/voronoi');
% addpath of layout
addpath('/imaging/astle/users/da04/PhD/toolboxes/layoutCode');
% addpath of Cohen's d
addpath('/imaging/astle/users/da04/PhD/toolboxes/computeCohen');
% addpath of sort files
addpath('/imaging/astle/users/da04/PhD/toolboxes/sort_nat');
% addpath of python colours
addpath('/imaging/astle/users/da04/PhD/toolboxes/Colormaps/Colormaps (5)/Colormaps/');
% addpath of colorbrewer colours
addpath('/imaging/astle/users/da04/PhD/toolboxes/colorBrewer');
% addpath of stdshade
addpath('/imaging/astle/users/da04/PhD/toolboxes/stdshade');
% set directories of the data
project = '/imaging/shared/users/ja02/CBUActors/SpatialRNNs/Results/';
% load the decoding summary table
SummaryFrame_L1 = readtable('/imaging/shared/users/ja02/CBUActors/SpatialRNNs/Results/mazeGenI_L1_mtIII/SummaryFrameI/SummaryFrameI.csv');
SummaryFrame_sE = readtable('/imaging/shared/users/ja02/CBUActors/SpatialRNNs/Results/mazeGenI_SE1_mtII/SummaryFrameI/SummaryFrameI.csv');
SummaryFrame_sWC = readtable('/imaging/shared/users/ja02/CBUActors/SpatialRNNs/Results/mazeGenI_SE1_sWc_mtIV/SummaryFrameI/SummaryFrameI.csv');
% form a summary array of each network
summary_array_swc_4 = table2array(SummaryFrame_sWC);
summary_array_l1_3 = table2array(SummaryFrame_L1);
summary_array_se_2 = table2array(SummaryFrame_sE);
% set up the sub-parts
str = {'0_101','101_202','202_303','303_404','404_505','505_606','606_707','707_808','808_909','909_1001'};
% initialise
summary_array_swc_4_1k = [];
summary_array_l1_3_1k = [];
% loop over sub-parts
for s = 1:length(str);
    % load tables
    summary_array_swc_4_1k_tab = table2array(readtable(sprintf(...
        '/imaging/shared/users/ja02/CBUActors/SpatialRNNs/Results/mazeGenI_SE1_sWc_mtIV_1k/SummaryFrameII/SummaryFrameII_%s.csv',str{s})));
    summary_array_l1_3_1k_tab = table2array(readtable(sprintf(...
        '/imaging/shared/users/ja02/CBUActors/SpatialRNNs/Results/mazeGenI_L1_mtIII_1k/SummaryFrameII/SummaryFrameII_%s.csv',str{s})));
    % update
    summary_array_swc_4_1k = [summary_array_swc_4_1k;summary_array_swc_4_1k_tab];
    summary_array_l1_3_1k = [summary_array_l1_3_1k;summary_array_l1_3_1k_tab];
    % display
    disp(sprintf('sWC & L1 %s 1k summary frames loaded...',str{s}));
end
% load the global statistics
load('/imaging/astle/users/da04/PhD/ann_jascha/data/l1_3_global_statistics.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_1_global_statistics.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_2_global_statistics.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_4_global_statistics.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_4r_global_statistics.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_4_1k_global_statistics.mat');
% load embedding
load('/imaging/astle/users/da04/PhD/ann_jascha/data/l1_3_1k_all_embedding_epoch9.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_4_1k_all_embedding_epoch9.mat');
% go to the project
cd(project);
% set paths
l1_3 = '/mazeGenI_L1_mtIII';
se_2 = '/mazeGenI_SE1_mtII';
se_swc_4  = '/mazeGenI_SE1_sWc_mtIV';
se_swc_4r  = '/mazeGenI_SE1_sWc_mtIV_randI';
se_swc_4_1k  = '/mazeGenI_SE1_sWc_mtIV_1k';
l1_3_1k = '/mazeGenI_L1_mtIII_1k';
%% load structural networks

%%% load structural se_swc_4 networks %%%

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

%%% load l1_3 networks %%%

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

%%% load se_2 networks %%%

% change directory
cd(strcat(project,se_2));
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
net_se_2 = zeros(nnet,tt,nnode,nnode);
wei_se_2 = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_se_2 = zeros(nnet,tt,2); % acc, val acc
parameters_se_2 = [];
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_se_2(i,t,:,:) = Training_History{t+1,1};
        wei_se_2(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_se_2(i,t,:) = Training_History{t+1,5:6};
        parameters_se_2(i,:) = [i Regulariser_Strength];
    end
    % display
    disp(sprintf('se_2 network %g loaded',i-1));
end

%%% load se_swc_4_1k networks %%%

% change directory
cd(strcat(project,se_swc_4_1k));
% set the number of networks in here
nnet = 1001;
% set how many timepoints
tt = 11;
% list the directory files
k = dir('*.mat'); 
data      = string({k.name})';
% reorder the 1k networks
[s,reorder1k] = sort_nat({k.name});
% reorder the networks
data = data(reorder1k);
% set parameters
parameters_se_swc_4_1k = [1:1001;linspace(0.001,0.3,1001)]';
% initialise
net_se_swc_4_1k = zeros(nnet,tt,nnode,nnode);
wei_se_swc_4_1k = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_se_swc_4_1k = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_se_swc_4_1k(i,t,:,:) = Training_History{t+1,1};
        wei_se_swc_4_1k(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_se_swc_4_1k(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('se_swc_4_1k network %g loaded',i-1));
end

%%% load l1_3_1k networks %%%

% change directory
cd(strcat(project,l1_3_1k));
% set the number of networks in here
nnet = 1001;
% set how many timepoints
tt = 11;
% list the directory files
k = dir('*.mat'); 
data      = string({k.name})';
% reorder the networks
data = data(reorder1k);
% initialise
net_l1_3_1k = zeros(nnet,tt,nnode,nnode);
wei_l1_3_1k = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_l1_3_1k = zeros(nnet,tt,2); % acc, val acc
parameters_l1_3_1k = [];
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_l1_3_1k(i,t,:,:) = Training_History{t+1,1};
        wei_l1_3_1k(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_l1_3_1k(i,t,:) = Training_History{t+1,5:6};
        parameters_l1_3_1k(i,:) = [i Regulariser_Strength];
    end
    % display
    disp(sprintf('l1_3_1k network %g loaded',i-1));
end
%% explore accuracy
% set accuracy data to explore
accuracy = squeeze(acc_se_swc_4_1k(:,:,2));
% visualise
figure;
subplot(1,2,1); 
imagesc(accuracy(:,2:end)); 
xlabel('Epoch'); ylabel('Network'); c = colorbar; caxis([0 1]); 
b = gca; b.TickLength = [0 0]; b.FontName = 'Arial'; b.FontSize = 12;
subplot(1,2,2);
v = sum(accuracy(:,2:end),2);
barh(v); 
xlabel('sum(accuracy)'); ylabel('Network');
b = gca; b.TickLength = [0 0]; b.FontName = 'Arial'; b.FontSize = 12;
%% compute basic statistics across all data
% set which array to use
summary_array = summary_array_swc_4_1k;
% set a string
dataTitle = 'swc_4_1k';
% set range of networks to evaluate
networks = [1:1:1000];
% set number of networks
nnetworks = 1000;
% keep titles
Headers = string(SummaryFrame_sWC.Properties.VariableNames);
% form a newer one
headers = string(...
    {'Network',...
    'Unit',...
    'Epoch',...
    'Time Window',...
    'Accuracy',...
    'Val Accuracy',...
    'Regu Strength',...
    'EV Goal',...
    'EV Choices',...
    'EV Correct Choice',...
    'Input Entropy',...
    'Output Entropy'});
% correlation matrix of set variables
setv = [3 5 7 8 9 10 11 12];
% form a colormap
nstep = 500;
lin = linspace(0,1,nstep)'; 
c = ones(nstep,3); c(:,1)=lin; c(:,2)=lin;
d = ones(nstep,3); d(:,2)=lin; d(:,3)=lin;
col = [d;flip(c)];
% correlation across all
[r p] = corr(summary_array(:,setv)); 
% settings
h = figure; h.Position = [100 100 600 600];
imagesc(r);
xticklabels(headers(setv)); xtickangle(45); 
yticklabels([]);
b = gca; b.TickLength = [0 0]; b.FontName = 'Arial'; b.FontSize = 25;
caxis([-1 1]); c = colorbar; c.Label.String = 'r'; colormap(col);
%% basic statistics across criteria
% set epochs to loop
epochset = [1:9];
% set time windows to loop
timewindowset = [1:9];
% correlation matrix of set variables
setv = [5 7 8 9 10 11 12];
% initialise figures
h = figure; h.Position = [100 100 1800 225];
% loop over epochs
for epoch = 1:length(epochset);
    % get specific epoch
    criteria = summary_array(:,3)==epochset(epoch);
    % correlation
    [r p] = corr(summary_array(criteria,setv)); 
    % settings
	subplot(1,length(epochset),epoch);
    % plot
    imagesc(r);
    xticks(1:length(setv));
    xticklabels(headers(setv)); xtickangle(45); 
    yticklabels([]);
    b = gca; b.TickLength = [0 0]; b.FontName = 'Arial';
    caxis([-1 1]); colormap(col);
    title(sprintf('%g',epochset(epoch)));
end
sgtitle('Epoch correlations');
% initialise figures
h = figure; h.Position = [100 100 1800 225];
% loop over epochs
for tw = 1:length(timewindowset);
    % get specific time window
    criteria = summary_array(:,4)==timewindowset(tw);
    % correlation
    [r p] = corr(summary_array(criteria,setv)); 
    % settings
	subplot(1,length(timewindowset),tw);
    % plot
    imagesc(r);
    xticks(1:length(setv));
    xticklabels(headers(setv)); xtickangle(45); 
    yticklabels([]);
    b = gca; b.TickLength = [0 0]; b.FontName = 'Arial';7
    caxis([-1 1]); colormap(col);
    title(sprintf('%g',timewindowset(tw)));
end
sgtitle('Time window correlations');
%% plot correlations over time windows in terms of correlations
% set epoch
epoch = 9;
% plot over time windows
tws = [6 7 8 9];
% set a accuracy lower limit
acclim = 0.9;
% set n clusters
nclus = 2;
% form colormap
col = 1-summer(length(tws));
% initialise
r = nan(nnetworks,length(tws));
m = zeros(nnetworks,length(tws));
sil = nan(nnetworks,length(tws));
exp = nan(nnetworks,2,length(tws));
% compute
for i = 1:length(tws);
    % display
    disp(sprintf('time window %g...',tws(i)));
    for network = 1:nnetworks;
        % display
        disp(sprintf('time window %g, network %g...',tws(i),network));
        % get indicator
        ind = ...
            summary_array(:,1)==network & ...
            summary_array(:,3)==epoch & ...
            summary_array(:,4)==tws(i) & ...
            summary_array(:,6)>acclim;
        % contingent on in not empty
        if sum(ind)>0;
        % get variables
        x = summary_array(ind,8);
        y = summary_array(ind,9);
        % get correlation
        r(network,i) = corr(x,y);
        % compute a clustering statistic
        % d = [x y];
        % e = kmeans(d,nclus);
        % u = silhouette(d,e);
        % sil(network,i) = mean(u);
        % compute principle eigenvector
        % [~,~,~,~,exp(network,:,i)] = pca(d);
        else
            % display
            disp(sprintf('time window %g, network %g below the accuracy threhsold...',tws(i),network));
        end
    end
end
% visualise correlations
h = figure; h.Position = [100 100 800 700];
for i = 1:length(tws);
    % get the correlations
    x = 1:nnetworks;
    y = r(:,i);
    % filter ones with no nan
    x(isnan(y)) = [];
    y(isnan(y),:) = [];
    % plot scatter
    h = scatter(x,y,'o',...
        'sizedata',100,...
        'markerfacecolor',col(i,:),...
        'markeredgecolor','w');
    hold on;
    % plot line of best fit
    p = polyfit(x,y,1);
    yfit = polyval(p,x);
    u = plot(x,yfit);
    u.LineWidth = 4;
    u.Color = col(i,:);
    hold on;
end
sgtitle(sprintf('%s | Epoch = %g | Validation accuracy > %g',dataTitle,epoch,acclim));
ylabel('Correlation'); xlabel('Network');
ylim([-1 1]);
% plot correlation of zero too
yline(0,'color','k','linewidth',4);
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 25;
%{
% plot the principle eigenvector variance explained
h = figure; h.Position = [100 100 1200 600];
for component = 1:2;
    subplot(1,2,component);
    for i = 1:length(tws);
        % get the explained variance
        x = 1:100;
        y = exp(:,component,i);
        % filter ones with no nan
        x(isnan(y)) = [];
        y(isnan(y),:) = [];
        % plot scatter
        h = scatter(x,y,'o',...
            'sizedata',200,...
            'markerfacecolor',col(i,:),...
            'markeredgecolor','w');
        hold on;
        % plot line of best fit
        p = polyfit(x,y,1);
        yfit = polyval(p,x);
        u = plot(x,yfit);
        u.LineWidth = 4;
        u.Color = col(i,:);
        hold on;
    end
    ylim([0 100]);
    ylabel(sprintf('PC%g (%%)',component)); xlabel('Network');
    b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 25;
end
sgtitle(sprintf('%s | Epoch = %g | Validation accuracy > %g',dataTitle,epoch,acclim));
% visualise community structure
h = figure; h.Position = [100 100 800 700];
for i = 1:length(tws);
    % get the silhoutte scores
    x = 1:100;
    y = sil(:,i);
    % filter ones with no nan
    x(isnan(y)) = [];
    y(isnan(y),:) = [];
    % plot scatter
    h = scatter(x,y,'o',...
        'sizedata',200,...
        'markerfacecolor',col(i,:),...
        'markeredgecolor','w');
    hold on;
    % plot line of best fit
    p = polyfit(x,y,1);
    yfit = polyval(p,x);
    u = plot(x,yfit);
    u.LineWidth = 4;
    u.Color = col(i,:);
    hold on;
end
ylim([0 1]);
sgtitle(sprintf('%s | Epoch = %g | Validation accuracy > %g | k = %g',dataTitle,epoch,acclim,nclus));
ylabel('Mean silhouette score'); xlabel('Network');
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 25;
%% compute explained variances
% set summary array
summary_array = summary_array_swc_1k;
% set epoch 
epoch = 6;
% take the networks
goal = []; choice = [];
for tw = 1:9;
    for net = 1:nnetworks;
        % get index
        idx = summary_array(:,1) == net & summary_array(:,3) == epoch & summary_array(:,4) == tw;
        % get explained variance
        goal(net,tw,:) = summary_array(idx,8)';
        choice(net,tw,:) = summary_array(idx,9)';
    end
end
%}
%% compute network statistiscs
% set what networks to compute in the group
ns = [1:1:1001]; 
ns(1)=1;
% set the number of local and global measures calculated
nlmeasures = 6;
ngmeasures = 8;
% set number of permutations for small worldness
nperm = 100;
% binarisation proportion threshold for small worldness
thr = 0.1;
% set networks
u = net_l1_3;
% global statistics labels
global_label = string(...
    {'Strength',...
    'Weighted edge length',...
    'Efficiency',...
    'Homophily per weight',...
    'Modularity',...
    'Efficiency per weight',...
    'Corr(weight,distance',...
    'Small-worldness'});
%{
% local statistics labels
local_label = string(...
    {'strength',...
    'clustering',...
    'betweenness',...
    'weighted edge length',...
    'communicability',...
    'matching'});
%}
%{
% initialise
local_statistics = zeros(length(ns),size(u,2),nnode,nlmeasures);
global_statistics = zeros(length(ns),size(u,2),ngmeasures);
topological_organization = zeros(length(ns),size(u,2),nlmeasures,nlmeasures);
% loop over the groups and compute global statistics
display('computing local, tf and global statistics...');
% loop over the nets
for i = 1:length(ns);
    train = size(u,2);
    % loop over training
    for t = 1:train;
        % get network
        n = squeeze(u(ns(i),t,:,:));
        % optional step to make all connections positive
        n = abs(n);
        % take binarisation
        a = threshold_proportional(n,thr); a = double(a>0);
        % compute local statistics
        %{
        local_statistics(i,t,:,1) = strengths_und(n)';
        local_statistics(i,t,:,2) = clustering_coef_wu(n);
        local_statistics(i,t,:,3) = betweenness_wei(n);
        local_statistics(i,t,:,4) = sum(n*Dbox)';
        local_statistics(i,t,:,5) = mean(expm(n))';
        local_statistics(i,t,:,6) = mean((matching_ind(a)+matching_ind(a)'))'; % binarisation
        % compute tf
        topological_organization(i,t,:,:) = corr(squeeze(local_statistics(i,t,:,:)));
        %}
        % compute global statistics
        global_statistics(i,t,1) = sum(n,'all');
        global_statistics(i,t,2) = mean(n*Dbox,'all');
        global_statistics(i,t,3) = efficiency_wei(n);
        global_statistics(i,t,4) = mean(squeeze(local_statistics(i,:,6)))./sum(n,'all');
        [~,global_statistics(i,t,5)] = modularity_und(n);
        global_statistics(i,t,6) = global_statistics(i,t,3)./sum(n,'all');
        global_statistics(i,t,7) = corr(Dbox(n>0),n(n>0));
        % small worldness % binarisation
        % compute nnode
        nnode = size(a,1);
        % compute number of edges
        m = nnz(a)/2;
        % compute observed
        clu = mean(clustering_coef_bu(a));
        cpl = charpath(a);
        % initialise
        clu_perm = [];
        cpl_perm = [];
        % compute nulls
        for j = 1:nperm
            % form a random
            Arand = makerandCIJ_und(nnode,m);
            % form a lattice
            clu_perm(j) = mean(clustering_coef_bu(Arand));
            cpl_perm(j) = charpath(Arand);
        end
        % calclate means
        mean_clu_perm = mean(clu_perm);
        mean_cpl_perm = mean(cpl_perm);
        % calculate smw
        global_statistics(i,t,8) = [clu/mean_clu_perm]/[cpl/mean_cpl_perm];
        % display training point
        display(sprintf('network %g, training point %g computed...',ns(i),t));
    end
end
%}
%% compute tsnes over training
% set network
network = net_se_swc_4_1k;
% get all networks together
tsne_inputs = [reshape(network,[1001*11,100,100])];
% take only upper triangle
ix = triu(ones(100),1);
tsne_i = [];
for i = 1:size(tsne_inputs,1);
    e = squeeze(tsne_inputs(i,:,:));
    tsne_i(i,:) = e(find(ix))';
end
% compute
y = tsne(tsne_i);
% set total epochs
nepoch = 11;
% form colour over epochs
idx_epoch = []; 
step = 1;
for i = 1:nepoch;
    idx_epoch(step:step+100)=i;
    step = step+101;
end
% visualise over epochs
h = figure; h.Position = [100 100 700 600];
u = gscatter(y(:,1),y(:,2),idx_epoch,pink(11),'.',10,'off');
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 12; box off;
xlabel('tSNE1'); ylabel('tSNE2');
c = colorbar; c.Label.String = 'Epoch'; c.Ticks = 1:nepoch; caxis([0 11]);
c.TickDirection = 'out';
colormap(pink(11));
% form colour over regularisations
idx_reg = [1:101 1:101 1:101 1:101 1:101 1:101 1:101 1:101 1:101 1:101 1:101]';
% visualise over regularisation
h = figure; h.Position = [100 100 700 600];
u = gscatter(y(:,1),y(:,2),idx_reg,flip(bone(101)),'.',10,'off');
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 12; box off;
xlabel('tSNE1'); ylabel('tSNE2');
c = colorbar; c.Label.String = 'Regularisation';
c.TickDirection = 'out';
c.TickLabels = c.Ticks.*max(parameters_se_swc_4(:,2));
colormap(flip(bone(101)));
% comparing groups
% get all networks together
tsne_inputs = [reshape(net_se_swc_4,[101*11,100,100]);reshape(net_l1_3,[101*11,100,100]);];
% take only upper triangle
ix = triu(ones(100),1);
tsne_i = [];
for i = 1:size(tsne_inputs,1);
    e = squeeze(tsne_inputs(i,:,:));
    tsne_i(i,:) = e(find(ix))';
end
% compute
y = tsne(tsne_i);
% groups
groups = [ones(101*11,1);2*ones(101*11,1)];
% visualise over epochs
h = figure; h.Position = [100 100 600 400];
u = scatterhist(y(:,1),y(:,2),...
    'group',groups,...
    'kernel','on',...
    'marker','.',...
    'linestyle','-',...
    'direction','out',...
    'color',[143 188 219; 41 64 82]./256);
b = gca; b.TickDir = 'out'; box off; b.FontName = 'Arial'; b.FontSize = 12; box off;
a = legend({'sWC','L1'},'box','off','fontsize',12);
xlabel('tSNE1'); ylabel('tSNE2');
%% run the same analysis but limited to those with structural information
% take a specific time-window and epoch to compare
tw = 9;
% take a accuracy threshold
acclim = 0.9;
% set epoch
epoch = 9;
% set networks to view
netsview = [1:1000];
% initialise
summary_comp = zeros(length(netsview),12); ms = zeros(length(netsview),1);
% loop over networks
for network = 1:length(netsview);
    % get the net
    net = ns(network);
    % get index
    ine = summary_array(:,1)==netsview(net) & summary_array(:,3)==epoch & summary_array(:,4)==tw & summary_array(:,6)>acclim;
    if sum(ine)>1;
        % mean
        summary_comp(network,:) = mean(summary_array(ine,:));
        ms(network) = corr(summary_array(ine,8),summary_array(ine,9));
    else summary_comp(network,:) = NaN;
        ms(network) = NaN;
    end
    % display
    disp(sprintf('epoch %g network %g complete',epoch,network));
end
% get tw 6 goal and tw9 choice
% take a specific time-windows
tws = [6 9];
% take a accuracy threshold
acclim = 0.9;
% set epoch
epoch = 9;
% set networks to view
netsview = [1:1000];
% initialise
goal_choice = zeros(length(tws),length(netsview),2);
% loop over networks
for tw = 1:length(tws);
for network = 1:length(netsview);
    % get the net
    net = ns(network);
    % get index
    ine = summary_array(:,1)==netsview(net) & summary_array(:,3)==epoch & summary_array(:,4)==tws(tw) & summary_array(:,6)>acclim;
    if sum(ine)>1;
        % mean
        goal_choice(tw,network,:) = mean(summary_array(ine,[8 9]));
    end
    % display
    disp(sprintf('tw%g epoch %g network %g complete',tws(tw),epoch,network));
end
end
% save
%{
summary_comp_epo9_tw9_accthr90 = summary_comp;
summary_comp_epo9_tw6_goal_tw9_choice_accthr90 = goal_choice;
save('/imaging/astle/users/da04/PhD/ann_jascha/data/summary_comp_epo9_tw9_accthr90.mat','summary_comp_epo9_tw9_accthr90');
save('/imaging/astle/users/da04/PhD/ann_jascha/data/summary_comp_epo9_tw6_goal_tw9_choice_accthr90.mat','summary_comp_epo9_tw6_goal_tw9_choice_accthr90');
%}
%% visualise structure function relationships
% load
load('/imaging/astle/users/da04/PhD/ann_jascha/data/summary_comp_epo9_tw9_accthr90.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/summary_comp_epo9_tw6_goal_tw9_choice_accthr90.mat');

% assign
summary_comp = summary_comp_epo9_tw9_accthr90;
summary_comp_goal_choice = summary_comp_epo9_tw6_goal_tw9_choice_accthr90;

% set the global statistics to compute
global_statistics = se_swc_4_1k_global_statistics(netsview+1,:,:);
% set the label
netlab = 'seRNN';
% set the embedding data
embeddingdata = se_swc_4_1k_all_embedding_epoch9;
embedding_goal = embeddingdata.pgoal(netsview,tw);
embedding_choice = embeddingdata.pchoice(netsview,tw);

% remove the accuracy thresholds
embedding_goal(embeddingdata.accuracy(netsview,tw)<acclim) = [];
embedding_choice(embeddingdata.accuracy(netsview,tw)<acclim) = [];

% remove nans from functional data
ind = isnan(sum(summary_comp'));
summary_comp_n = summary_comp;
ms_n = ms;
summary_comp_n(ind,:) = [];
ms_n(ind) = [];

% remove non required structural data
rem_structure = [2 4 6];
struc = squeeze(global_statistics(:,epoch+1,:)); 
struc(ind,:) = []; 
struc(:,rem_structure) = [];

% remove non required functional data
% we now do not require epoch, unit, time data, accuracy, val accuracy also remove homophily per weight as it was not calculated
rem_function = [1 2 3 4 5 6 10 11 12];
funct = summary_comp_n; funct(:,rem_function) = [];
% add on the ms after function
funct = [funct ms_n];

% concatenate data
summary_cat = [struc embedding_goal embedding_choice funct];
% concatenate correpsonding labels
struclab = global_label; struclab(rem_structure) = [];
functlab = headers; functlab(rem_function) = []; 
cat_labels = [struclab 'p_{perm} goal','p_{perm} choice' functlab 'corr(Goal,Choice)'];

% form manual version
cat_labels = string({'Strength','Efficiency','Modularity (Q)','corr(W,D)',...
    'SMW (\sigma)','p_{perm} goal','p_{perm} choice','Reg','EV goal','EV choice','corr(goal,choice)'});

% compute correlations
[r p] = corr(summary_cat); 

% reset the order
ind = [1:11];
u = modularity_und(r);
order = [ind(u==1) ind(u==2)];
order = [8 10 3 7 5 4 9 11 6 1 2];
% reorder the correaltion matrix 
r = r(order,order);

% visualise
h = figure; h.Position = [100 100 700 600];
u = imagesc(r);
xticks(1:length(cat_labels));
yticks(1:length(cat_labels));
%xticklabels(cat_labels(order));
%xtickangle(45);
b = gca; b.TickLength = [0 0]; b.FontName = 'Arial'; b.FontSize = 16;
caxis([-1 1]); c = colorbar; c.Label.String = 'r';
% set colourmap 
colormap(brewermap([],"RdBu"));
% title
sgtitle(sprintf('%s, Averaged Epoch = %g, TW = %g, V. acc > %g',netlab,epoch,tw,acclim),'FontSize',15);

%% plot specific correlations
% key correlations
% regularisation (8) with strength (1) mod (3) corr (4) smw (5) ev choice (10) ms (11)
% efficiency (2) with modularity (3) trade-off
% modularity (3) with ev choice (10) and ev goal (9)
% set predicter and outcome
x = 3;
y = 10;
% get the data
xdata = summary_cat(:,x);
ydata = summary_cat(:,y);
% specific corr
[r p] = corr(xdata,ydata);
% plot
%h = figure; h.Position = [100 100 300 250];
h = figure; h.Position = [100 100 450 275];
%{
h = scatter(xdata,100*ydata,'o',...
     'sizedata',50,...
     'markerfacecolor',[.85 .85 .85],...
     'markeredgecolor',[.5 .5 .5],...
     'markerfacealpha',.6);    
%}
h = scatter(xdata,ydata,'o',...
    'sizedata',30,...
    'markerfacecolor',[.85 .85 .85],...
    'markeredgecolor',[.5 .5 .5],...
    'markerfacealpha',.6);
xlabel(cat_labels(x)); 
ylabel(cat_labels(y));
%title(sprintf('r=%.3g, p=%.3g',r,p));
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 16;
box off;
%% plot specific correlations in a column
% visualise
h = figure; h.Position = [100 100 300 1000];
% strength
subplot(4,1,1);
% set predicters and outcome
x = 1;
y = 1;
% get the data
xdata = summary_cat(:,x);
ydata = summary_cat(:,y);
% plot
h = scatter(xdata,ydata,'o',...
     'sizedata',40,...
     'markerfacecolor',[.85 .85 .85],...
     'markeredgecolor',[.5 .5 .5],...
     'markerfacealpha',.6);    
ylabel(cat_labels(y)); xticks([]);
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 16;
box off;
xticklabels({'0%','50%','100%'});
% corr
subplot(4,1,2);
% set predicters and outcome
x = 8;
y = 4;
% get the data
xdata = summary_cat(:,x);
ydata = summary_cat(:,y);
% plot
h = scatter(xdata,ydata,'o',...
     'sizedata',40,...
     'markerfacecolor',[.85 .85 .85],...
     'markeredgecolor',[.5 .5 .5],...
     'markerfacealpha',.6);    
ylabel(cat_labels(y)); xticks([]);
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 16;
box off
%{
% mod
subplot(6,1,3);
% set predicters and outcome
x = 8;
y = 3;
% get the data
xdata = summary_cat(:,x);
ydata = summary_cat(:,y);
% plot
h = scatter(xdata,ydata,'o',...
     'sizedata',50,...
     'markerfacecolor',[.85 .85 .85],...
     'markeredgecolor',[.5 .5 .5],...
     'markerfacealpha',.6);    
ylabel(cat_labels(y)); xticks([]);
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 16;
box off;
% smw
subplot(6,1,4);
% set predicters and outcome
x = 8;
y = 5;
% get the data
xdata = summary_cat(:,x);
ydata = summary_cat(:,y);
% plot
h = scatter(xdata,ydata,'o',...
     'sizedata',50,...
     'markerfacecolor',[.85 .85 .85],...
     'markeredgecolor',[.5 .5 .5],...
     'markerfacealpha',.6);    
ylabel(cat_labels(y)); xticks([]);
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 16;
box off;
%}
% ms
subplot(4,1,3);
% set predicters and outcome
x = 8;
y = 11;
% get the data
xdata = summary_cat(:,x);
ydata = summary_cat(:,y);
% plot
h = scatter(xdata,ydata,'o',...
     'sizedata',40,...
     'markerfacecolor',[.85 .85 .85],...
     'markeredgecolor',[.5 .5 .5],...
     'markerfacealpha',.6);    
ylabel(cat_labels(y)); xticks([]); 
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 16;
box off;
subplot(4,1,4);
% set predicters and outcome
x = 8;
% remove zeros from summary_comp_goal_choice ***
ind = summary_comp_goal_choice(2,:,2)==0;
y = summary_comp_goal_choice;
y(:,ind,:) = [];
ii = [1 1; 2 2];
% colours
col = flip(brewermap(3,"BrBG"));
col(2,:) = [];
% get the data
for i = 1:2;
xdata = summary_cat(:,x);
ydata = y(ii(i),:,ii(i));
% plot
h = scatter(xdata,ydata,'o',...
     'sizedata',40,...
     'markerfacecolor',col(i,:),...
     'markeredgecolor',[.5 .5 .5],...
     'markerfacealpha',.6);    
ylabel('EV');
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 16;
box off;
xlabel('Regularisation');
xticklabels({'0%','50%','100%'});
hold on;
end
%% plot a specific sets with lines of best fit
% set the specific predictor
x = 8;
% view
yset = [2 3 5];
% set colour palette
j = brewermap(6,"RdBu");
j([2 4 5],:) = [];
% color order
colord = [1 3 2];
r = []; 
h = figure; h.Position = [100 100 450 275];
for i = 1:length(yset);
    y = yset(i);
    % get the data
    xdata = rescale(summary_cat(:,x));
    ydata = rescale(summary_cat(:,y));
    % specific corr
    [r(i) p] = corr(xdata,ydata);
    % plot
    h = scatter(xdata,ydata,'o',...
        'sizedata',30,...
        'markerfacecolor',j(colord(i),:),...
        'markeredgecolor',[.75 .75 .75],...
        'markerfacealpha',.75);
    % xlabel(cat_labels(x)); 
    xlabel('Regularization');
    xticks([]); xticklabels();
    yticks([]); yticklabels([]);
    ylim([0 1]);
    b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 16;
    box off;
    hold on;
    % lob
    % e = polyfit(xdata,ydata,1);
    % f = polyval(e,xdata);
    % plot(xdata,f,'color',j(i,:),'linewidth',4);
end
%% look at explaned variance change over time winodws
% set network to use
network = net_se_swc_4_1k;
% set summary array to use
summary_array = summary_array_swc_4_1k;
% label
lab = 'sWc';
% set settings
epoch = 9;
net = 308;
% visualise
h = figure; h.Position = [100 100 1400 140];
for tw = 1:9;
    subplot(1,9,tw);
    % get index
    idx = summary_array(:,1)==net & summary_array(:,3)==epoch & summary_array(:,4)==tw;
    % get variance explained
    b = summary_array(idx,[8 9]);
    % seperate
    goal = squeeze(b(:,1));
    choice = squeeze(b(:,2));
    % plot
    histogram(goal-choice,'edgecolor','w'); box off; 
    b = gca; b.TickDir = 'out'; 
    b.FontName = 'Arial';
    ylabel('Frequency');
    xlabel('Choice | Goal');
    xlim([-3 3]);
    title(sprintf('tw=%g',tw));
    sgtitle(sprintf('%s, network %g, epoch %g, variance explained',lab,net,epoch));
end
%% explore any systematic spatial allocations of goal versus choices
% this only works when clusters are resonable size based on how I coded it

% set network to use
network = net_se_swc_4_1k;
% set summary array to use
summary_array = summary_array_swc_4_1k;
% set accuracy to use
acc = acc_se_swc_4_1k;
% set label
lab = 'swc_4_1k';

% set settings
% note - the p value for clusters should be smaller at the transitions e.g. 6 relative to 9 because the cluster just establishes 
epoch = 9;
tw = 9;
net = 290; %acc_test(144) % 290 epoch 6 (set 7) tw 6
% get index for specific tw
idx = summary_array(:,1)==net & summary_array(:,3)==epoch & summary_array(:,4)==tw;
% get network
n = squeeze(network(net,epoch+1,:,:));
% get accounacy
acc_pick = squeeze(acc(net,epoch+1,2));

% remove diagnoal and absolute
n(find(eye(size(n,1)))) = 0;
n = abs(n);
% get variance explained
v = summary_array(idx,[8 9 10]);
% seperate
goal = squeeze(v(:,1));
choice = squeeze(v(:,2));

% visualise the variance
h = figure; h.Position = [100 100 1000 220];
subplot(1,3,1); histogram(goal,'edgecolor','w'); 
box off; xlabel('Goal variance (%)'); ylabel('Frequency');
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 12;
subplot(1,3,2); histogram(choice,'edgecolor','w'); 
box off; xlabel('Choice variance (%)'); ylabel('Frequency');
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 12;
subplot(1,3,3); histogram(goal-choice,'edgecolor','w'); 
box off; xlabel('Goal-Choice variance (%)'); ylabel('Frequency');
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 12;
sgtitle(sprintf('network %g epoch %g tw %g',net,epoch,tw)); 

% make a single figure
h = figure; h.Position = [100 100 650 400];
r = plot(graph(n,'upper'),...
    'XData',Coordinates(1,:),...
    'YData',Coordinates(2,:),...
    'ZData',Coordinates(3,:),...
    'edgecolor',[.8 .8 .8],...
    'edgealpha',.08,...
    'nodelabel',[],...
    'markersize',15.*abs(sum(v,2))+1e-6,...
    'edgecolor',[.8 .8 .8]);
box off; xlabel('X'); ylabel('Y'); zlabel('Z');
xticks([]); yticks([]); zticks([]);
b = gca; b.FontSize = 16; b.FontName = 'Arial';

% compute goal versus choice clusters
clua = (abs(goal) - abs(choice)); % make absolute for small variation
clu = [];
clu(clua>0) = 1; % goal
clu(clua<0) = 2; % choice
nclu = [sum(clu==1) sum(clu==2)];

% update the colours
res = 256;
vv = round(rescale(clua).*(res-1)) + 1;
col = brewermap(res,"BrBG");
kk = col(vv,:);
r.NodeColor = kk; 
c = colorbar; 
colormap(col); 
c.Ticks = [0 0.5 1];
c.TickLabels = {'Choice','Neutral','Goal'};
c.TickDirection = 'out';
c.Location = 'eastoutside';
c.Label.String = 'Decoding direction';

%% make color the decoding direction
% plot ev choice in the space
h = figure; h.Position = [100 100 1800 450];
subplot(2,4,[1 5]);
r = plot(graph(n,'upper'),...
    'XData',Coordinates(1,:),...
    'YData',Coordinates(2,:),...
    'ZData',Coordinates(3,:),...
    'edgecolor',[.9 .9 .9],...
    'edgealpha',.05,...
    'nodelabel',[],...
    'markersize',10.*abs(sum(v,2))+1e-6,...
    'edgecolor',[.8 .8 .8]);
box off; xlabel('x'); ylabel('y'); zlabel('z');
xticks([]); yticks([]); zticks([]);
b = gca; b.FontSize = 12; b.FontName = 'Arial';

% compute goal versus choice clusters
clua = (abs(goal) - abs(choice)); % make absolute for small variation
clu = [];
clu(clua>0) = 1; % goal
clu(clua<0) = 2; % choice
nclu = [sum(clu==1) sum(clu==2)];

% update the colours
res = 256;
vv = round(rescale(clua).*(res-1)) + 1;
col = brewermap(res,"BrBG");
kk = col(vv,:);
r.NodeColor = kk; 
c = colorbar; 
colormap(col); 
c.Ticks = [0 0.5 1];
c.TickLabels = {'Choice','Neutral','Goal'};
c.TickDirection = 'out';
c.Location = 'southoutside';
c.Label.String = 'Decoding direction';

% must do a permutation test, given the spatial autocorrelations of the box
% note, the disparity in cluster size confounds the null distributions 

nperm = 1000; % number of permutations
pperm = []; dperm = []; p2perm = []; d2perm = [];

% loop
for perm = 1:nperm;
    % run a permutation for cluster 1 and cluster 2 seperately
    for cluster = 1:2;
        x = randsample([1:length(clu)],sum(clu==cluster)); % get a random smaple of length of the observed cluster size
        %{
        % binary version
        intra_d = triu(Dbox(x,x),1); intra_d = intra_d(intra_d>0); % intra cluster
        pperm(perm,cluster) = mean(intra_d); % compute null
        %}
        % weighted version
        dd = Dbox(x,x);
        cv = abs(clua(clu==cluster));
        pperm(perm,cluster) = mean(dd.*cv,'all');
    end
end

% compute intra versus inter cluster euclideans 
% goal
% binary version
intra_d = triu(Dbox(clu==1,clu==1),1); intra_d = intra_d(intra_d>0);
p = mean(intra_d);
pgoal = sum(p>pperm(:,1))./nperm;

% weighted version
dd = Dbox(clu==1,clu==1);
cv = abs(clua(clu==1));
p = mean(dd.*cv,'all');
pgoal = sum(p>pperm(:,1))./nperm;

% visualise euclidean
subplot(2,4,2);
histogram(intra_d,'edgecolor','w'); 
title('Goal');
b = gca; b.TickDir = 'out'; b.FontSize = 12; b.FontName = 'Arial';
xlabel('Euclidean distance'); ylabel('Frequency'); box off;
% visualise permutation
subplot(2,4,6);
histogram(pperm(:,1),'edgecolor','w'); 
hold on;
xline(p);
title(sprintf('p_{perm} = %.4g | %g neurons',pgoal,nclu(1)));
b = gca; b.TickDir = 'out'; b.FontSize = 12; b.FontName = 'Arial';
xlabel('Permuted distribution'); ylabel('Frequency'); box off;

% choice
% binary version
intrachoice_d = triu(Dbox(clu==2,clu==2),1); intrachoice_d = intrachoice_d(intrachoice_d>0);
p = mean(intrachoice_d);
pchoice = sum(p>pperm(:,2))./nperm;

% weighted version
dd = Dbox(clu==2,clu==2);
cv = abs(clua(clu==2));
p = mean(dd.*cv,'all');
pchoice = sum(p>pperm(:,2))./nperm;

% visualise euclidean
subplot(2,4,3);
histogram(intrachoice_d,'edgecolor','w'); 
title('Choice');
b = gca; b.TickDir = 'out'; b.FontSize = 12; b.FontName = 'Arial';
xlabel('Euclidean distance'); ylabel('Frequency'); box off;
% visualise permutation
subplot(2,4,7);
histogram(pperm(:,2),'edgecolor','w'); 
hold on;
xline(p);
title(sprintf('p_{perm} = %.4g | %g neurons',pchoice,nclu(2)));
b = gca; b.TickDir = 'out'; b.FontSize = 12; b.FontName = 'Arial';
xlabel('Permuted distribution'); ylabel('Frequency'); box off;

% plot the variance explained over neurons
vartw = [];
for twi = 1:9;
    % get index for specific tw
    idx = summary_array(:,1)==net & summary_array(:,3)==epoch & summary_array(:,4)==twi;
    % get variance explained
    vartw(:,twi,:) = summary_array(idx,[8 9]);
end
subplot(2,4,[4 8]);
plot([squeeze(vartw(:,:,1))-squeeze(vartw(:,:,2))]',...
    'color',[.6 .6 .6]);
xline(4,'linewidth',4); xline(6,'linewidth',4); yline(0,'linewidth',4);
xline(tw,'linewidth',1,'color','r');
box off; b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 12;
xlabel('Time window'); ylabel('Goal - Choice');
xlim([1 9]); 

% title
sgtitle(sprintf('Network %g | Epoch %g | Time window %g',net,epoch,tw));

% produce a 2d force-atlas version of the neurons
h = figure; h.Position = [100 100 400 450];
r = plot(graph(Cost_Matrix,'upper'),...
    'edgecolor',[.9 .9 .9],...
    'edgealpha',.05,...
    'nodelabel',[],...
    'markersize',10.*abs(sum(v,2))+1e-6,...
    'edgecolor',[.8 .8 .8]);
box off; xlabel('x'); ylabel('y');
xticks([]); yticks([]); zticks([]);
b = gca; b.FontSize = 12; b.FontName = 'Arial';

% compute goal versus choice clusters
clua = (abs(goal) - abs(choice)); % make absolute for small variation
clu = [];
clu(clua>0) = 1; % goal
clu(clua<0) = 2; % choice
nclu = [sum(clu==1) sum(clu==2)];

% update the colours
res = 256;
vv = round(rescale(clua).*(res-1)) + 1;
col = brewermap(res,"BrBG");
kk = col(vv,:);
r.NodeColor = kk; 
c = colorbar; 
colormap(col); 
c.Ticks = [0 0.5 1];
c.TickLabels = {'Choice','Neutral','Goal'};
c.TickDirection = 'out';
c.Location = 'southoutside';
c.Label.String = 'Decoding direction';
%% compute statistically for all
% set network and summary array
network = net_se_swc_4_1k;
summary_array = summary_array_swc_4_1k;
accuracy = acc_se_swc_4_1k;
% set hyperparameters
nperm = 1000;
ntw = 9;
% set epoch 
epoch = 9;
% size of networks
nnet = 1000;
% initialise
pgoal = nan(nnet,ntw); 
pchoice = nan(nnet,ntw); 
acc = nan(nnet,ntw); 
rall = nan(nnet,ntw);
nclus = nan(nnet,ntw,2);
notrun = [];
step = 1;
% loop
    for tw = 1:9;
        for net = 1:nnet;
            % get index
            idx = summary_array(:,1)==net & summary_array(:,3)==epoch & summary_array(:,4)==tw;
            % get variance explained
            b = summary_array(idx,[8 9]);
            goal = b(:,1);
            choice = b(:,2);
            % seperate
            clua = (abs(goal) - abs(choice)); % make absolute for small variation
            clu = [];
            clu(clua>0) = 1;
            clu(clua<0) = 2;
            nclu = [sum(clu==1) sum(clu==2)];
             % keep correlations
            rall(net,step) = corr(goal,choice);
            % keep number of clusters
            nclus(net,step,:) = nclu;
            % keep accuracy
            acc(net,step) = mean(summary_array(idx,6));
            % permutation test
            pperm = nan(nperm,2);
            % loop
            for perm = 1:nperm;
                for cluster = 1:2;
                    x = randsample([1:length(clu)],nclu(cluster)); % form permutation group 1
                    %{
                    % binary version
                    intra_d = triu(Dbox(x,x),1); intra_d = intra_d(intra_d>0); % goal
                    pperm(perm,cluster) = mean(intra_d); % compute null
                    %}
                    % weighted version
                    dd = Dbox(x,x);
                    cv = abs(clua(clu==cluster));
                    pperm(perm,cluster) = mean(dd.*cv,'all');
                end
            end
            % goal
            %{
            % binary version
            intragoal_d = triu(Dbox(clu==1,clu==1),1); intragoal_d = intragoal_d(intragoal_d>0);
            p = mean(intragoal_d);
            pgoal(net,step) = sum(p>pperm(:,1))./nperm;
            %}
            % weighted version
            dd = Dbox(clu==1,clu==1);
            cv = abs(clua(clu==1));
            p = mean(dd.*cv,'all');
            pgoal(net,step) = sum(p>pperm(:,1))./nperm;
            % choice
            % binary version
            %{
            intrachoice_d = triu(Dbox(clu==2,clu==2),1); intrachoice_d = intrachoice_d(intrachoice_d>0);
            p = mean(intrachoice_d);
            pchoice(net,step) = sum(p>pperm(:,2))./nperm;
            %}
            dd = Dbox(clu==2,clu==2);
            cv = abs(clua(clu==2));
            p = mean(dd.*cv,'all');
            pchoice(net,step) = sum(p>pperm(:,2))./nperm;
            disp(sprintf('tw%g, network %g complete',tw,net));
        end
        step = step + 1;
        disp(sprintf('tw %g complete',tw));
    end
    % save network
    se_swc_4_1k_all_embedding_epoch9 = struct;
    se_swc_4_1k_all_embedding_epoch9.pgoal = pgoal;
    se_swc_4_1k_all_embedding_epoch9.pchoice = pchoice;
    se_swc_4_1k_all_embedding_epoch9.nclus = nclus;
    se_swc_4_1k_all_embedding_epoch9.rall = rall;
    se_swc_4_1k_all_embedding_epoch9.accuracy = acc;
    se_swc_4_1k_all_embedding_epoch9.epoch = epoch;
    se_swc_4_1k_all_embedding_epoch9.nperm = nperm;
    se_swc_4_1k_all_embedding_epoch9.notrun = notrun;
    save('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_4_1k_all_embedding_epoch9.mat','se_swc_4_1k_all_embedding_epoch9','-v7.3');
%% form the aternative single variable clustering test
% set network to use
network = net_se_swc_4_1k;
% set summary array to use
summary_array = summary_array_swc_4_1k;
% set accuracy to use
acc = acc_se_swc_4_1k;
% set label
lab = 'l1_3_1k';
% set settings
% note - the p value for clusters should be smaller at the transitions e.g. 6 relative to 9 because the cluster just establishes 
epoch = 9;
tw = 6;
net = 290; %acc_test(144) % 290 epoch 6 (set 7) tw 6
% get index for specific tw
idx = summary_array(:,1)==net & summary_array(:,3)==epoch & summary_array(:,4)==tw;
% get network
n = squeeze(network(net,epoch+1,:,:));
% get accounacy
acc_pick = squeeze(acc(net,epoch+1,2));
% remove diagnoal and absolute
n(find(eye(size(n,1)))) = 0;
n = abs(n);
% get variance explained
v = summary_array(idx,[8 9 10]);
% seperate
goal = squeeze(v(:,1));
choice = squeeze(v(:,2));
cchoice = squeeze(v(:,3));
% set the cut-off variable nodes 
thr = 0.5;
% compute number of nodes (same for all)
nnode = thr*100;
% loop over observed variables
variance = [goal choice cchoice];
owe = [];
for j = 1:size(variance,2);
    % variance
    a = variance(:,j);
    [u in] = sort(a);
    x = in(nnode+1:end); 
    i = sort(x);
    % get their euclidean distances
    do = Dbox(i,i);
    % observed euclidean
    oe(j) = mean(do,'all');
end
oe_perm = [];
nperm = 1000;
% loop over permutations
for perm = 1:nperm;
    % take a random sample
    x = randsample([1:100],nnode); % get a random smaple of length of the observed cluster size
    % get their euclidean distances
    do = Dbox(x,x);
    % observed weighted euclidean
    oe_perm(perm) = mean(do,'all');
end
% histogram
histogram(oe_perm); hold on; xline(oe(1),'color','r'); xline(oe(2),'color','g'); xline(oe(3),'color','k');
%% do binary one-sided tests for all
% set network and summary array
network = net_l1_3_1k;
summary_array = summary_array_l1_3_1k;
accuracy = acc_l1_3_1k;
% set hyperparameters
tws = [3 6 9];
nperm = 1000;
epoch = 9;
nnet = 1000;
% set the top node thresholds to evaluate 
nnode = [50 25 10];
% number of permutations
nperm = 1000;

% initialise
oe_perm = nan(nperm,length(nnode));
pperm = nan(nnet,length(tws),length(nnode),3);

% loop over permutations
for nn = 1:length(nnode);
    for perm = 1:nperm;
        % take a random sample
        x = randsample([1:100],nnode(nn)); % get a random smaple of length of the observed cluster size
        % get their euclidean distances
        do = Dbox(x,x);
        % observed weighted euclidean
        oe_perm(perm,nn) = mean(do,'all');
    end
end

% loop over networks
for net = 1:nnet;

% loop over tws
for tw = 1:length(tws);
    % loop over number of nodes
    for nn = 1:length(nnode);
        % get index
        idx = summary_array(:,1)==net & summary_array(:,3)==epoch & summary_array(:,4)==tws(tw);
        % get variance explained
        b = summary_array(idx,[8 9 10]);
        goal = b(:,1);
        choice = b(:,2);
        cchoice = b(:,3);
        variance = [goal choice cchoice];
        % compute observed
        oe = [];
            for j = 1:size(variance,2);
                % variance
                a = variance(:,j);
                [u in] = sort(a);
                b = 101-nnode(nn);
                x = in(b:end); 
                i = sort(x);
                % get their euclidean distances
                do = Dbox(i,i);
                % observed euclidean
                oe(j) = mean(do,'all');
                % compare to permutation
                pperm(net,tw,nn,j) = (sum(oe(j)>oe_perm(:,nn)))./nperm;
            end
            disp(sprintf('net %g time window %g number of nodes %g',net,tws(tw),nnode(nn)));
    end
end

end
pperm_binary_l1_3 = struct;
pperm_binary_l1_3.pperm = pperm;
pperm_binary_l1_3.epoch = epoch;
pperm_binary_l1_3.tws = tws;
pperm_binary_l1_3.nnodes = nnode;
pperm_binary_l1_3.nperm = nperm; 
pperm_binary_l1_3.details = 'nnet (1000) x tw (3,6,9) x nnodes (top 50%, top 25%, top 10%) x variable (goal,choice,cchoice)';
save('/imaging/astle/users/da04/PhD/ann_jascha/data/pperm_binary_l1_3.mat','pperm_binary_l1_3','-v7.3');
%% visualise
% load 
load('/imaging/astle/users/da04/PhD/ann_jascha/data/pperm_binary_se_swc_4.mat');
% take data
pperm = pperm_binary_l1_3.pperm;
% set threshold
threshold = 1;
% loop over
step = 1;
h = figure; h.Position = [100 100 650 500];
for variable = 1:3;
    for tw = 1:3;
        data = squeeze(pperm(:,tw,threshold,variable));
        subplot(3,3,step);
        histogram(data,'edgecolor','w');
        box off;
        b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 12;
        ylabel('Frequency'); xlabel('p_{perm}');
        step = step + 1;
    end
end
%% visualise spatial structure-function statistics

%{
must do some filtering!
%}

label = 'sWC_4_1k';
% plot
h = figure; h.Position = [100 100 1000 1000];
subplot(2,2,1); 
imagesc(pgoal); 
xlabel('Time window'); ylabel('Network');
c = colorbar; c.Label.String = 'P';  c.TickDirection = 'out'; title('Goal');
colormap(flip(pink));
b = gca; 
b.TickDir = 'out'; 
b.TickLength = [0 0]; 
b.FontName = 'Arial';
b.FontSize = 18;
box off;
xticks([1:4]); 
xticklabels({'6','7','8','9'});
subplot(2,2,2); 
imagesc(pchoice);
xlabel('Time window'); ylabel('Network');
c = colorbar; c.Label.String = 'P'; c.TickDirection = 'out'; title('Choice'); 
colormap(flip(pink)); 
b = gca; 
b.TickDir = 'out';
b.TickLength = [0 0]; 
b.FontName = 'Arial';
b.FontSize = 18;
box off; 
xticks([1:4]); 
xticklabels({'6','7','8','9'});
sgtitle(sprintf('Epoch=%g',epoch));
subplot(2,2,3); 
histogram(pgoal,'edgecolor','w');
b = gca; b.TickDir = 'out'; box off; b.FontSize = 18;
xlabel('P value'); ylabel('Frequency');
subplot(2,2,4); 
histogram(pchoice,'edgecolor','w');
b = gca; b.TickDir = 'out'; box off; b.FontSize = 18;
xlabel('P value'); ylabel('Frequency');
sgtitle(sprintf('%s | Epoch=%g | nperms=%g',label,epoch,nperm));
% reform the colormap
nstep = 500;
lin = linspace(0,1,nstep)'; 
c = ones(nstep,3); c(:,1)=lin; c(:,2)=lin;
d = ones(nstep,3); d(:,2)=lin; d(:,3)=lin;
col = [d;flip(c)];

% plot correlations
h = figure; h.Position = [100 100 500 300];
imagesc(rall); 
b = gca; b.TickLength = [0 0]; b.FontSize = 12; b.FontName = 'Arial';
xticks([1:4]); xticklabels([6:9]); c = colorbar; caxis([-1 1]); c.Label.String = 'corr(goal,choice)'; colormap(col);
sgtitle(sprintf('%s | Epoch=%g | nperms=%g',label,epoch,nperm));

% plot clusters
h = figure; h.Position = [100 100 900 300];
subplot(1,2,1); 
imagesc(nclus(:,1:4)); 
b = gca; b.TickLength = [0 0]; b.FontSize = 12; b.FontName = 'Arial';
xticklabels([6:9]); c = colorbar; caxis([0 100]); c.Label.String = 'Number of goal clusters'; colormap(flip(pink));
subplot(1,2,2);
imagesc(nclus(:,5:8)); 
b = gca; b.TickLength = [0 0]; b.FontSize = 12; b.FontName = 'Arial';
xticklabels([6:9]); c = colorbar; caxis([0 100]); c.Label.String = 'Number of choice clusters'; colormap(flip(pink));
sgtitle(sprintf('%s | Epoch=%g | nperms=%g',label,epoch,nperm));

%% compare adaptive codings
% take summary arrays
summary_array_group = {summary_array_swc_4_1k summary_array_l1_3_1k};
% time windows
tws = [1:9];
% set epoch
epoch = 9;
% set acc thr
accthr = 0.9;
% initialise
goal = nan(2,9,1000,100);
choice = nan(2,9,1000,100);
cchoice = nan(2,9,1000,100);
% loop over networks and compute the goal and choice distributions
for group = 1:2;
    summary_array = summary_array_group{group};
    for tw = 1:length(tws);
        step = 1;
        for net = 1:1000;
        % get index
        idx = summary_array(:,1)==net & summary_array(:,3)==epoch & summary_array(:,4)==tw & summary_array(:,6)>accthr;
            if sum(idx)>1;
            % get goal and choice distributions and mean
            goal(group,tw,step,:) = summary_array(idx,8);
            choice(group,tw,step,:) = summary_array(idx,9);
            cchoice(group,tw,step,:) = summary_array(idx,10);
            disp(sprintf('group %g net %g tw %g done',group,net,tw));
            end
            step = step + 1;
        end
    end
end
% save
adaptive_coding_epo9_tw1to9 = struct;
adaptive_coding_epo9_tw1to9.goal = goal;
adaptive_coding_epo9_tw1to9.choice = choice;
adaptive_coding_epo9_tw1to9.correct_choice = cchoice;
save('/imaging/astle/users/da04/PhD/ann_jascha/data/adaptive_coding_epo9_tw1to9.mat','adaptive_coding_epo9_tw1to9');
%% plot distribution
load('/imaging/astle/users/da04/PhD/ann_jascha/data/adaptive_coding_epo9_tw1to9.mat');
goal = adaptive_coding_epo9_tw1to9.goal;
choice = adaptive_coding_epo9_tw1to9.choice;
cchoice = adaptive_coding_epo9_tw1to9.correct_choice;
% form palettes
pal = [0,176,178;242,98,121]./256;
% take average across units
goalav = squeeze(mean(goal,4));
choiceav = squeeze(mean(choice,4));
% take goal information
a = squeeze(goalav(1,:,:));
b = squeeze(goalav(2,:,:));
% take choice information
c = squeeze(choiceav(1,:,:));
d = squeeze(choiceav(2,:,:));
% plot difference
e = squeeze(a-c);
f = squeeze(b-d);
h = figure; h.Position = [100 100 300 175];
stdshade(e',.5,pal(2,:)); % se swc
hold on;
stdshade(f',.5,pal(1,:)); % l1
box off;
b = gca; b.TickDir = 'out'; xlabel('Time window'); ylabel('EV goal - choice');
b.FontName = 'Arial'; b.FontSize = 14;
yline(0);
xlim([1 9]);
% plot histograms
group = 1;
adiff = squeeze(goal(group,3,:,:)) - squeeze(choice(group,3,:,:));
bdiff = squeeze(goal(group,6,:,:)) - squeeze(choice(group,6,:,:));
cdiff = squeeze(goal(group,9,:,:)) - squeeze(choice(group,9,:,:));
cmap = flip(brewermap(5,"BrBG"));
h = figure; h.Position = [100 100 300 175];
histogram(adiff,'edgecolor',[.1 .1 .1],'facecolor',cmap(2,:)); 
hold on; 
histogram(bdiff,'edgecolor',[.1 .1 .1],'facecolor',cmap(3,:)); 
hold on; 
histogram(cdiff,'edgecolor',[.1 .1 .1],'facecolor',cmap(4,:));
xlabel('EV goal - choice'); ylabel('Frequency');
xlim([-3 3]); ylim([0 4000]);
box off;
b = gca; b.TickDir = 'out'; b.FontSize = 14; b.FontName = 'Arial';
% plot specifically for certain networks
% set epoch and statistic
epo = 9; stat = 5; 
% take global statistics
data = squeeze(global_statistics(:,epo+1,stat));
% threshold by accuracy
indthr = acc_se_swc_4_1k(2:end,epo+1,2)>=0.9;
% get filtered data
findthr = find(indthr);
fil = data; fil(~indthr) = [];
% sort the networks from low to high
[d,s] = sort(fil);
% take distributions in windows
nets = {[1:129],[131:259],[260:388]}; % ensure same size
cmap = flip(brewermap(5,"BrBG"));
% initialise
statkeep = [];
% visualise
h = figure; h.Position = [100 100 800 250];
for i = 1:length(nets);
    % get networks
    nn = s(nets{i});
    % subplot
    subplot(1,length(nets),i);
    % form histograms
    adiff = squeeze(goal(1,3,nn,:)) - squeeze(choice(1,3,nn,:));
    bdiff = squeeze(goal(1,6,nn,:)) - squeeze(choice(1,6,nn,:));
    cdiff = squeeze(goal(1,9,nn,:)) - squeeze(choice(1,9,nn,:));
    % plot
    histogram(adiff,'edgecolor',[.1 .1 .1],'facecolor',cmap(2,:)); 
    hold on; 
    histogram(bdiff,'edgecolor',[.1 .1 .1],'facecolor',cmap(3,:)); 
    hold on; 
    histogram(cdiff,'edgecolor',[.1 .1 .1],'facecolor',cmap(4,:));
    xlabel('EV goal- choice'); ylabel('Frequency');
    box off;
    b = gca; b.TickDir = 'out'; b.FontSize = 14; b.FontName = 'Arial';
    xlim([-2.5 2.5]);
    ylim([0 1800]);
end
%% relationship between spatial-function axis and network statistic

% goal embedding = greater smw (8), greater modularity (5), sparser strength (1) for the goal

% set parametes
stat = 8;

% get data
x = pgoal(:,1); % as you move across the time windows, you see a flattening -> this means that as the task goes on, the representation disperses
y = squeeze(se_swc_4_1k_global_statistics(acc_test,epoch,stat));
%y = rall(:,4);

% visualise
figure;
scatter(x,y,50,'.');
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 18; box off;
[r p] = corr(x,y); xline(0.05);
ylabel(global_label(stat)); 
%ylabel('Mixed selectivity, r');
xlabel('Embedding p_{perm} GOAL (TW 6)');
title(sprintf('r=%.3g, p=%.3g',r,p));

%% look at profiles of explained variances - regularisation changes the trade-off temporally
% set epoch
epoch = 6;
% set summary array
summary_array = summary_array_swc_4_1k;

% take mean variances across regulisations
y = []; acc = [];
for net = 1:100;
    ind = summary_array(:,1)==net & summary_array(:,3)==epoch;
    y(net,1) = mean(summary_array(ind,8));
    y(net,2) = mean(summary_array(ind,9));
    acc(net) = mean(summary_array(ind,6));
end
% the middle group have equal for both, because they do not "shift" from
% goal to choice instead they forget the goal then are choice ****** 
x = [1:100 1:100]';
yz = y(:);
figure; 
scatterhist(x,yz,...
    'group',[ones(100,1);2*ones(100,1)],...
    'kernel','on',...
    'direction','out',...
    'linestyle','-',...
    'marker','.',...
    'markersize',20); 
xlabel('Regularisation'); ylabel('Mean explained variance (%)'); 
legend({'Goal','Choice'},'box','off','location','northeast'); box off;
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 18;

% plot accuracy as colour
col = jet(256); i = round(255*rescale(acc))+1; col = col(i,:);
figure; scatter(x,yz,60,[col;col],'marker','.');
b = gca; b.TickDir = 'out'; box off; 
b.FontName = 'Arial'; b.FontSize = 20;
xlabel('Regularisation'); ylabel('Variance explained');
c = colorbar; c.Label.String = 'Accuracy (%)'; colormap(jet);

% set net
net = 52;
% get index
for tw = 1:9;
    idx = summary_array(:,1)==net & summary_array(:,3)==epoch & summary_array(:,4)==tw;;
    goal(:,tw) = summary_array(idx,8);
    choice(:,tw) = summary_array(idx,9);
end

% plot
figure;
plot([goal-choice]','color',[.5 .5 .5],'linewidth',1); xlim([1 9]);
hold on; yline(0); hold on; xline(4); hold on; xline(6);
xlabel('Time window'); ylabel('Goal - Choice explained variance');
b = gca; b.TickDir = 'out'; b.FontSize = 15; b.FontName = 'Arial'; box off; 
