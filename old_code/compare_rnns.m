%% compare topology of network 
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
% addpath of cohen's d
addpath('/imaging/astle/users/da04/PhD/toolboxes/computeCohen');
% addpath of stdshade
addpath('/imaging/astle/users/da04/PhD/toolboxes/stdshade');
% addpath of python colours
addpath('/imaging/astle/users/da04/PhD/toolboxes/Colormaps/Colormaps (5)/Colormaps/');
% addpath of sort files
addpath('/imaging/astle/users/da04/PhD/toolboxes/sort_nat');
% load global statistics
load('/imaging/astle/users/da04/PhD/ann_jascha/data/l1_3_global_statistics.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/l1_3_1k_global_statistics_extra.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_1_global_statistics.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_2_global_statistics.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/swc_1_global_statistics_extra.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_4_global_statistics.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_4_1k_global_statistics_extra.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_4_1k_global_statistics_extra_rescaled.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_4r_global_statistics.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_4h_global_statistics_extra.mat');
% load embedding
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_4_1k_embedding_epoch6.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/l1_3_1k_embedding_epoch6.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_4_1k_all_embedding_epoch9.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/l1_3_1k_all_embedding_epoch9.mat');
% load generative modeling findings
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_4_1k_290_generative.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/l1_3_1k_290_generative.mat');
% load summary arrays
summary_array_swc_4_1k = [];
summary_array_l1_3_1k = [];
str = {'0_101','101_202','202_303','303_404','404_505','505_606','606_707','707_808','808_909','909_1001'};
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
% set directories of the data
project = '/imaging/shared/users/ja02/CBUActors/SpatialRNNs/Results/';
% go to the project
cd(project);
% set paths
l1 = '/mazeGenI_L1_mtI';
l1_2 = '/mazeGenI_L1_mtII';
l1_2r = '/mazeGenI_L1_mtII_randI';
l1_3 = '/mazeGenI_L1_mtIII';
l1_3_1k = '/mazeGenI_L1_mtIII_1k';
se_1 = '/mazeGenI_SE1_mtI';
se_2 = '/mazeGenI_SE1_mtII';
swc_1 = '/mazeGenI_SE1_sWcExcl_mtI';
se_sc_1  = '/mazeGenI_SE1_sc_mtI';
se_swc_1 = '/mazeGenI_SE1_sWc_mtI';
se_swc_2 = '/mazeGenI_SE1_sWc_mtII';
se_swc_2r = '/mazeGenI_SE1_sWc_mtII_randI';
se_swc_3 = '/mazeGenI_SE1_sWc_mtIII';
se_swc_4 = '/mazeGenI_SE1_sWc_mtIV';
se_swc_4_1k = '/mazeGenI_SE1_sWc_mtIV_1k';
se_swc_4r = '/mazeGenI_SE1_sWc_mtIV_randI';
se_swc_4h = '/mazeGenII_SE1_sWc_mtI';
nd = '/NetworkDepot';
rb = 'RegulariserBaselineI';
mn_2 = '/sMNIST_RNNII';
sn_1 = '/SupervisedNetworksI';
sn_2 = '/SupervisedNetworksII';
sn_3 = '/SupervisedNetworksIII';
sn_4 = '/SupervisedNetworksIV';
sn_5 = '/SupervisedNetworksV';
sn_6 = '/SupervisedNetworksVI';
% set labels
network_labels = string({'L1','S','S+C','S+C (R)'});
% set number of nodes
nnode = 100;
% set time
ntime = 11;
% set number of groups we consider
ngroups = 2;
%% load main recurrent neural networks

%%% load l1_3_1k %%%
% change directory
cd(strcat(project,l1_3_1k));
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
parameters_l1_3_1k = [1:1001;linspace(0.001,0.3,1001)]';
% initialise
net_l1_3_1k = zeros(nnet,tt,nnode,nnode);
wei_l1_3_1k = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_l1_3_1k = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_l1_3_1k(i,t,:,:) = Training_History{t+1,1};
        wei_l1_3_1k(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_l1_3_1k(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('l1_3_1k network %g loaded',i-1));
end

%%% load se_swc_4_1k %%%
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

%% load baseline rnns

% ;1 normal but 100!
%%% load l1_3 %%%
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
reorder = [1 13 24 35 46 57 68 79 90 101 3:12 14:23 25:34 36:45 47:56 58:67 69:78 80:89 91:100 2];
% reorder the networks
data = data(reorder);
% set parameters
parameters_l1_3 = [1:101;linspace(0.001,0.3,101)]';
% initialise
net_l1_3 = zeros(nnet,tt,nnode,nnode);
wei_l1_3 = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_l1_3 = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_l1_3(i,t,:,:) = Training_History{t+1,1};
        wei_l1_3(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_l1_3(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('l1_3 network %g loaded',i-1));
end


% se only!
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
% set parameters
parameters_se_2 = [1:101;linspace(0.00001,0.02,101)]';
% initialise
net_se_2 = zeros(nnet,tt,nnode,nnode);
wei_se_2 = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_se_2 = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_se_2(i,t,:,:) = Training_History{t+1,1};
        wei_se_2(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_se_2(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('se_2 network %g loaded',i-1));
end

% swc only!
%%% load swc_1 networks %%%
% change directory
cd(strcat(project,swc_1));
% set the number of networks in here
nnet = 101;
% set how many timepoints
tt = 11;
% list the directory files
k = dir('*.mat'); 
data      = string({k.name})';
% reorder the networks
data = data(reorder);
% set parameters
parameters_swc_1 = [1:101;linspace(0.00001,0.02,101)]';
% initialise
net_swc_1 = zeros(nnet,tt,nnode,nnode);
wei_swc_1 = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_swc_1 = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_swc_1(i,t,:,:) = Training_History{t+1,1};
        wei_swc_1(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_swc_1(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('swc_1 network %g loaded',i-1));
end

% load se_swc_ no task!
%%% load se_swc_4r %%%
% change directory
cd(strcat(project,se_swc_4r));
% set the number of networks in here
nnet = 101;
% set how many timepoints
tt = 11;
% list the directory files
k = dir('*.mat'); 
data      = string({k.name})';
% reorder the networks
data = data(reorder);
% set parameters
parameters_se_swc_4r = [1:101;linspace(0.001,0.3,101)]';
% initialise
net_se_swc_4r = zeros(nnet,tt,nnode,nnode);
wei_se_swc_4r = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_se_swc_4r = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_se_swc_4r(i,t,:,:) = Training_History{t+1,1};
        wei_se_swc_4r(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_se_swc_4r(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('se_swc_4r network %g loaded',i-1));
end

% se_swc normal but 100!
%%% load se_swc_4 %%%
% change directory
cd(strcat(project,se_swc_4));
% set the number of networks in here
nnet = 101;
% set how many timepoints
tt = 11;
% list the directory files
k = dir('*.mat'); 
data      = string({k.name})';
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

% se_swc harder task trials!

%%% load se_swc_4h networks %%% hard task
% change directory
cd(strcat(project,se_swc_4h));
% set the number of networks in here
nnet = 101;
% set how many timepoints
tt = 11;
% list the directory files
k = dir('*.mat'); 
data      = string({k.name})';
% reorder the networks
data = data(reorder);
% set parameters
parameters_se_swc_4h = [1:101;linspace(0.001,0.3,101)]';
% initialise
net_se_swc_4h = zeros(nnet,tt,nnode,nnode);
wei_se_swc_4h = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_se_swc_4h = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_se_swc_4h(i,t,:,:) = Training_History{t+1,1};
        wei_se_swc_4h(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_se_swc_4h(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('se_swc_4h network %g loaded',i-1));
end

%% load prior HD-MEA and CALM geneartive models
% rodent
load('/imaging/astle/users/da04/PhD/hd_gnm_generative_models/data/rodent_50k_div14_10ms_generative_models.mat');
% calm
load('/imaging/astle/users/da04/PhD/ann_jascha/data/external/calm/generative10000_energy.mat');
%% visualise prior data
% form calm matrices
calm = min(generative10000_energy,[],3);
calm_rules = nan(270*5,4);
a = calm(:,1);
b = calm(:,2:3); b = b(:);
c = calm(:,4:8); c = c(:);
d = calm(:,9:end); d = d(:);
calm_rules(1:270,1) = a;
calm_rules(1:2*270,2) = b;
calm_rules(:,3) = c;
calm_rules(:,4) = d;
% form MEA matrices (DIV14)
mea = squeeze(rodent_50k_div14_10ms_generative_models.top_energy_mean(1,:,:))';
mea_rules = nan(6*5,4);
a = mea(:,1);
b = mea(:,2:3); b = b(:);
c = mea(:,4:8); c = c(:);
d = mea(:,9:end); d = d(:);
mea_rules(1:6,1) = a;
mea_rules(1:2*6,2) = b;
mea_rules(:,3) = c;
mea_rules(:,4) = d;
% set xticks
xticklabelsrules = {'Spatial','Homophily','Clustering','Degree'};
% visualise calm
u = figure; u.Position = [100 100 800 400]; 
subplot(1,2,1);
h = iosr.statistics.boxPlot(calm_rules,...
    'theme','colorall',...
    'symbolMarker','x',...
    'symbolColor',[.5 .5 .5]);
% set colors
% palette = [255 255 0; 255 51 51; 1 255 1; 1 1 255]./256; % old colours
vu = viridis;
palette = [vu(255,:); 1 0.1992 0.1992; vu(180,:); vu(50,:)];
for i = 1:4;
    h.handles.box(i).FaceColor = palette(i,:);
end
box off; 
b = gca; b.TickDir = 'out'; 
b.FontName = 'Arial'; b.FontSize = 24;
ylabel('Energy');
ylim([0 .7]);
xticklabels(xticklabelsrules); xtickangle(45);
% visualise mean
subplot(1,2,2);
h = iosr.statistics.boxPlot(mea_rules,...
    'theme','colorall',...
    'symbolMarker','x',...
    'symbolColor',[.5 .5 .5]);
% set colors
for i = 1:4;
    h.handles.box(i).FaceColor = palette(i,:);
end
box off; 
b = gca; b.TickDir = 'out'; 
b.FontName = 'Arial'; b.FontSize = 24;
ylabel('Energy');
ylim([0 .7]);
xticklabels(xticklabelsrules); xtickangle(45);
%% load generative model data
% set directories
se_swc_4_gen = '/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_iv_generative_models';
l1_3_gen = '/imaging/astle/users/da04/PhD/ann_jascha/data/l1_iii_generative_models';
% generative model string
genmodels = string({'sptl','neighbours','matching',...
    'c-avg','clu-min','clu-max','clu-diff','clu-prod',...
    'deg-avg','deg-min','deg-max','deg-diff','deg-prod'});
% set parameters
thr = 0.05;
epo = 9;
netrange = [100 600];
allnets = [100:600];
nnet = netrange(2)-netrange(1);
% initialise
se_swc_energy = []; 
se_swc_parameters = [];
se_swc_networks = {};
% update steps
step = 1;
% change directories
cd(se_swc_4_gen);
% loop 
for net = netrange(1):netrange(2);
    % swc
    str = sprintf('se_swc_4_1k_thr%g_net%g_epo%g_generative_model.mat',thr,net,epo);
    load(str);
    se_swc_energy(step,:,:) = output.energy;
    se_swc_parameters(step,:,:,:) = output.parameters;
    se_swc_networks{step} = output.networks;
    % step
    step = step + 1;
    % display
    disp(sprintf('se_swc networks %g loaded',net));
end
% initialise
l1_energy = []; 
l1_parameters = [];
l1_networks = {};
% update steps
step = 1;
% change directories
cd(l1_3_gen);
% loop 
for net = netrange(1):netrange(2);
    % change directories
    cd(l1_3_gen);
    % swc
    str = sprintf('l1_3_1k_thr%g_net%g_epo%g_generative_model.mat',thr,net,epo);
    load(str);
    l1_energy(step,:,:) = output.energy;
    l1_parameters(step,:,:,:) = output.parameters;
    l1_networks{step} = output.networks;
    % step
    step = step + 1;
    % display
    disp(sprintf('l1 networks %g loaded',net));
end
%% compute generative models
% set energy, accuracy data and indices e.g 100 to 400 (1:300) to remove
energydata = se_swc_energy;
parametersgen = se_swc_parameters;
accuracydata = acc_se_swc_4_1k;
parametersdata = parameters_se_swc_4_1k;
rem = [];
% take parameters and network
m = []; ind = [];
for net = 1:length(allnets);
    data = squeeze(energydata(net,:,:))';
    [m(net,:) ind(net,:)] = min(data);
end
% get accuracy
accuracy_use = squeeze(accuracydata(allnets,epo+1,2));
% use limits
m(rem,:) = [];
indb=ind; indb(rem,:) = [];
parametersdata = parametersdata(allnets,2); 
parametersdata(rem) = [];
network_ind = allnets; 
network_ind(rem)=[]; 
accuracy_use(rem) = [];
% filter by accuracy
accthr = 0.9;
m(accuracy_use<accthr,:) = [];
indb(accuracy_use<accthr,:) = [];
network_ind(accuracy_use<accthr) = [];
parametersdata(accuracy_use<accthr) = [];
% visualise
u = figure; u.Position = [100 100 800 500]; 
h = iosr.statistics.boxPlot(m,...
    'theme','colorall',...
    'symbolMarker','x',...
    'symbolColor',[.5 .5 .5]);
% set colors
% palette = [255 255 0; 255 51 51; 1 255 1; 1 1 255]./256;
palettei = [1 2 2 3 3 3 3 3 4 4 4 4 4];
for i = 1:13;
    h.handles.box(i).FaceColor = palette(palettei(i),:);
end
box off; 
b = gca; b.TickDir = 'out'; 
b.FontName = 'Arial'; b.FontSize = 24;
ylabel('Energy'); xlabel('Generative model');
ylim([0 .9]);
% visualise over regularisation
model = 3;
h = figure; h.Position = [100 100 350 250];
reg = 1:size(m,1);
[r p] = corr(reg',m(:,model));
u = scatter(parametersdata,m(:,model),100,'o',...
    'markerfacecolor',[255 51 51]./256,...
    'markeredgecolor','w',...
    'markerfacealpha',.5);
ylabel('Energy'); xlabel('Regularisation');
ylim([0.05 0.38]);
disp(sprintf('r=%.3g, p=%.3g',r,p));
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 18;
% visualsie over regualrisation different settings
h = figure; h.Position = [100 100 300 250];
h = scatter(parametersdata,m(:,model),'o',...
     'sizedata',50,...
     'markerfacecolor',[.85 .85 .85],...
     'markeredgecolor',[.5 .5 .5],...
     'markerfacealpha',.6);    
xlabel('Reg'); ylabel('Homophily energy');
ylim([0.05 0.38]);
title('');
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 16;
box off;
% plot the parameter space of a representative network
net = 350;
model = 3;
x = squeeze(parametersgen(net,model,:,:));
y = squeeze(energydata(net,model,:));
figure; 
scatter(x(:,1),x(:,2),30,y);
% correlation matrix
reg = 1:size(m,1);
data = [reg' m];
[r p] = corr(data);
h = figure; h.Position = [100 100 700 600];
imagesc(r); caxis([-1 1]);
set(gca,'XTick',[1:14],'YTick',[1:14]);
xticklabels(['Regularisation' genmodels]);
yticklabels({'Reg',1:13});
xtickangle(45);
colormap(magma); c = colorbar; c.Label.String = 'r';
b = gca; b.TickLength = [0 0]; b.FontName = 'Arial'; b.FontSize = 18;
% plot correlation with reg
h = figure; h.Position = [100 100 450 250];
u = bar(r(2:end,1),'facealpha',.5,'edgecolor',[.8 .8 .8]); 
u.FaceColor = 'flat';
for i = 1:13;
    u.CData(i,:) = palette(palettei(i),:);
end
xlabel('Generative model');
ylabel('r');
ylim([-1 1]);
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 18;
box off;
hold on;
xline(0,'color','w');
% anova
[p anovatab stats] = anova1(m);
cmp = multcompare(stats);
% visualise
model = 3;
i = double(cmp(:,1)==model) + double(cmp(:,2)==model);
figure; bar(cmp(find(i),6));
jj = genmodels; jj(model)=[];
xticklabels(jj);
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 18; xtickangle(45);
ylabel('p value');
% statistically directly compare energy between se_swc and l1
a = se_swc_energy;
b = l1_energy;
model = 3;
x = min(squeeze(a(:,model,:)),[],2);
y = min(squeeze(b(:,model,:)),[],2);
[h p] = ttest(x,y);
d = computeCohen_d(x,y);
%% compute the topological dissimilarity
%{
% binarise the observed networks as there were previous
threshold = 0.05;
epo = 9;
% initialise
Abin = [];
for net = 1:1001;
    a = abs(squeeze(net_l1_3_1k(net,epo+1,:,:))); % check!
    b = threshold_proportional(a,threshold);
    c = b + b';
    Abin(net,:,:) = double(c>0);
    % display
    disp(sprintf('network %g thresholded at %g',net-1,threshold));
end
% loop over networks
for net = 1:length(allnets);
    % loop over models
    % get observed
    Aobs = squeeze(Abin(99+net,:,:));
    % compute measures
    y = [];
    y(:,1) = degrees_und(Aobs);
    y(:,2) = clustering_coef_bu(Aobs);
    y(:,3) = betweenness_bin(Aobs)';
    y(:,4) = sum(Cost_Matrix .* Aobs)';
    y(:,5) = efficiency_bin(Aobs,1);
    y(:,6) = sum(matching_ind(Aobs) + matching_ind(Aobs)')';
    for model = 1:13;
        % get the best performing network 
        Ab = se_swc_networks{net}(model,:,ind(net,model));
        % form the network
        Asynth = zeros(100,100);
        Asynth(Ab) = 1;
        Asynth = Asynth + Asynth';
        % compute measures
        x = [];
        x(:,1) = degrees_und(Asynth);
        x(:,2) = clustering_coef_bu(Asynth);
        x(:,3) = betweenness_bin(Asynth)';
        x(:,4) = sum(Cost_Matrix .* Asynth)';
        x(:,5) = efficiency_bin(Asynth,1);
        x(:,6) = sum(matching_ind(Asynth) + matching_ind(Asynth)')';
        % form correlation matrice
        tfd(net,model) = norm(corr(x)-corr(y));
    end
    %display
    disp(sprintf('tfd computed network %g',net));
end
l1_100_600_epo9_thr005_tfd = tfd;
save('l1_100_600_epo9_thr005_tfd.mat','l1_100_600_epo9_thr005_tfd');
%}
%% form tfd plot
% find network (308 epo9)
model = 3;
[a b] = min(m);
c = network_ind(b(model));
d = parametersgen(c,model,ind(c,model),:);
% visualise (manual)
h = figure; h.Position = [100 100 750 300];
subplot(1,2,1);
imagesc(corr(x)-corr(y)); caxis([-.25 .25]);
set(gca,'XTick',[],'YTick',[]);
colormap(magma);
subplot(1,2,2);
imagesc(corr(y)); caxis([-1 1]);
set(gca,'XTick',[],'YTick',[]);
colormap(viridis);
%% visualise toological dissimilarity
load('/imaging/astle/users/da04/PhD/ann_jascha/data/l1_100_600_epo9_thr005_tfd.mat');
tfd = l1_100_600_epo9_thr005_tfd;
tfd(rem,:) = [];
accuracy_use(rem) = [];
% filter by accuracy
accthr = 0.9;
tfd(accuracy_use<accthr,:) = [];
% visualise
u = figure; u.Position = [100 100 800 500]; 
h = iosr.statistics.boxPlot(tfd,...
    'theme','colorall',...
    'symbolMarker','x',...
    'symbolColor',[.5 .5 .5]);
% set colors
% palette = [255 255 0; 255 51 51; 1 255 1; 1 1 255]./256;
palette = [vu(255,:); 1 0.1992 0.1992; vu(180,:); vu(50,:)];
palettei = [1 2 2 3 3 3 3 3 4 4 4 4 4];
for i = 1:13;
    h.handles.box(i).FaceColor = palette(palettei(i),:);
end
box off; 
b = gca; b.TickDir = 'out'; 
b.FontName = 'Arial'; b.FontSize = 24;
ylabel('TFdissimilarity'); xlabel('Generative model');
% visualise over parameters
model = 13;
figure;
reg = 1:size(tfd,1);
[r p] = corr(reg',tfd(:,model));
scatter(reg,tfd(:,model),30,'x');
ylabel('Tfd'); xlabel('Regularisation');
title(sprintf('r=%.3g, p=%.3g',r,p));
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 14;
% anova
[p anovatab stats] = anova1(tfd);
cmp = multcompare(stats);
%% load across epochs
% set directories
se_swc_4_gen = '/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_iv_generative_models';
l1_3_gen = '/imaging/astle/users/da04/PhD/ann_jascha/data/l1_iii_generative_models';
% generative model string
genmodels = string({'sptl','neighbours','matching',...
    'c-avg','clu-min','clu-max','clu-diff','clu-prod',...
    'deg-avg','deg-min','deg-max','deg-diff','deg-prod'});
% set parameters
thr = 0.05;
epo = [3 5 7 9];
netrange = [100 600];
allnets = [100:600];
nnet = netrange(2)-netrange(1);
% initialise
se_swc_energy_epo = nan(4,500,13,1000); 
se_swc_parameters_epo = nan(4,500,13,1000,2);
%{
% change directories
cd(se_swc_4_gen);
% loop 
for epoch = 1:length(epo);
    step = 1;
for net = netrange(1):netrange(2);
    % swc
    str = sprintf('se_swc_4_1k_thr%g_net%g_epo%g_generative_model.mat',thr,net,epo(epoch));
    load(str);
    se_swc_energy_epo(epoch,step,:,:) = output.energy;
    se_swc_parameters_epo(epoch,step,:,:,:) = output.parameters;
    % step
    step = step + 1;
    % display
    disp(sprintf('epo_%g se_swc networks %g loaded',epo(epoch),net));
end
end
se_swc_energy_epoch3579_100_600 = se_swc_energy_epo;
save('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_energy_epoch3579_100_600.mat','se_swc_energy_epoch3579_100_600');
%}
% change directories
cd(l1_3_gen);
% loop 
for epoch = 1:length(epo);
    step = 1;
for net = netrange(1):netrange(2);
    % swc
    str = sprintf('l1_3_1k_thr%g_net%g_epo%g_generative_model.mat',thr,net,epo(epoch));
    load(str);
    l1_energy_epo(epoch,step,:,:) = output.energy;
    l1_parameters_epo(epoch,step,:,:,:) = output.parameters;
    % step
    step = step + 1;
    % display
    disp(sprintf('epo_%g l1 networks %g loaded',epo(epoch),net));
end
end
l1_energy_epoch3579_100_600 = l1_energy_epo;
save('/imaging/astle/users/da04/PhD/ann_jascha/data/l1_energy_epoch3579_100_600.mat','l1_energy_epoch3579_100_600');
%% visualise difference
% load data
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_energy_epoch3579_100_600.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/l1_energy_epoch3579_100_600.mat');
% place data
energydata_a = se_swc_energy_epoch3579_100_600;
energydata_b = l1_energy_epoch3579_100_600;
% comparison x
x = [2 3];
% comparison y
y = [1 4:13];
% palette
pal = [0,176,178;242,98,121]./256;
% group rules
ra = []; rb = [];
for epo = 1:4;
    % group a
    a = squeeze(energydata_a(epo,:,:,:));
    b = min(a,[],3);
    h = mean(b(:,x),2);
    o = mean(b(:,y),2);
    ra(:,epo) = o./h;
    % group b
    a = squeeze(energydata_b(epo,:,:,:));
    b = min(a,[],3);
    h = mean(b(:,x),2);
    o = mean(b(:,y),2);
    rb(:,epo) = o./h;
end
% visualise
h = figure; h.Position = [100 100 450 250];
%plot(1:4,r,...
%    'color',[255 51 51]./256,...
%    'linewidth',3);
stdshade(ra,.5,pal(2,:));
hold on;
stdshade(rb,.5,pal(1,:));
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 18;
xlabel('Training time/Epoch'); ylabel('Relative homophily');
box off;
xticks([1:4]);
xticklabels([3 5 7 9]);
xlim([0.5 4.5]);
ylim([0.7 2.3]);
%% test
% from small world propensity statistics
a = se_swc_4_1k_global_statistics_extra(:,:,9);
bb = se_swc_4_1k_global_statistics_extra(:,:,10);
c = se_swc_4_1k_global_statistics_extra(:,:,11);
d = l1_3_1k_global_statistics_extra(:,:,9);
e = l1_3_1k_global_statistics_extra(:,:,10);
f = l1_3_1k_global_statistics_extra(:,:,11);
% set the bounds
a(a<0) = 0;
bb(bb<0) = 0; bb(bb>1) = 1;
c(c<0) = 0; c(c>1) = 1;
d(d<0) = 0;
e(e<0) = 0; e(e>1) = 1;
f(f<0) = 0; f(f>1) = 1;
% filter
thr = 0.01;
swc_ind = acc_se_swc_4_1k(:,:,2);
swc_ind = swc_ind<thr;
l1_ind = acc_l1_3_1k(:,:,2);
l1_ind = l1_ind<thr;
% compute filters
a(find(swc_ind)) = NaN;
bb(find(swc_ind)) = NaN;
c(find(swc_ind)) = NaN;
d(find(l1_ind)) = NaN;
e(find(l1_ind)) = NaN;
f(find(l1_ind)) = NaN;
% visualise
h = figure;
h.Position = [100 100 1000 400];
subplot(2,3,1);
imagesc(a); 
co = colorbar; co.Label.String = 'SWP'; ylabel('Network'); xlabel('Epoch');
bo = gca; bo.FontName = 'Arial'; bo.FontSize = 12; bo.TickLength = [0 0];
subplot(2,3,2);
imagesc(bb); 
co = colorbar; co.Label.String = 'dC'; ylabel('Network'); xlabel('Epoch');
bo = gca; bo.FontName = 'Arial'; bo.FontSize = 12; bo.TickLength = [0 0];
subplot(2,3,3);
imagesc(c);
co = colorbar; co.Label.String = 'dL'; ylabel('Network'); xlabel('Epoch');
bo = gca; bo.FontName = 'Arial'; bo.FontSize = 12; bo.TickLength = [0 0];
subplot(2,3,4);
imagesc(d); 
co = colorbar; co.Label.String = 'SWP'; ylabel('Network'); xlabel('Epoch');
bo = gca; bo.FontName = 'Arial'; bo.FontSize = 12; bo.TickLength = [0 0];
subplot(2,3,5);
imagesc(e); 
co = colorbar; co.Label.String = 'dC'; ylabel('Network'); xlabel('Epoch');
bo = gca; bo.FontName = 'Arial'; bo.FontSize = 12; bo.TickLength = [0 0];
subplot(2,3,6);
imagesc(f); 
co = colorbar; co.Label.String = 'dL'; ylabel('Network'); xlabel('Epoch');
bo = gca; bo.FontName = 'Arial'; bo.FontSize = 12; bo.TickLength = [0 0];
% visualise means
% estimates to plot on top
estimates = [.84,.78, .80,.60,.64,.71];
estimates_labels = string({'Human DSI','Human rs-fMRI','Cat Cortex','C.elegans (binary)','C elegans (weighted)','Macaque'});
h = figure; h.Position = [100 100 1000 400];
subplot(1,3,1); 
stdshade(a,.6,[.4 .2 .7]); hold on; stdshade(d,.9,[.9 .9 .9]); 
yline(0.6);
ylabel('SWP'); xlabel('Epoch'); box off; ylim([0.4 1]);
bo = gca; bo.FontName = 'Arial'; bo.FontSize = 14; bo.TickDir = 'out';
subplot(1,3,2);  
stdshade(bb,.6,[.4 .2 .7]); hold on; stdshade(e,.9,[.9 .9 .9]); 
ylabel('dC'); xlabel('Epoch'); box off; ylim([0 inf]);
bo = gca; bo.FontName = 'Arial'; bo.FontSize = 14; bo.TickDir = 'out';
subplot(1,3,3); 
stdshade(c,.6,[.4 .2 .7]); hold on; stdshade(f,.9,[.9 .9 .9]); 
ylabel('dL'); xlabel('Epoch'); box off; ylim([0 inf]);
bo = gca; bo.FontName = 'Arial'; bo.FontSize = 14; bo.TickDir = 'out';
%% collect the recurrent neural network data into ordered cell arrays
% network data
ann_nets = {net_l1_3_1k,net_se_swc_4_1k};
% summary array data
summary_array_data = {summary_array_l1_3_1k summary_array_swc_4_1k};
% compute the sample size of each network
nnets = [];
for i = 1:length(ann_nets);
    nnets(i) = size(ann_nets{i},1);
end
% accuracy data
acc_nets = {acc_l1_3_1k,acc_se_swc_4_1k}
% parameters
parameters = {parameters_l1_3_1k,parameters_se_swc_4_1k};
% box cost matrix
box = load(strcat(project,se_swc_1,'/SN_1_20211118-153814.mat'));
D = box.Cost_Matrix;
C = box.Coordinates;
%% initialise statistics
% set what networks to compute
ns = [1:1:1001]; 
ns(1) = 1;
% set the number of local and global measures calculated
nlmeasures = 6;
ngmeasures = 8;
% set number of permutations for small worldness
nperm = 1000;
% binarisation proportion threshold for small worldness
thr = 0.2;
% global statistics labels
global_label = string(...
    {'Total weight',...
    'Total weighted edge length',...
    'Global efficiency',...
    'Homophily per weight',...
    'Modularity (Q)',...
    'Efficiency per weight',...
    'Corr(Weight,Euclidean)',...
    'Small-worldness (\sigma)',...
    'SWP',...
    'dC',...
    'dL'});

global_label = string({'Total weight','Corr(Weight,Euclidean)','Modularity (Q)','Small-worldness (\sigma)'});
l1_networks_neuron_statistics = squeeze(l1_networks_neuron_statistics(:,:,[1 7 5 8]));

% local statistics labels
local_label = string(...
    {'Strength',...
    'Clustering',...
    'Betweenness',...
    'Weighted edge length',...
    'Communicability',...
    'Matching'});
%{
%% compute some simple statistics
% set threshold
thre = [.5 .25 .1];
nthre = length(thre);
% initialise
global_statistics = [];
% loop over the nets
for group = 1:ngroups;
    % take the group
    u = ann_nets{group};
    % over each network, compute the statistics
    for i = 1:length(ns);
        % take the length of training
        train = size(u,2);
        % loop over training
        for t = 1:train;
            % get network
            n = squeeze(ann_nets{group}(ns(i),t,:,:));
            % optional step to make all connections positive
            n = abs(n);
            % loop over thresholds
            for thr = 1:nthre;
                thrl = thre(thr);
                % take binarisation
                a = threshold_proportional(n,thrl);
                a = double(a>0);
                
                
                % compute modularity
                [~,global_statistics(group,i,t,thr,1)] = modularity_und(a);
                
                
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
                for j = 1:nperm;
                    % form a random
                    Arand = makerandCIJ_und(nnode,m);
                    % form a lattic
                    clu_perm(j) = mean(clustering_coef_bu(Arand));
                    cpl_perm(j) = charpath(Arand);
                end
                % calclate means
                mean_clu_perm = mean(clu_perm);
                mean_cpl_perm = mean(cpl_perm);
                % calculate smw
                global_statistics(group,i,t,thr,2) = [clu/mean_clu_perm]/[cpl/mean_cpl_perm];
            end
                % display training point
                display(sprintf('%s network %g, training point %g computed...',network_labels(group),ns(i),t));
        end
    end
    % display network
    display(sprintf('%s network %g computed',network_labels(group),ns(i)));
end
%% look at thresholding effects
figure;
for group = 1:3;
    x = squeeze(global_statistics(group,:,:,:,2));
    m = squeeze(mean(x,1));
    plot(m);
    hold on;
end
    %}
%% load pre-computed statistics 
% global_statistics = network_statistics_comparisons.global_statistics;
% local_statistics = network_statistics_comparisons.local_statistics;
% topological_organization = network_statistics_comparisons.topological_organization;
ns = [1:1000];
global_statistics = ...
    {l1_3_1k_global_statistics_extra(2:1001,:,:)...
    se_swc_4_1k_global_statistics_extra(2:1001,:,:)};
acc_nets = {acc_l1_3_1k(2:1001,:,:),...
    acc_se_swc_4_1k(2:1001,:,:)};
%% compute statistic differences
% set limits for each measure
alimits = [0 900; 0 25; 0 0.12; 0 0.35; 0 0.9; 0 7e-4; -0.6 0.1; 0 8];
mlimits = [];
% set measure
measure = 1;
% plot all global statitics over time
h = figure; h.Position = [100 100 700 300];
for group = 1:ngroups;
    o = squeeze(global_statistics{group}(:,:,measure))';
    subplot(1,ngroups,group);
    for i = 1:length(ns);
        plot(o(:,i),...
            'linewidth',3);
        hold on;
    end
    ylabel(global_label(measure),'linewidth',10);
    bb = gca; bb.TickDir = 'out'; bb.FontName = 'Arial'; bb.FontSize = 14;
    xlabel('time');
    ylim(alimits(measure,:));
    box off;
end
sgtitle(sprintf('comparisons between networks across regularisations',network_labels{group}));
% plot all global statitics over time in an order
order = [1 2 3 4 6 7 5 8];
h = figure; h.Position = [100 100 1200 500];
for measure = 1:ngmeasures;
    subplot(2,ngmeasures/2,measure);
    for group = 1:ngroups;
        % take the ordered measure 
        o = squeeze(global_statistics{group}(:,:,order(measure)))';
        % plot the mean and std
        errorbar(mean(o,2),std(o,[],2),'linewidth',3);
        ylabel(global_label(order(measure)),'linewidth',10);
        bb = gca; bb.TickDir = 'out'; bb.FontName = 'Arial'; bb.FontSize = 14;
        xlabel('time'); box off;
        hold on;
    end
end
sgtitle(sprintf('comparisons between networks across regularisations',network_labels{group}));
%% nicer version
% plot all global statitics over time in an order
order = [1 7 5 8];
% accuracy limit
acclim = 0.9;
% regularisation limits
rem = [];
% colorpallete
% pal = [0,176,178;111,86,149;242,98,121;178,162,150]./256;
pal = [0,176,178;242,98,121;111,86,149;178,162,150]./256;
okeep = []; akeep = [];
% visualise
% h = figure; h.Position = [100 100 1000 300]; % original sizing
h = figure; h.Position = [100 100 1600 300]; % original sizing
for i = 1:length(order);
    subplot(1,length(order),i);
    for group = 1:ngroups;
        % get the size of the network
        w = squeeze(global_statistics{group}(:,:,1));
        % remove 
        w(rem,:) = [];
        % continue
        w = mean(w);
        w = w(:); % ignoring the zero indexed part e.g. 2:end
        w = 1.5*w;
        % take the ordered measure and keep
        o = squeeze(global_statistics{group}(:,:,order(i)))';
        okeep(:,:,i,group) = o';
        % remove
        okeep(rem,:,:,:) = [];
        % continue
        % remove first epoch
        %o(1,:) = [];
        % take accuracy
        a = (acc_nets{group}(:,:,2)<acclim)';
        akeep(:,:,i,group) = a';
        % remove
        akeep(rem,:,:,:) = [];
        % continue
        o(a) = NaN;
        n = sum(o~=0,2);
        % plot the mean and std
        bar = mean(o,2,'omitnan');
        err = std(o,[],2,'omitnan')./sqrt(n); % standard error
        u = errorbar(bar,2*err,... % 3 standard errors
            'linewidth',4,...
            'color',pal(group,:));
        u.Line.ColorType = 'truecoloralpha';
        u.Line.ColorData(4) = 0.5*256;
        % plot on the size of the sparsity
        hold on;
        ind = ~isnan(bar);
        k = scatter(find(ind),bar(ind),...
            'filled',...
            'markerfacecolor',pal(group,:),...
            'sizedata',w(ind),...
            'markerfacealpha',.5);
        % labels
        ylabel(global_label(order(i)),'linewidth',10);
        bb = gca; 
        bb.TickDir = 'out'; 
        bb.FontName = 'Arial'; 
        bb.FontSize = 16;
        bb.TickLength = [.02 .02];
        xlabel('Training time/Epoch'); 
        box off;
        xlim([1 11]); xticks([1:2:11]); xticklabels(0:2:10);
    end
end
% compute statistical differences
stat = 3;
epoch = 9;
x = squeeze(okeep(:,epoch+1,stat,1));
indx = squeeze(akeep(:,epoch+1,stat,1));
x(find(indx)) = [];
y = squeeze(okeep(:,epoch+1,stat,2));
indy = squeeze(akeep(:,epoch+1,stat,2));
y(find(indy)) = [];
h = figure; h.Position = [100 100 350 300];
histogram(x,...
    'facecolor',pal(1,:),...
    'edgecolor','w'); hold on; 
histogram(y,...
    'facecolor',pal(2,:),...
    'edgecolor','w');
box off;
xlabel(global_label(order(stat))); ylabel('Frequency');
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 16;
a = nan(1000,2);
a(1:length(x),1) = x;
a(1:length(y),2) = y;
[h p] = ttest(a(:,1),a(:,2));
d = computeCohen_d(a(:,1),a(:,2));
%% plot the parameter space of statistics
% set group
group = 2;
% set statistic
stat = 1;
% get data
space = squeeze(global_statistics{group}(:,2:end,stat));
% get percentage
spacen = space./max(space,[],'all');
% get acc data
accspace = squeeze(acc_nets{group}(:,2:end,2));
% get percentage
accspacen = accspace./max(accspace,[],'all');
% visualise
figure; 
imagesc(spacen);
%imagesc(accspacen); 
c = colorbar; c.Label.String = global_label(stat); %c.Label.String = 'Accuracy';
c.TickLabels = {'0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'};
colormap(viridis); caxis([0 1]);
xlabel('Training time/Epoch'); ylabel('Regularisation');
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 16; box off;
% relate to accuracy
d = acc_nets{group}(:,2:end,2);
% plot
figure; scatter(d(:),space(:)); 
[r p] = corr(space(:),d(:));
xlabel(global_label(i)); ylabel('Accuracy (%)');
%% plot accuracy
% accuracy
h = figure; h.Position = [100 100 350 300];
for group = 1:ngroups;
    % get the size of the network
    w = squeeze(global_statistics{group}(:,:,1));
    w = mean(w);
    w = w(:); % ignoring the zero indexed part e.g. 2:end
    w = 1.5*w;
    % take accuracy
    a = (acc_nets{group}(:,:,2))';
    % plot the mean and std
    bar = mean(a,2,'omitnan');
    err = std(a,[],2,'omitnan')./sqrt(n); % standard error
    u = errorbar(bar,2*err,...
        'linewidth',4,...
        'color',pal(group,:));
    u.Line.ColorType = 'truecoloralpha';
    u.Line.ColorData(4) = 0.5*256;
    % plot on the size of the sparsity
    hold on;
    ind = ~isnan(bar);
    k = scatter(find(ind),bar(ind),...
        'filled',...
        'markerfacecolor',pal(group,:),...
        'sizedata',w(ind),...
        'markerfacealpha',.5);
    % labels
    ylabel('Validation accuracy (%)','linewidth',10);
    bb = gca; 
    bb.TickDir = 'out'; 
    bb.FontName = 'Arial'; 
    bb.FontSize = 16;
    bb.TickLength = [.02 .02];
    xlabel('Training time/Epoch'); 
    box off;
    xlim([1 11]); xticks([1:2:11]); xticklabels(0:2:10);
    yticks([.5:.1:.9]); yticklabels(50:10:90); ylim([0.5 0.9]);
end
%% plot accuracy backwards from a specific epoch
% set epoch
epoch = 9;
% set accuracy threshold
accthr = 0.9;
% accuracy
h = figure; h.Position = [100 100 350 300];
for group = 1:ngroups;
    % get the size of the network
    w = squeeze(global_statistics{group}(:,:,1));
    w = mean(w);
    w = w(:); % ignoring the zero indexed part e.g. 2:end
    w = 1.5*w;
    % take accuracy
    a = (acc_nets{group}(:,:,2))';
    % take threshold at the specific epoch
    u = a(epoch+1,:);
    i = find(u<accthr);
    a(:,i) = NaN;
    % plot the mean and std
    bar = mean(a,2,'omitnan');
    err = std(a,[],2,'omitnan')./sqrt(n); % standard error
    u = errorbar(bar,2*err,...
        'linewidth',4,...
        'color',pal(group,:));
    u.Line.ColorType = 'truecoloralpha';
    u.Line.ColorData(4) = 0.5*256;
    % plot on the size of the sparsity
    hold on;
    ind = ~isnan(bar);
    k = scatter(find(ind),bar(ind),...
        'filled',...
        'markerfacecolor',pal(group,:),...
        'sizedata',w(ind),...
        'markerfacealpha',.5);
    % labels
    ylabel('Validation accuracy (%)','linewidth',10);
    bb = gca; 
    bb.TickDir = 'out'; 
    bb.FontName = 'Arial'; 
    bb.FontSize = 16;
    bb.TickLength = [.02 .02];
    xlabel('Training time/Epoch'); 
    box off;
    xlim([1 11]); xticks([1:2:11]); xticklabels(0:2:10);
    yticks([.5:.1:1]); yticklabels(50:10:100); ylim([0.5 1]);
end
%% plot a single epoch's accuracy
% set hyperparameters
group = 2;
epoch = 9;
% take data
d = squeeze(acc_nets{group}(:,epoch+1,2));
% get colours
c = viridis(8);
c = c(3,:);
% visualise over regulisation
h = figure; h.Position = [100 100 350 400];
scatterhist(parameters_se_swc_4_1k(2:end,2),100*d,...
    'kernel','on',...
    'direction','out',...
    'marker','o',...
    'color',c,...
    'markersize',6);
xlabel('Regularisation'); ylabel('Validation accuracy (%)');
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 16;
box off;
%% pick a single epoch to view
% set epoch 
epoch = 6;
% plot all global statitics over time in an order
order = [7 5 8];
% accuracy limit
acclim = 0.95;
% colorpallete
pal = [0,176,178;111,86,149;242,98,121;178,162,150]./256;
% visualise
h = figure; h.Position = [100 100 1200 350];
wall = [];
for i = 1:length(order);
    subplot(1,length(order),i);
    for group = 1:ngroups;
        % get the statistic
        w = squeeze(global_statistics{group}(:,epoch+1,order(i)));
        % get the accuracy 
        a = squeeze(acc_nets{group}(2:end,epoch+1,2));
        alim = find(a<acclim);
        % keep only at the correct limit
        w(alim) = NaN;
        % keep data
        wall(:,group) = w; % ignoring the zero indexed part
    end
    % statistical test
    [h p ci] = ttest(wall(:,1),wall(:,2));
    % compute Cohen's d
    d = computeCohen_d(wall(:,2),wall(:,1));
    % visualise
    h = iosr.statistics.boxPlot(wall,...
        'theme','colorall',...
        'boxalpha',.5,...
        'symbolmarker','x',...
        'symbolcolor',[.5 .5 .5]);
    xticklabels({'L1','SWC'});
    ylabel(global_label(order(i)));
    bb = gca; bb.TickDir = 'out'; 
    bb.FontName = 'Arial'; bb.FontSize = 16;
    bb.TickLength = [.02 .02];
    h.handles.box(1).FaceColor = pal(group,:);
    title(sprintf('p=%.3g, d=%.3g',p,d));
end
sgtitle(sprintf('Network comparisons, epoch=%g, accuracy >%.3g%%',epoch,acclim*100));
%% compute mixed selectivity
%{
% set epoch
epoch = 6;
% plot over time windows
tws = [9];
% set a accuracy lower limit
acclim = 0;
% initialise
r = nan(2,10,1000,length(tws)); % check second dimension
delta = nan(2,10,1000,100,length(tws));
goal = nan(2,10,1000,100);
choice = nan(2,10,1000,100);
% take data
summary_array_data = {summary_array_l1_3_1k,summary_array_swc_4_1k};
% compute
for group = 1:2;
    for epoch = 1:10; % check
    for i = 1:length(tws);
        % display
        disp(sprintf('time window %g...',tws(i)));
        for network = 1:1000;
            % display
            disp(sprintf('epoch %g, time window %g, network %g...',epoch,tws(i),network));
            % get indicator
            ind = ...
                summary_array_data{group}(:,1)==network & ...
                summary_array_data{group}(:,3)==epoch & ...
                summary_array_data{group}(:,4)==tws(i) & ...
                summary_array_data{group}(:,6)>acclim;
            % contingent on in not empty
            if sum(ind)>0;
            % get variables
            x = summary_array_data{group}(ind,8);
            y = summary_array_data{group}(ind,9);
            % get correlation
            r(group,epoch,network,i) = corr(x,y);
            delta(group,epoch,network,:,i) = x-y;
            % keep data
            goal(group,epoch,network,:) = x;
            choice(group,epoch,network,:) = y;
            else
                % display
                disp(sprintf('epoch %g time window %g, network %g below the accuracy threhsold...',epoch,tws(i),network));
            end
        end
    end
    end
end
tw9_epo1to10_goal_choice_ms = struct;
tw9_epo_1_10_goal_choice_ms.goal = goal;
tw9_epo_1_10_goal_choice_ms.choice = choice;
tw9_epo_1_10_goal_choice_ms.mixed_selectivity = r;
save('tw9_epo_1_10_goal_choice_ms.mat','tw9_epo_1_10_goal_choice_ms','-v7.3');
%}
%% plot the parameter space of functional measures
load('/imaging/astle/users/da04/PhD/ann_jascha/data/tw6_epo_1_10_goal_choice_ms.mat');
% set group
group = 2;
% get data
space = squeeze(tw6_epo_1_10_goal_choice_ms.mixed_selectivity(group,:,:))';
% visualise
figure; 
imagesc(space);; 
c = colorbar; 
colormap(viridis); caxis([-1 1]);
xlabel('Training time/Epoch'); ylabel('Regularisation');
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 16; box off;
% filter by accuracy
d = squeeze(acc_nets{group}(:,2:end,2));
%% visualise mixed selectivity analysis
% set the data to view
a = squeeze(r(1,:,1));
b = squeeze(r(2,:,1));
% statistical test
[h p] = ttest(a,b);
d = computeCohen_d(a,b);
% visualise
h = figure; h.Position = [100 100 300 300];
u = histogram(a,15,...
    'edgecolor',...
     [0,176,178]./256,...
     'facealpha',.75,...
     'edgealpha',.5,...
     'linewidth',6,...
     'displaystyle','stairs');
hold on;
%{
xline(median(a,'omitnan'),...
    'color',[0,176,178]./256,...
    'linewidth',6);
%}
histogram(b,15,...
    'edgecolor',...
    [242,98,121]./256,...
    'edgealpha',.75,...
    'facealpha',.5,...
    'linewidth',6,...
    'displaystyle','stairs');
hold on;
xline(0,'color','k')
%{
xline(median(b,'omitnan'),...
    'color',[242,98,121]./256,...
    'linewidth',6);
%}
box off;
xlabel('corr(Goal,Choice)');
ylabel('Frequency');
u = gca; u.FontName = 'Arial'; u.FontSize = 16; u.TickDir = 'out'; u.TickLength = [.02 .02];
xlim([-1 1]); 

% plot against regularisation 
% set stat
stat = 8;
% set xlims
xlims = [0 4.5];
% set xlab
xlab = 'Small-worldness (\sigma)';
% visualise
h = figure; h.Position = [100 100 800 300];

subplot(1,2,1); 
x = global_statistics{1}(:,7,stat);
[ra p] = corr(x(~isnan(a)),a(~isnan(a))');
u = scatter(x(~isnan(a)),a(~isnan(a)),300,viridis(sum(~isnan(a))),...
    'marker','.');
ylim([-1 1]); 
xlim(xlims);
xlabel(xlab); ylabel('corr(Goal,Choice');
bb = gca; bb.TickDir = 'out'; bb.FontName = 'Arial'; bb.FontSize = 16; box off;
c = colorbar; c.Label.String = 'Regularisation'; 
colormap(viridis); c.TickLength = [0 0]; c.Ticks = [];

subplot(1,2,2); 
x = global_statistics{2}(:,7,stat);
[rb p] = corr(x(~isnan(b)),b(~isnan(b))');
u = scatter(x(~isnan(b)),b(~isnan(b)),300,viridis(sum(~isnan(b))),...
    'marker','.');
ylim([-1 1]); 
xlim(xlims);
xlabel(xlab); ylabel('corr(Goal,Choice');
bb = gca; bb.TickDir = 'out'; bb.FontName = 'Arial'; bb.FontSize = 16; box off;
c = colorbar; c.Label.String = 'Regularisation'; 
colormap(viridis); c.TickLength = [0 0]; c.Ticks = [];

% set the difference between time windows
a = squeeze(delta(2,:,:,1));
b = squeeze(delta(2,:,:,2));
% remove
a(isnan(sum(a')),:) = []; a = a(:);
b(isnan(sum(b')),:) = []; b = b(:);

% statistical test
[h p] = ttest(a,b);
d = computeCohen_d(a,b);

uu = viridis;
cola = uu(200,:);
colb = uu(126,:);
% visualise
h = figure; h.Position = [100 100 300 300];
u = histogram(a,...
    'edgecolor',...
     cola,...
     'facealpha',.75,...
     'edgealpha',.5,...
     'linewidth',4,...
     'displaystyle','stairs');
hold on;
%{
xline(median(a,'omitnan'),...
    'color',[0,176,178]./256,...
    'linewidth',6);
%}
histogram(b,...
    'edgecolor',...
     colb,...
    'edgealpha',.75,...
    'facealpha',.5,...
    'linewidth',4,...
    'displaystyle','stairs');
hold on;
xline(0,'color','k')
%{
xline(median(b,'omitnan'),...
    'color',[242,98,121]./256,...
    'linewidth',6);
%}
ylim([0 2200]); xlim([-4 4]);
box off;
xlabel('Goal-Choice');
ylabel('Frequency');
u = gca; u.FontName = 'Arial'; u.FontSize = 16; u.TickDir = 'out'; u.TickLength = [.02 .02];
%% view embedding statistics
% set time window
tw = 9;
% get the embedding data
a = squeeze(se_swc_4_1k_all_embedding_epoch9.pchoice(:,tw));
b = squeeze(l1_3_1k_all_embedding_epoch9.pchoice(:,tw));
% accuracy threshold
accthr = 0.9;
c = squeeze(se_swc_4_1k_all_embedding_epoch9.accuracy(:,tw));
d = squeeze(l1_3_1k_all_embedding_epoch9.accuracy(:,tw));
% get distribution values
h = figure; h.Position = [100 100 320 160];
subplot(1,2,1);
histogram(a(c>accthr),20,...
     'facecolor',...
     [242,98,121]./256,...
     'edgecolor','w');
ylim([0 160]);
xlim([0 1]);
box off;
xlabel('p_{perm}');
ylabel('Frequency');
u = gca; u.FontName = 'Arial'; u.FontSize = 14; u.TickDir = 'out'; u.TickLength = [.02 .02];
subplot(1,2,2);
histogram(b(d>accthr),20,...
     'facecolor',...
     [0,176,178]./256,...
     'edgecolor','w')
ylim([0 160]);
xlim([0 1]);
box off;
xlabel('p_{perm}');
u = gca; u.FontName = 'Arial'; u.FontSize = 14; u.TickDir = 'out'; u.TickLength = [.02 .02];
%% view baseline comparisons statistics: structural
% set the global statistics
ns = [1:100];
global_statistics = {...
    l1_3_global_statistics.global_statistics,...
    se_2_global_statistics(2:101,:,:),...
    swc_1_global_statistics_extra(2:101,:,:),...
    se_swc_4_global_statistics.global_statistics};
acc_nets = {...
    acc_l1_3(2:101,:,:),...
    acc_se_2(2:101,:,:),...
    acc_swc_1(2:101,:,:),...
    acc_se_swc_4(2:101,:,:)};
% set label
global_label = l1_3_global_statistics.labels;
global_label = string({'Total weight','Weighted edge length','Efficiency','Homophiliy per weight',...
    'Modularity (Q)','Efficiency per weight','corr(W,D)','Small-worldness (\sigma)'});
% set number of groups
nbase = 4;
% set xlims
xlimits = [2 10];
% plot all global statitics over time in an order
order = [1 7 5 8];
% accuracy limit
acclim = 0.9;
% regularisation limits
rem = [];
% color pallete
% pal = [0,176,178;111,86,149;242,98,121;178,162,150]./256;
% pal = [0,176,178;242,98,121;111,86,149;178,162,150;200 200 200]./256;
pal = [...
    0 176 178;...
    255 218 148;...
    128 128 128;...
    242 98 121]./256;
okeep = []; akeep = [];
% visualise
% h = figure; h.Position = [100 100 1000 300]; % original sizing
h = figure; h.Position = [100 100 1600 300]; % original sizing
for i = 1:length(order);
    subplot(1,length(order),i);
    for group = 1:nbase;
        % get the size of the network
        w = squeeze(global_statistics{group}(:,:,1));
        % remove 
        w(rem,:) = [];
        % continue
        w = mean(w);
        w = w(:); % ignoring the zero indexed part e.g. 2:end
        w = 1.5*w;
        % take the ordered measure and keep
        o = squeeze(global_statistics{group}(:,:,order(i)))';
        okeep(:,:,i,group) = o';
        % remove
        okeep(rem,:,:,:) = [];
        % continue
        % remove first epoch
        %o(1,:) = [];
        % take accuracy
        a = (acc_nets{group}(:,:,2)<acclim)';
        akeep(:,:,i,group) = a';
        % remove
        akeep(rem,:,:,:) = [];
        % continue
        o(a) = NaN;
        n = sum(o~=0,2);
        % plot the mean and std
        bar = mean(o,2,'omitnan');
        err = std(o,[],2,'omitnan')./sqrt(n); % standard error
        u = errorbar(bar,2*err,... % 3 standard errors
            'linewidth',4,...
            'color',pal(group,:));
        u.Line.ColorType = 'truecoloralpha';
        u.Line.ColorData(4) = 0.5*256;
        % plot on the size of the sparsity
        hold on;
        ind = ~isnan(bar);
        k = scatter(find(ind),bar(ind),...
            'filled',...
            'markerfacecolor',pal(group,:),...
            'sizedata',w(ind),...
            'markerfacealpha',.5);
        % labels
        ylabel(global_label(order(i)),'linewidth',10);
        bb = gca; 
        bb.TickDir = 'out'; 
        bb.FontName = 'Arial'; 
        bb.FontSize = 16;
        bb.TickLength = [.02 .02];
        xlabel('Training time/Epoch'); 
        box off;
        xlim(xlimits+1); xticks([1:2:11]); xticklabels(0:2:10);
    end
end
% compute statistical differences
stat = 3;
epoch = 9;
x = squeeze(okeep(:,epoch+1,stat,1));
indx = squeeze(akeep(:,epoch+1,stat,1));
x(find(indx)) = [];
y = squeeze(okeep(:,epoch+1,stat,2));
indy = squeeze(akeep(:,epoch+1,stat,2));
y(find(indy)) = [];
h = figure; h.Position = [100 100 350 300];
histogram(x,...
    'facecolor',pal(1,:),...
    'edgecolor','w'); hold on; 
histogram(y,...
    'facecolor',pal(2,:),...
    'edgecolor','w');
box off;
xlabel(global_label(order(stat))); ylabel('Frequency');
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 16;
a = nan(1000,2);
a(1:length(x),1) = x;
a(1:length(y),2) = y;
[h p] = ttest(a(:,1),a(:,2));
d = computeCohen_d(a(:,1),a(:,2));
%% view baseline comparisons statistics: functional
% set the global statistics
ns = [1:100];
global_statistics = {...
    se_swc_4_global_statistics.global_statistics,...
    se_swc_4r_global_statistics(2:101,:,:),...
    se_swc_4h_global_statistics_extra(2:101,:,:)};
acc_nets = {...
    acc_se_swc_4(2:101,:,:),...
    acc_se_swc_4r(2:101,:,:),...
    acc_se_swc_4h(2:101,:,:)};
% set label
global_label = l1_3_global_statistics.labels;
global_label = string({'Total weight','Weighted edge length','Efficiency','Homophiliy per weight',...
    'Modularity (Q)','Efficiency per weight','corr(W,D)','Small-worldness (\sigma)'});
% set number of groups
nbase = 3;
% set xlims
xlimits = [2 10];
% plot all global statitics over time in an order
order = [1 7 5 8];
% accuracy limit
acclim = 0.9;
% regularisation limits
rem = [];
% color pallete
% pal = [0,176,178;111,86,149;242,98,121;178,162,150]./256;
% pal = [0,176,178;242,98,121;111,86,149;178,162,150;200 200 200]./256;
pal = [...
    242 98 121
    115 147 179;...
    0 150 255;...
    ]./256;
okeep = []; akeep = [];
% visualise
% h = figure; h.Position = [100 100 1000 300]; % original sizing
h = figure; h.Position = [100 100 1600 300]; % original sizing
for i = 1:length(order);
    subplot(1,length(order),i);
    for group = 1:nbase;
        % make a contingency on the acclim - if random make 0
        if group == 2;
            acclim = 0;
        else
            acclim = 0.9;
        end
        % get the size of the network
        w = squeeze(global_statistics{group}(:,:,1));
        % remove 
        w(rem,:) = [];
        % continue
        w = mean(w);
        w = w(:); % ignoring the zero indexed part e.g. 2:end
        w = 1.5*w;
        % take the ordered measure and keep
        o = squeeze(global_statistics{group}(:,:,order(i)))';
        okeep(:,:,i,group) = o';
        % remove
        okeep(rem,:,:,:) = [];
        % continue
        % remove first epoch
        %o(1,:) = [];
        % take accuracy
        a = (acc_nets{group}(:,:,2)<acclim)';
        akeep(:,:,i,group) = a';
        % remove
        akeep(rem,:,:,:) = [];
        % continue
        o(a) = NaN;
        n = sum(o~=0,2);
        % plot the mean and std
        bar = mean(o,2,'omitnan');
        err = std(o,[],2,'omitnan')./sqrt(n); % standard error
        u = errorbar(bar,2*err,... % 3 standard errors
            'linewidth',4,...
            'color',pal(group,:));
        u.Line.ColorType = 'truecoloralpha';
        u.Line.ColorData(4) = 0.5*256;
        % plot on the size of the sparsity
        hold on;
        ind = ~isnan(bar);
        k = scatter(find(ind),bar(ind),...
            'filled',...
            'markerfacecolor',pal(group,:),...
            'sizedata',w(ind),...
            'markerfacealpha',.5);
        % labels
        ylabel(global_label(order(i)),'linewidth',10);
        bb = gca; 
        bb.TickDir = 'out'; 
        bb.FontName = 'Arial'; 
        bb.FontSize = 16;
        bb.TickLength = [.02 .02];
        xlabel('Training time/Epoch'); 
        box off;
        xlim(xlimits+1); xticks([1:2:11]); xticklabels(0:2:10);
    end
end
% compute statistical differences
stat = 3;
epoch = 9;
x = squeeze(okeep(:,epoch+1,stat,1));
indx = squeeze(akeep(:,epoch+1,stat,1));
x(find(indx)) = [];
y = squeeze(okeep(:,epoch+1,stat,2));
indy = squeeze(akeep(:,epoch+1,stat,2));
y(find(indy)) = [];
h = figure; h.Position = [100 100 350 300];
histogram(x,...
    'facecolor',pal(1,:),...
    'edgecolor','w'); hold on; 
histogram(y,...
    'facecolor',pal(2,:),...
    'edgecolor','w');
box off;
xlabel(global_label(order(stat))); ylabel('Frequency');
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 16;
a = nan(1000,2);
a(1:length(x),1) = x;
a(1:length(y),2) = y;
[h p] = ttest(a(:,1),a(:,2));
d = computeCohen_d(a(:,1),a(:,2));
%% form statistic distributions
% set time
t = 5;
% plot all local statitics over time in an order
order = [1:6];
h = figure; h.Position = [100 100 1200 500];
for measure = 1:nlmeasures;
    subplot(2,nlmeasures/2,measure);
    for group = 1:ngroups;
        % take the ordered measure 
        o = squeeze(local_statistics(group,:,t,:,order(measure)));
        % compute pca scores
        % plot the mean and std
        z = histogram(o(o>0),50,'edgecolor','w');
        xlabel(local_label(order(measure)),'linewidth',10);
        ylabel('Frequency');
        bb = gca; bb.TickDir = 'out'; bb.FontName = 'Arial'; bb.FontSize = 14;
        hold on;
    end
end
sgtitle(sprintf('t=%g. comparisons between l1 (blue), se (orange) and se_swc (yellow) local distributions',t));
%% explore normalizations of local statistics
% set time
t = 6;
% set group
group = 3;
% compute
data = squeeze(local_statistics(group,:,t,:,:));
% concatenate
data = permute(data,[3 1 2]);
data = data(:,:)';
% normalize
normdata = normalize(data); % < set how to normalize here
% visualise
figure;
for i = 1:nlmeasures;
    subplot(2,nlmeasures/2,i);
    histogram(normdata(:,i),'edgecolor','w');
    xlabel(local_label(i)); ylabel('Frequency');
end
sgtitle(sprintf('%s normalized data',network_labels(group)));
%% compute pca across local statistics and compute clusters
% set time
t = 7;
% initialise
coeff = []; score = []; exp = [];
% run pcas
for group = 1:ngroups;
    % take all networks
    o = squeeze(local_statistics(group,:,t,:,:));
    % permute
    o = permute(o,[3 1 2]);
    % all nodes and node statistics for the group
    co = o(:,:)';
    % normalize
    co = normalize(co); % < change how the data is provided to the pca
    % compute a pca
    [coeff(group,:,:),score(group,:,:),~,~,exp(group,:)] = pca(co); 
end
% visualise the two
h = figure; h.Position = [100 100 600 800];
step = 1;
for group = 1:ngroups;
    % subplot
    subplot(ngroups,2,step);
    % scores
    scatter(squeeze(score(group,:,1)),squeeze(score(group,:,2)),'.');
    xlabel(sprintf('PC1: %.3g',exp(group,1)));
    ylabel(sprintf('PC2: %.3g',exp(group,2)));
    title(network_labels{group});
    step = step + 1;
    bb = gca; bb.TickDir = 'out'; bb.FontName = 'Arial'; bb.FontSize = 14;
    % subplot
    subplot(ngroups,2,step);
    % coefficients
    bar(squeeze(coeff(group,:,[1 2]))); 
    xticklabels(local_label); xtickangle(45);
    step = step + 1;
    bb = gca; bb.TickDir = 'out'; bb.FontName = 'Arial'; bb.FontSize = 14;
    ylim([-1 1]);
end
sgtitle(sprintf('t=%g',t));
% plot the statitics of the different groups
nclusters = [6 6 6];
% set the number of components
ncomp = 3;
% form clusters
for group = 1:ngroups;
    idx{group} = kmeans(squeeze(score(group,:,[1:ncomp])),nclusters(group));
end
% form colorpalette
col_l1 = []; col_se_swc_1 = [];
% colors
pal = [105,165,131;...
    41,64,82;...
    143,188,219;...
    244,214,188;...
    217,131,130;...
    136,56,45]./256;
% loop
palettes = {};
for group = 1:ngroups;
    for i = 1:nnode*length(ns);
        for j = 1:nclusters(group);
            if idx{group}(i) == j;
                palettes{group}(i,:) = pal(j,:);
            end
        end
    end
end
% visualise
h = figure; h.Position = [100 100 600 800];
step = 1;
for group = 1:ngroups;
    % subplot
    subplot(ngroups,2,step);
    % scores
    x = squeeze(score(group,:,1))';
    y = squeeze(score(group,:,2))';
    u = scatter(x,y,30,palettes{group},'.');
    xlabel(sprintf('PC1: %.3g',exp(group,1)));
    ylabel(sprintf('PC2: %.3g',exp(group,2)));
    title(network_labels{group});
    step = step + 1;
    bb = gca; bb.TickDir = 'out'; bb.FontName = 'Arial'; bb.FontSize = 14;
    % subplot
    subplot(ngroups,2,step);
    bb = squeeze(local_statistics(group,:,t,:,:));
    bb = permute(bb,[3 1 2]);
    bb = (bb(:,:))'; % < change how data is shown to be different
    for i = 1:nclusters(group);
        % take only this cluster
        zz = bb(idx{group}==i,:);
        % plot mean and std % < note normalization may change how we want to represent this
        errorbar(mean(zz),std(zz),...
            'color',pal(i,:),...
            'linewidth',4);
        xticklabels(local_label);
        xtickangle(45);
        bb = gca; bb.TickDir = 'out'; bb.FontName = 'Arial'; bb.FontSize = 14;
        hold on;
    end
    step = step + 1;
end
sgtitle(sprintf('t=%g',t));
