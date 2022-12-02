%% explore all recurrent neural network data
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
% addpath of sort files
addpath('/imaging/astle/users/da04/PhD/toolboxes/sort_nat');
% addpath of python colours
addpath('/imaging/astle/users/da04/PhD/toolboxes/Colormaps/Colormaps (5)/Colormaps/');
% addpath of weighted smallworld propensity
addpath('/imaging/astle/users/da04/PhD/toolboxes/SWP(1)');
% set directories of the data
project = '/imaging/shared/users/ja02/CBUActors/SpatialRNNs/Results/';
% go to the project
cd(project);
% set paths
l1 = '/mazeGenI_L1_mtI';
l1_2 = '/mazeGenI_L1_mtII';
l1_2r = '/mazeGenI_L1_mtII_randI';
l1_3 = '/mazeGenI_L1_mtIII';
l1_3_1k = '/mazeGenI_L1_mtIII_1k'; % new
se_1 = '/mazeGenI_SE1_mtI';
se_2  = '/mazeGenI_SE1_mtII'; % new
se_sc_1  = '/mazeGenI_SE1_sc_mtI';
swc_1 = '/mazeGenI_SE1_sWcExcl_mtI'; % new new(no se) < CHECK
se_swc_1 = '/mazeGenI_SE1_sWc_mtI';
se_swc_2 = '/mazeGenI_SE1_sWc_mtII';
se_swc_2r = '/mazeGenI_SE1_sWc_mtII_randI';
se_swc_3 = '/mazeGenI_SE1_sWc_mtIII';
se_swc_4 = '/mazeGenI_SE1_sWc_mtIV';
se_swc_4_1k = '/mazeGenI_SE1_sWc_mtIV_1k'; % new
se_swc_4r = '/mazeGenI_SE1_sWc_mtIV_randI'; % new
se_swc_4h = '/mazeGenII_SE1_sWc_mtI'; % new new (hard) < CHECK
se_swc_5 = '/mazeGenI_SE1_sWc_mtV'; % inverse comm
nd = '/NetworkDepot';
rb = 'RegulariserBaselineI';
mn_2 = '/sMNIST_RNNII';
sn_1 = '/SupervisedNetworksI';
sn_2 = '/SupervisedNetworksII';
sn_3 = '/SupervisedNetworksIII';
sn_4 = '/SupervisedNetworksIV';
sn_5 = '/SupervisedNetworksV';
sn_6 = '/SupervisedNetworksVI';
% load global statistics
load('/imaging/astle/users/da04/PhD/ann_jascha/data/l1_3_global_statistics.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/l1_3_1k_global_statistics_extra.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_1_global_statistics.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_2_global_statistics.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_4_global_statistics.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_4_1k_global_statistics_extra.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_4_1k_global_statistics_extra_rescaled.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_4r_global_statistics.mat');
% load embedding
load('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_4_1k_embedding_epoch6.mat');
load('/imaging/astle/users/da04/PhD/ann_jascha/data/l1_3_1k_embedding_epoch6.mat');
% set labels
network_labels = string(...
    {'l1_1','l1_2','l1_2r','l1_3','l1_3_1k',...
    'se_1','se_2',...
    'swc_1',...
    'se_sc_1',...
    'se_swc_1','se_swc_2','se_swc_2r','se_swc_3','se_swc_4','se_swc_4_1k','se_swc_4r','se_swc_4h','se_swc_5',...
    'nd','rb_dist','rb_schuff','mn_2',...
    'sn_1','sn_2','sn_3','sn_4','sn_5','sn_6'});
% set number of nodes
nnode = 100;
%% load all recurrent neural networks

%%% load l1 networks %%%

% change directory
cd(strcat(project,l1));
% set the number of networks in here
nnet = 101;
% set how many timepoints
tt = 11;
% list the directory files
k = dir('*.mat'); 
data      = string({k.name})';
% reorder the networks
reorder = [1 13 24 35 46 57 68 79 90 101 3:12 14:23 25:34 36:45 47:56 58:67 69:78 80:89 91:100 2];
data = data(reorder);
% set parameters
parameters_l1_1 = [1:101;linspace(0.00001,0.04,101)]';
% initialise
net_l1_1 = zeros(nnet,tt,nnode,nnode);
wei_l1_1 = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_l1_1 = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_l1_1(i,t,:,:) = Training_History{t+1,1};
        wei_l1_1(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_l1_1(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('l1 network %g loaded',i-1));
end

%%% load l1_2 networks %%%

% change directory
cd(strcat(project,l1_2));
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
net_l1_2 = zeros(nnet,tt,nnode,nnode);
wei_l1_2 = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_l1_2 = zeros(nnet,tt,2); % acc, val acc
parameters_l1_2 = [];
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_l1_2(i,t,:,:) = Training_History{t+1,1};
        wei_l1_2(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_l1_2(i,t,:) = Training_History{t+1,5:6};
        parameters_l1_2(i,:) = [i Regulariser_Strength];
    end
    % display
    disp(sprintf('l1_2 network %g loaded',i-1));
end

%%% load l1_2r networks %%%

% change directory
cd(strcat(project,l1_2r));
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
net_l1_2r = zeros(nnet,tt,nnode,nnode);
wei_l1_2r = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_l1_2r = zeros(nnet,tt,2); % acc, val acc
parameters_l1_2r = [];
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_l1_2r(i,t,:,:) = Training_History{t+1,1};
        wei_l1_2r(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_l1_2r(i,t,:) = Training_History{t+1,5:6};
        parameters_l1_2r(i,:) = [i Regulariser_Strength];
    end
    % display
    disp(sprintf('l1_2r network %g loaded',i-1));
end

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
% reorder the 1k networks
[s,reorder1k] = sort_nat({k.name});
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

%%% load se_1 networks %%%

% change directory
cd(strcat(project,se_1));
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
parameters_se_1 = [1:101;linspace(0.00001,0.02,101)]';
% initialise
net_se_1 = zeros(nnet,tt,nnode,nnode);
wei_se_1 = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_se_1 = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_se_1(i,t,:,:) = Training_History{t+1,1};
        wei_se_1(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_se_1(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('se_1 network %g loaded',i-1));
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
    disp(sprintf('supervised networks se_2 network %g loaded',i-1));
end

%%% load swc only networks %%%

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
    disp(sprintf('supervised networks swc_1 network %g loaded',i-1));
end

%%% load se_sc_1 networks %%%

% change directory
cd(strcat(project,se_sc_1));
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
parameters_se_sc_1 = [1:101;linspace(0.00001,0.02,101)]';
% initialise
net_se_sc_1 = zeros(nnet,tt,nnode,nnode);
wei_se_sc_1 = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_se_sc_1 = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_se_sc_1(i,t,:,:) = Training_History{t+1,1};
        wei_se_sc_1(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_se_sc_1(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('supervised networks se_sc_1 network %g loaded',i-1));
end

%%% load se_swc_1 %%%

% change directory
cd(strcat(project,se_swc_1));
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
parameters_se_swc_1 = [1:101;linspace(0.001,0.3,101)]';
% initialise
net_se_swc_1 = zeros(nnet,tt,nnode,nnode);
wei_se_swc_1 = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_se_swc_1 = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_se_swc_1(i,t,:,:) = Training_History{t+1,1};
        wei_se_swc_1(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_se_swc_1(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('se_swc_1 network %g loaded',i-1));
end

%%% load se_swc_2 networks %%%

% change directory
cd(strcat(project,se_swc_2));
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
parameters_se_swc_2 = [1:101;linspace(0.001,0.3,101)]';
% initialise
net_se_swc_2 = zeros(nnet,tt,nnode,nnode);
wei_se_swc_2 = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_se_swc_2 = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_se_swc_2(i,t,:,:) = Training_History{t+1,1};
        wei_se_swc_2(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_se_swc_2(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('se_swc_2 network %g loaded',i-1));
end

%%% load se_swc_2r networks %%%

% change directory
cd(strcat(project,se_swc_2r));
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
parameters_se_swc_2r = [1:101;linspace(0.001,0.3,101)]';
% initialise
net_se_swc_2r = zeros(nnet,tt,nnode,nnode);
wei_se_swc_2r = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_se_swc_2r = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_se_swc_2r(i,t,:,:) = Training_History{t+1,1};
        wei_se_swc_2r(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_se_swc_2r(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('se_swc_2r network %g loaded',i-1));
end

%%% load se_swc_3 networks %%%

% change directory
cd(strcat(project,se_swc_3));
% set the number of networks in here
nnet = 11;
% set how many timepoints
tt = 11;
% list the directory files
k = dir('*.mat'); 
data      = string({k.name})';
% reorder the networks
reorderx = [1 3:11 2];
data = data(reorderx);
% set parameters
parameters_se_swc_3 = [1:101;linspace(0.001,0.3,101)]';
% initialise
net_se_swc_3 = zeros(nnet,tt,nnode,nnode);
wei_se_swc_3 = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_se_swc_3 = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_se_swc_3(i,t,:,:) = Training_History{t+1,1};
        wei_se_swc_3(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_se_swc_3(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('se_swc_3 network %g loaded',i-1));
end

%%% load se_swc_4 networks %%%

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

%%% load se_swc_4r networks %%%

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

% load se_swc_5 % invesre comm

% change directory
cd(strcat(project,se_swc_5));
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
parameters_se_swc_5 = [1:101;linspace(0.001,0.3,101)]';
% initialise
net_se_swc_5 = zeros(nnet,tt,nnode,nnode);
wei_se_swc_5 = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_se_swc_5 = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_se_swc_5(i,t,:,:) = Training_History{t+1,1};
        wei_se_swc_5(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_se_swc_5(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('se_swc_5 network %g loaded',i-1));
end


%%% load nd networks %%%

% change directory
cd(strcat(project,nd));
% set the number of networks in here
nnet = 9;
% list the directory files
k = dir('*.csv'); 
data      = string({k.name})';
% indices of reordering
order = [13 14 15 7 8 9 4 5 6 10 11 12 25 26 27 22 23 24 19 20 21 16 17 18 1 2 3];
% reorder
data = data(order);
% get the indices of networks
neti = [3 6 9 12 15 18 21 24 27];
% set parameters
parameters_nd = [1:8;0,0,0,10^-4,0,10^-3,10^-2,10^-1]';
% intialise 
cost_nd = zeros(nnet,nnode,nnode);
net_nd = zeros(nnet,nnode,nnode);
% loop and load
for i = 1:nnet;
    net_nd(i,:,:) = load(data(neti(i)));
    % display
    disp(sprintf('nd network %g loaded',i));
end

%%% load rb %%%

% change directory
cd(strcat(project,rb));
% set the number of networks in here
nnet_dist = 10;
nnet_shuff = 10;
% list the directory files
k = dir('*.csv'); 
data = string({k.name})';
% seperate the groups
data_dist = data(1:10);
data_shuff = data(11:20);
% intialise 
net_rb_dist = zeros(nnet_dist,nnode,nnode);
net_rb_shuff = zeros(nnet_shuff,nnode,nnode);
% loop and load
for i = 1:nnet_dist;
    % load dist data
    net_rb_dist(i,:,:) = load(data_dist(i));
    % display
    disp(sprintf('rb dist network %g loaded',i));
end
% loop and load
for i = 1:nnet_shuff;
    % load dist data
    net_rb_shuff(i,:,:) = load(data_shuff(i));
    % display
    disp(sprintf('rb shuff network %g loaded',i));
end

%%% load mn_2 networks %%%

% change directory
cd(strcat(project,mn_2));
% set the number of networks in here
nnet = 4;
% set how many timepoints
tt = 21;
% list the directory files
k = dir('*.mat'); 
data      = string({k.name})';
% set parameters
parameters_mn_2 = [1:4;0,10^-3,10^-2,5*10-3]';
% intialise 
net_mn_2 = zeros(nnet,tt,nnode,nnode);
wei_mn_2 = zeros(nnet,tt,3); % weight count >0, >1e-7, >1e-3
acc_mn_2 = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_mn_2(i,t,:,:) = Training_History{t+1,1};
        wei_mn_2(i,t,:) = Training_History{t+1,2:4};
        acc_mn_2(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('mn_2 network %g loaded',i));
end

%%% load sn_1 networks %%%

% change directory
cd(strcat(project,sn_1));
% set the number of networks in here
nnet = 8;
% list the directory files
k = dir('*.csv'); 
data      = string({k.name})';
% indices of reordering
order = [10 11 12 4 5 6 1 2 3 7 8 9 22 23 24 19 20 21 16 17 18 13 14 15];
% reorder
data = data(order);
% get the indices of networks
neti = [3 6 9 12 15 18 21 24];
% set parameters 
parameters_sn_1 = NaN;
% intialise 
net_si = zeros(nnet,nnode,nnode);
% loop and load
for i = 1:nnet;
    net_sn_1(i,:,:) = load(data(neti(i)));
    % display
    disp(sprintf('sn_1 network %g loaded',i));
end

%%% load sn_2 networks %%%

% change directory
cd(strcat(project,sn_2));
% set the number of networks in here
nnet = 3;
% set how many timepoints
tt = 11;
% list the directory files
k = dir('*.mat'); 
data      = string({k.name})';
% set parameters
parameters_sn_2 = [1:3;0,10^-2,5*10^-2]';
% intialise 
net_sn_2 = zeros(nnet,tt,nnode,nnode);
wei_sn_2 = zeros(nnet,tt,3); % weight count >0, >1e-7, >1e-3
acc_sn_2 = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_sn_2(i,t,:,:) = Training_History{t+1,1};
        wei_sn_2(i,t,:) = Training_History{t+1,2:4};
        acc_sn_2(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('sn_2 network %g loaded',i));
end

%%% load sn_3 networks %%%

% change directory
cd(strcat(project,sn_3));
% set the number of networks in here
nnet = 15;
% set how many timepoints
tt = 11;
% list the directory files
k = dir('*.mat'); 
data      = string({k.name})';
% indices of reordering
order = [7:15 1:6];
% reorder
data = data(order);
% set parameters
parameters_sn_3 = [1:15;10^-3,10^-2,10^-1,10^-3,0,10^-3,10^-2,10^-1,10^-3,0,10^-4,10^-3,10^-2,5*10^-3,0]';
% intialise 
net_sn_3 = zeros(nnet,tt,nnode,nnode);
wei_sn_3 = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_sn_3 = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_sn_3(i,t,:,:) = Training_History{t+1,1};
        wei_sn_3(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_sn_3(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('sn_3 network %g loaded',i));
end

%%% load sn_4 networks %%%

% change directory
cd(strcat(project,sn_4));
% set the number of networks in here
nnet = 5;
% set how many timepoints
tt = 11;
% list the directory files
k = dir('*.mat'); 
data      = string({k.name})';
% set parameters
parameters_sn_4 = [1:5;10^-2,10^-1,1,10,10^2]';
% intialise 
net_sn_4 = zeros(nnet,tt,nnode,nnode);
wei_sn_4 = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_sn_4 = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_sn_4(i,t,:,:) = Training_History{t+1,1};
        wei_sn_4(t,:) = Training_History{t+1,[2:4 7:8]};
        acc_sn_4(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('sn_4 network %g loaded',i));
end

%%% load sn_5 networks %%%

% change directory
cd(strcat(project,sn_5));
% set the number of networks in here
nnet = 5;
% set how many timepoints
tt = 11;
% list the directory files
k = dir('*.mat'); 
data      = string({k.name})';
% set parameters
parameters_sn_5 = [1:5;10^-5,10^-4,10^-3,10^-2,10^-1]';
% intialise 
net_sn_5 = zeros(nnet,tt,nnode,nnode);
wei_sn_5 = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_sn_5 = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_sn_5(i,t,:,:) = Training_History{t+1,1};
        wei_sn_5(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_sn_5(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('sn_5 network %g loaded',i));
end

%%% load sn_6 networks %%%

% change directory
cd(strcat(project,sn_6));
% set the number of networks in here
nnet = 5;
% set how many timepoints
tt = 11;
% list the directory files
k = dir('*.mat'); 
data      = string({k.name})';
% set parameters
parameters_sn_6 = NaN;
% intialise 
net_sn_6 = zeros(nnet,tt,nnode,nnode);
wei_sn_6 = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_sn_6 = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_sn_6(i,t,:,:) = Training_History{t+1,1};
        wei_sn_6(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_sn_6(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('sn_6 network %g loaded',i));
end

% move back to the origonal directory
cd(project);

%% collect all recurrent neural network data into ordered cell arrays

% network data
ann_nets = {net_l1_1,net_l1_2,net_l1_2r,net_l1_3,net_l1_3_1k,...
    net_se_1,net_se_2,...
    net_swc_1,...
    net_se_sc_1,...
    net_se_swc_1,net_se_swc_2,net_se_swc_2r,net_se_swc_3,net_se_swc_4,net_se_swc_4_1k,net_se_swc_4r,net_se_swc_4h,net_se_swc_5,...
    net_nd,net_rb_dist,net_rb_shuff,net_mn_2,...
    net_sn_1,net_sn_2,net_sn_3,net_sn_4,net_sn_5,net_sn_6};

% compute the sample size of each network
nnets = [];
for i = 1:length(ann_nets);
    nnets(i) = size(ann_nets{i},1);
end

% accuracy data
acc_nets = {acc_l1_1,acc_l1_2,acc_l1_2r,acc_l1_3,acc_l1_3_1k,...
    acc_se_1,acc_se_1,...
    acc_swc_1,...
    acc_se_sc_1,...
    acc_se_swc_1,acc_se_swc_2,acc_se_swc_2r,acc_se_swc_3,acc_se_swc_4,acc_se_swc_4_1k,acc_se_swc_4r,acc_se_swc_4h,acc_se_swc_5,...
    NaN,NaN,NaN,acc_mn_2,...
    NaN,acc_sn_2,acc_sn_3,acc_sn_4,acc_sn_5,acc_sn_6};

% parameters
parameters = {parameters_l1_1,parameters_l1_2,parameters_l1_2r,parameters_l1_3,parameters_l1_3_1k,...
    parameters_se_1,parameters_se_2,...
    parameters_swc_1,...
    parameters_se_sc_1,...
    parameters_se_swc_1,parameters_se_swc_2,parameters_se_swc_2r,parameters_se_swc_3,parameters_se_swc_4,parameters_se_swc_4_1k,parameters_se_swc_4r,parameters_se_swc_4h,parameters_se_swc_5,...
    parameters_nd,NaN,NaN,parameters_mn_2,...
    parameters_sn_1,parameters_sn_2,parameters_sn_3,parameters_sn_4,parameters_sn_5,parameters_sn_6};

% box cost matrix
box = load(strcat(project,se_sc_1,'/SN_1_20211006-112921.mat'));
Dbox = box.Cost_Matrix;
Cbox = box.Coordinates;

% sheet cost matrix
sheet = load(strcat(project,sn_3,'/SNIII_11_20210802-173110.mat'));
Dsheet = sheet.Cost_Matrix;
Csheet = sheet.Coordinates(1:2,:);

% groups that have no time
notime = [19 20 21 23];
%% Display the groups

% display to see what index corresponds to what set
disp(network_labels);

%% visualise a network

% set group of networks e.g. 15
group = 15;
% set network e.g. 50, 290, 700
net = 700;
% time point
time = 6;
% marker multiplier (e.g. 1,6,10,12); 
m = 6;
% set cost coordinates
c = Cbox;
% set distance
d = Dbox;
% get network
gr = ann_nets{group};
% take the matrix
if ismember(group,notime);
    a = abs(squeeze(gr(net,:,:)));
else
    a = abs(squeeze(gr(net,time,:,:)));
end
b = a;
% form a graph
W = graph(b,'upper');
% plot
h = figure; h.Position = [100 100 440 500];
imagesc(a); 
xlabel('Neuron'); ylabel('Neuron');
b = gca; b.TickDir = 'out'; b.TickLength = [0 0]; b.FontName = 'Arial'; b.FontSize = 16;
cu = colorbar; cu.Label.String = 'Weight'; 
cu.Location = 'southoutside'; box off;
cu.TickDirection = 'out';
colormap(viridis); 
h = figure; h.Position = [100 100 650 500];
plot(W,...
    'XData',c(1,:),'YData',c(2,:),'ZData',c(3,:),...
    'linewidth',0.01,...
    'markersize',m*strengths_und(a),...
    'nodelabel',[],...
    'edgecolor',[.3 .3 .3],...
    'edgealpha',.075,...
    'nodecolor',[.7 .7 .7]); 
xticks([]); yticks([]); zticks([]); box off;
h = figure; h.Position = [100 100 440 440];
[r p] = corr(d(a>0),a(a>0));
j = viridis(8);
h = scatter(d(a>0),a(a>0),100,'o',...
    'markerfacecolor',j(3,:),...
    'markerfacealpha',.5,...
    'markeredgecolor','w',...
    'markerfacealpha',1); 
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 16;
xlabel('Euclidean distance');
ylabel('Weight');
if ismember(group,notime)
    sgtitle(sprintf('%s network %g',network_labels{group},net));
else
    sgtitle(sprintf('%s,%g,reg %.3g,t=%g,r=%.3g,p=%.3g',...
        network_labels{group},net,parameters{group}(net,2),time,r,p));
end
%% explore accuracy as a function of training

% set group to plot
group = 18;
% set nets to plot 
ne = 1:1:101;
% set a threshold of accuracy that you consider
thr = 0.9;
% input the labels of the networks
lab = string(parameters{group}(ne,2));
% form the colour palette based on this
col = parula(length(ne));
% take the accuracy data
data = acc_nets{group}(ne,2:end,:);
% take the network data
net_data = squeeze(ann_nets{group}(ne,:,:,:));
% take the parameters
ps = parameters{group}(:,2);
% take same size
n = size(data,1);

% visualise the training and validation accuracy
h = figure; h.Position = [10 100 800 300];
for i = 1:n;
    subplot(1,2,1);
    plot(data(i,:,1),'linewidth',2,'color',col(i,:));
    hold on;
    ylim([0 1]); xlim([1 inf]);
    ylabel('Test accuracy'); xlabel('Time'); xticks([1:10]);
    b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 14; box off;
    subplot(1,2,2);
    plot(data(i,:,2),'linewidth',2,'color',col(i,:));
    hold on;
    ylim([0 1]); xlim([1 inf]);
    ylabel('Validation accuracy'); xlabel('Time'); xticks([1:10]);
    b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 14; box off;
end
c = colorbar; caxis([parameters{group}(ne(1),2) parameters{group}(ne(end),2)]);
c.Label.String = 'Regularisation';
sgtitle(sprintf('%s networks',network_labels{group}));

colsin = flip(bone(length(ne)));
% single figure
h = figure; h.Position = [10 100 500 400];
for i = 1:n;
    f = plot(100*data(i,:,2),'linewidth',2.5,'color',colsin(i,:)); f.Color(4)=.5;
    hold on;
    ylim([0 100]); xlim([1 inf]);
    ylabel('Accuracy (%)'); xlabel('Epoch');
    b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 14; box off;
    xticks([1:10]);
end
c = colorbar; caxis([parameters{group}(ne(1),2) parameters{group}(ne(end),2)]);
colormap(colsin);
c.Label.String = 'Regularisation'; c.Label.FontName = 'Arial'; c.TickDirection = 'out';
sgtitle(sprintf('%s networks',network_labels{group}));

% display what percentage above and below
% histogram
u = squeeze(data(:,:,2));
figure; histogram(u,20,'edgecolor','w');
b = gca; b.TickDir = 'out'; b.FontSize = 25; b.FontName = 'Arial';
xlabel('Accuracy'); ylabel('Frequency');

% visualise the accuracy per unit weight for all
h = figure; h.Position = [10 100 1200 300];
for i = 1:n;
    % compute the network weight over time
    w = sum(abs(squeeze(net_data(i,2:end,:,:))),[2 3]);
    subplot(1,2,1);
    plot(data(i,:,1)./w','linewidth',2,'color',col(i,:));
    hold on;
    xlim([1 inf]);
    ylabel('Test accuracy per weight'); xlabel('Epoch');
    b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 14; box off;
    subplot(1,2,2);
    plot(data(i,:,2)./w','linewidth',2,'color',col(i,:));
    hold on;
    xlim([1 inf]);
    ylabel('Validation accuracy per weight'); xlabel('Epoch');
    b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 14; box off;
end
c = colorbar; caxis([parameters{group}(ne(1),2) parameters{group}(ne(end),2)]);
c.Label.String = 'Regularisation';
sgtitle(sprintf('%s networks',network_labels{group}));

% visualise the accuracy per unit weight for all filtered
h = figure; h.Position = [10 100 1200 300];
for i = 1:n;
    % compute the network weight over time
    w = sum(abs(squeeze(net_data(i,2:end,:,:))),[2 3]);
    % only consider networks which have a accuracy > threshold
    tt = data(i,:,1) > thr;
    tv = data(i,:,2) > thr;
    % plot
    subplot(1,2,1);
    % take the accuracy per weight
    apwt = data(i,:,1)./w';
    % then pick
    apwt(~tt) = NaN;
    % plot
    plot(apwt,'linewidth',2,'color',col(i,:));
    hold on;
    xlim([1 inf]);
    ylabel('Test accuracy per weight'); xlabel('Epoch'); xticks([1:10]);
    b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 14; box off;
    % same for validation
    subplot(1,2,2);
    % take the accuracy per weight
    apwv = data(i,:,1)./w';
    % then pick
    apwv(~tv) = NaN;
    plot(apwv,'linewidth',2,'color',col(i,:));
    hold on;
    xlim([1 inf]);
    ylabel('Validation accuracy per weight'); xlabel('Epoch'); xticks([]);
    b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 14; box off;
end
c = colorbar; caxis([parameters{group}(ne(1),2) parameters{group}(ne(end),2)]);
c.Label.String = 'Regularisation';
sgtitle(sprintf('>%g performance, %s networks',100*thr,network_labels{group}));

% plot the weight space
idx = data>thr;
pw = [];
for i = 1:length(ne);
    pw(i,:) = sum(abs(squeeze(net_data(i,2:end,:,:))),[2 3]);
end
h = figure; h.Position = [100 100 1800 350];
subplot(1,4,2);
% interpolate
pw2 = interp2(pw,5);
a = imagesc(pw); c = colorbar; c.Label.String = 'Total weight'; c.Label.FontName = 'Arial'; c.TickDirection = 'out';
xlabel('Training'); ylabel('Regularisation'); set(gca,'YTickLabels',round(ps,3)); box off; 
b = gca; b.TickDir = 'out'; b.FontSize = 12; b.FontName = 'Arial';
% plot the accuracy space
subplot(1,4,1);
bb = 100*squeeze(data(:,:,2));
% interpolate
bb2 = interp2(bb,5);
a = imagesc(bb2); c = colorbar; c.Label.String = 'Accuracy (%)'; c.Label.FontName = 'Arial'; c.TickDirection = 'out';
xlabel('Training'); ylabel('Regularisation'); set(gca,'YTickLabels',round(ps,3)); box off; 
b = gca; b.TickDir = 'out'; b.FontSize = 12; b.FontName = 'Arial';
% plot the accuracy per cost space
apw = [];
for epoch = 1:10;
    for net = 1:length(ne);
        apw(net,epoch) = bb(net,epoch)/pw(net,epoch);
    end
end
subplot(1,4,3);
% interpolate
apw2 = interp2(apw,5);
a = imagesc(apw); c = colorbar; c.Label.String = 'Accuracy (%) per weight'; c.Label.FontName = 'Arial'; c.TickDirection = 'out';
xlabel('Training'); ylabel('Regularisation'); set(gca,'YTickLabels',round(ps,3)); box off; 
b = gca; b.TickDir = 'out'; b.FontSize = 12; b.FontName = 'Arial';
% plot the filtered accuracy per cost space
subplot(1,4,4);
fapw = apw; fapw(bb(2:end,:)<100*thr)=NaN;
% interpolate
fapw2 = interp2(fapw,5);
a = imagesc(fapw); c = colorbar; c.Label.String = sprintf('Accuracy (%%) per weight (>%g%%)',thr*100); c.Label.FontName = 'Arial'; c.TickDirection = 'out';
xlabel('Training'); ylabel('Regularisation'); set(gca,'YTickLabels',round(ps,3)); box off; 
b = gca; b.TickDir = 'out'; b.FontSize = 12; b.FontName = 'Arial';
colormap(flip(pink));
% display the optimal
[m,y] = max(fapw);

% plot the border of the accuracy per weight matrix
figure; 
surf(apw,'edgecolor','none','facealpha',.5);
box off; grid off;
xlim([1,size(apw,2)]); 
b = gca; b.TickDir = 'out'; b.FontSize = 14; b.FontName = 'Arial';
xlabel('Epoch'); ylabel('Regularisation'); zlabel('Accuracy per weight');

% histogram of high accuracy per weight
epoch = 6;
y = fapw(:,epoch); 
x = find(~isnan(y));
y = y(~isnan(y));
figure; 
u = scatterhist(x,y,...
    'color',[.4 .4 .8],...
    'marker','.',...
    'markersize',10,...
    'kernel','on',...
    'direction','out'); 
box off;
xlabel('Regularisation'); 
ylabel('Accuracy (%) per weight');
b = gca; 
b.TickDir = 'out';
b.FontName = 'Arial';
b.FontSize = 14;

%% explore the local, global and tf statistics of a single group

% set the group to compute
group = 18;
% set what networks to compute in the group
ns = [1:1:101]; 
ns(1)=1;
%ns = 1:10;
% display if indexed outside of network range
if max(ns)>nnets(group);
    disp('Warning! Will fail as aiming to index outside of the number of available networks.');
end
% set the number of local and global measures calculated
nlmeasures = 6;
ngmeasures = 11;
% set number of permutations for small worldness
nperm = 1000;
% binarisation proportion threshold for small worldness
thr = 0.1;
% get network 
u = ann_nets{group};
% global statistics labels
global_label = string(...
    {'total strength',...
    'total weighted edge length',...
    'global efficiency',...
    'homophily per weight',...
    'modularity',...
    'efficiency per weight',...
    'corr(weight,distance)',...
    'small worldness',...
    'SWP',...
    'SWP deltaC',...
    'SWP deltaL'});
% local statistics labels
local_label = string(...
    {'strength',...
    'clustering',...
    'betweenness',...
    'weighted edge length',...
    'communicability',...
    'matching'});
% initialise
if ismember(group,notime) % no training
    local_statistics = zeros(length(ns),nnode,nlmeasures);
    global_statistics = zeros(length(ns),ngmeasures);
    topological_organization = zeros(length(ns),nlmeasures,nlmeasures);
else % there is training
    local_statistics = zeros(length(ns),size(u,2),nnode,nlmeasures);
    global_statistics = zeros(length(ns),size(u,2),ngmeasures);
    topological_organization = zeros(length(ns),size(u,2),nlmeasures,nlmeasures);
end
% loop over the groups and compute global statistics
display('computing local, tf and global statistics...');
% loop over the nets
for i = 1:length(ns);
    % take network if we have no training data
    if ismember(group,notime);
        % take network
        na = squeeze(ann_nets{group}(ns(i),:,:));
        % optional step to make all connections positive
        n = abs(na);
        % rescale [0 1]
        % n = rescale(n);
        % take binarisation
        a = threshold_proportional(n,thr); a = double(a>0);
        % compute local statistics
        local_statistics(i,:,1) = strengths_und(n)';
        local_statistics(i,:,2) = clustering_coef_wu(n);
        local_statistics(i,:,3) = betweenness_wei(n);
        local_statistics(i,:,4) = sum(n*Dbox)';
        local_statistics(i,:,5) = mean(expm(n))';
        local_statistics(i,:,6) = (mean((matching_ind(a)+matching_ind(a)'))); % binarization
        % compute tf
        topological_organization(i,:,:) = corr(squeeze(local_statistics(i,:,:)));
        % compute global statistics
        global_statistics(i,1) = sum(n,'all');
        global_statistics(i,2) = mean(n*Dbox,'all');
        global_statistics(i,3) = efficiency_wei(n);
        global_statistics(i,4) = mean(squeeze(local_statistics(i,:,6)));
        [~,global_statistics(i,5)] = modularity_und(a); % binarization
        global_statistics(i,6) = global_statistics(i,3)./sum(n,'all');
        global_statistics(i,7) = corr(Dbox(n>0),n(n>0));
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
        global_statistics(i,8) = [clu/mean_clu_perm]/[cpl/mean_cpl_perm];
        % calcualte weighted small world propensity
        [swp dc dl] = small_world_propensity(n);
        global_statistics(i,9) = swp;
        global_statistics(i,10) = dc;
        global_statistics(i,11) = dl;
        else % get how many training steps
            train = size(u,2);
            % loop over training
            for t = 1:train;
                if group == 5;
                    D = Dsheet;
                else
                    D = Dbox;
                end
                % get network
                n = squeeze(ann_nets{group}(ns(i),t,:,:));
                % optional step to make all connections positive
                n = abs(n);
                % rescale [0 1]
                % n = rescale(n);
                % take binarisation
                a = threshold_proportional(n,thr); a = double(a>0);
                %{
                % compute local statistics
                local_statistics(i,t,:,1) = strengths_und(n)';
                local_statistics(i,t,:,2) = clustering_coef_wu(n);
                local_statistics(i,t,:,3) = betweenness_wei(n);
                local_statistics(i,t,:,4) = sum(n*Dbox)';
                local_statistics(i,t,:,5) = mean(expm(n))';
                local_statistics(i,t,:,6) = mean((matching_ind(a)+matching_ind(a)'))'; % binarisation
                %}
                % compute tf
                % topological_organization(i,t,:,:) = corr(squeeze(local_statistics(i,t,:,:)));
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
                for j = 1:nperm;
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
                % calcualte weighted small world propensity
                [swp dc dl] = small_world_propensity(n);
                global_statistics(i,t,9) = swp;
                global_statistics(i,t,10) = dc;
                global_statistics(i,t,11) = dl;
                % display training point
                display(sprintf('group %g, network %g, training point %g computed...',group,ns(i),t));
            end
        end
        % display network
        display(sprintf('group %g, network %g computed',group,ns(i)));
    end
    % display group
    disp(sprintf('group %g group statistics computed',group));
    
    % save
    se_swc_5_global_statistics_extra = global_statistics;
    save('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_5_global_statistics_extra.mat','se_swc_5_global_statistics_extra','-v7.3');
%% set the global statistics to view    

global_statistics = se_swc_4_1k_global_statistics_extra;

%% visualise network properties for groups of networks

% set some limits for visualising each statistic property
limits = [0 1000];

% visualise network properties for groups of networks with no time
if ismember(group,notime);
% form a color pallete
col = parula(length(ns));
% plot global statitics over time
h = figure; h.Position = [100 100 1200 500];
for measure = 1:ngmeasures;
    o = squeeze(global_statistics(:,measure))';
    subplot(2,4,measure);
    for i = 1:length(ns);
        bar(o);
        hold on;
    end
    ylabel(global_label(measure));
    b = gca; b.TickDir = 'out';
    xlabel('network');
    grid on;
end
sgtitle(network_labels{group});
% plot topological fingerprints across all networks 
h = figure; h.Position = [100 100 1400 140];
for net = 1:length(ns);
    subplot(1,length(ns),net);
    imagesc(squeeze(topological_organization(net,:,:)));
    caxis([-1 1]);
    sgtitle(sprintf('topological organisation %s',network_labels{group}));
    xticks([]); yticks([]);
    title(sprintf('network %g',net));
end
else % for those with time
% form a color pallete
col = parula(length(ns));
% plot global statitics over time
h = figure; h.Position = [100 100 1200 500];
for measure = 1:ngmeasures;
    o = squeeze(global_statistics(:,:,measure))';
    subplot(2,4,measure);
    for i = 1:length(ns);
        plot(o(:,i),...
            'linewidth',3,...
            'color',col(i,:))
        hold on;
    end
    ylabel(global_label(measure),'linewidth',10);
    b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 14;
    xlabel('time');
    grid on;
end
c = colorbar; caxis([parameters{group}(ne(1),2) parameters{group}(ne(end),2)]);
c.Label.String = 'regularisation';
sgtitle(sprintf('%s global network changes with training',network_labels{group}));
% plot topological fingerprints of a select network in the current ns
net = 10;
% compute time
tt = size(topological_organization,2);
h = figure; h.Position = [100 100 1400 140];
for t = 1:tt;
    subplot(1,tt,t);
    imagesc(squeeze(topological_organization(net,t,:,:)));
    caxis([-1 1]);
    sgtitle(sprintf('topological organisation %s network %g regularisation %.3g',network_labels{group},ns(net),parameters{group}(ns(net),2)));
    b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 14;
    xticks([]); yticks([]);
    if t==1;
        yticks([1:nlmeasures]);
        yticklabels(local_label)
    end
    title(sprintf('t=%g',t));
end
% plot basic distributions at a time point for this network too
time = 6;
% set stats
stats = [1 2 3 4 5 6];
h = figure; h.Position = [100 100 1400 180];
for stat = 1:length(stats);
    subplot(1,length(stats),stat);
    histogram(local_statistics(net,time,:,stat));
    xlabel(local_label(stat)); ylabel('Frequency');
    sgtitle(sprintf('topolgoical distributions %s network %g regularisation %.3g time=%g',...
    network_labels{group},ns(net),parameters{group}(ns(net),2),time));
    b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 14;
end
end
%% relationship between network changes and accuracy
% set scalar which plots larger regularisation as bigger
rs = 300;
% take nets that of which statistics were computed
acc_pick = squeeze(acc_nets{group}(ns,:,:));
h = figure; h.Position = [100 100 800 700];
% collapse over time points
for measure = 1:ngmeasures;
    subplot(4,2,measure);
    x = global_statistics(:,:,measure); x(:,1) = [];
    y = acc_pick(:,:,2); y(:,1) = [];
    for ni = 1:length(ns);
        u = scatter(y(ni,:),x(ni,:),0.5+rs*parameters{group}(ns(ni),2),'.','markeredgecolor','r'); % points which are more regularised are bigger
        hold on;
    end
    xlim([0.2 1]); xlabel('Accuracy');
    ylabel(global_label(measure));
    b = gca; b.TickDir = 'out'; b.FontSize = 10; b.FontName = 'Arial';
end
sgtitle(sprintf('%s relationship between network measures and accuracy (large points = large regularisation)',...
    network_labels{group}));
%% compute the connection density weight distribution
% set group
group = 5;
% set network
net = 46;
% set time
time = 6;
% set threshold vector
thrvec = [0:0.01:1];
% compute the density and weights at these threhsolds
xa = []; xb = [];
for i = 1:length(thrvec);
    thr = thrvec(i);
    % form a network
    a = abs(squeeze(ann_nets{group}(net,time,:,:)));
    b = threshold_proportional(a,thr);
    c = b + b';
    d = double(c>0);
    % compute density and weight
    xa(i) = (sum(triu(b,1),'all'))./(sum(triu(a,1),'all')); % total percentage weight
    xb(i) = density_und(d); % total density
end
% plot against each other
h = figure; h.Position = [100 100 800 600];
subplot(1,2,1);
scatter(100*xb,100*xa,80,thrvec,'marker','.'); 
xlabel('Connection inc. self (%)'); ylabel('Weight (%)');
c = colorbar; c.Label.String = 'Proportional threshold';
b = gca; b.TickDir = 'out'; b.FontSize = 14; b.FontName = 'Arial';
xticks([0:10:100]); yticks([0:10:100]);
% plot just the ratio
subplot(1,2,2);
scatter(thrvec,[(100*xa)./(100*xb)],80,'marker','.');
xlabel('Proportional threshold'); ylabel('Weight:Connection Ratio');
xticks([0:0.1:1]);
b = gca; b.TickDir = 'out'; b.FontSize = 14; b.FontName = 'Arial';
% display some key thresholds
thrkey = [0:0.01:0.3];
ind = ismember(thrvec,thrkey);
T = table(thrvec(ind)',xa(ind)',xb(ind)',[xa(ind)./xb(ind)]',...
    'VariableNames',{'Proportional thresholds','Weight %','% Connections inc. self %','Ratio'});
disp(T);
%% visualise the effect of binarisation
% set group
group = 5;
% set network
net = 46;
% set time
time = 6;
% set threshold
thr = 0.05;
% form a network
a = abs(squeeze(ann_nets{group}(net,time,:,:)));
b = threshold_proportional(a,thr);
c = b + b';
d = double(c>0);
% visualise
h = figure; h.Position = [100 100 1000 400];
subplot(1,2,1); 
imagesc(a); 
b = gca; b.TickDir = 'out'; b.FontSize = 14; b.FontName = 'Arial';
subplot(1,2,2); 
imagesc(d);
b = gca; b.TickDir = 'out'; b.FontSize = 14; b.FontName = 'Arial';
sgtitle(sprintf('%s network %g t=%g %g threshold',network_labels(group),net,time,thr));
%% binarise networks to prepare generative models
% set group
group = 14;
% set network
net = 290;
% set epoch
epoch = 6;
% set threshold
thr = 0.1;
% form a network
a = abs(squeeze(ann_nets{group}(net,epoch+1,:,:)));
b = threshold_proportional(a,thr);
c = b + b';
d = double(c>0);
% set model type
modeltypes = string({'sptl',...
    'neighbors','matching',...
    'clu-avg','clu-min','clu-max','clu-diff','clu-prod',...
    'deg-avg','deg-min','deg-max','deg-diff','deg-prod'});
% set nuber of models
nmodels = 13;
% set whether the model is based on powerlaw or exponentials
modelvar = [{'powerlaw'},{'powerlaw'}];
% set eta and gamma limits
eta = [-3,3];
gam = [-3,3];
% parameters related to the optimization
pow = 2; % severity
nlvls = 5; % number of steps
nreps = 200; % number of repetitions/samples per step
% set target network to this pipeline
Atgt = d;
% take cost
D = Dbox;
% take the nnode
nnode = size(D,1);
% get key observed statistics
x = cell(4,1);
x{1} = sum(Atgt,2);
x{2} = clustering_coef_bu(Atgt);
x{3} = betweenness_bin(Atgt)';
x{4} = D(triu(Atgt,1) > 0);
% set seed
Aseed = zeros(nnode,nnode);
% set number of connections
m = nnz(Atgt)/2;
% initialise
output = struct;
output.energy = zeros(nmodels,nlvls.*nreps);
output.ks = zeros(nmodels,nlvls.*nreps,4);
output.networks = zeros(nmodels,m,nlvls.*nreps);
output.parameters = zeros(nmodels,nlvls.*nreps,2);
% nnode
n = size(Atgt,1);
%% run the generative model
for model = 1:nmodels;
    % print text
    disp(sprintf('running network %g model %s...',net,modeltypes(model)));
    % run the model
    [E,K,N,P] = fcn_sample_networks(Atgt,Aseed,D,m,modeltypes(model),modelvar,nreps,nlvls,eta,gam,pow);
    % store the output
    output.energy(model,:) = E;
    output.ks(model,:,:) = K;
    output.networks(model,:,:) = N;
    output.parameters(model,:,:) = P;
end
se_swc_4_1k_290_generative = struct;
se_swc_4_1k_290_generative.energy = output.energy;
se_swc_4_1k_290_generative.ks = output.ks;
se_swc_4_1k_290_generative.networks = output.networks;
se_swc_4_1k_290_generative.parameters = output.parameters;
se_swc_4_1k_290_generative.proportional_threshold = thr;
se_swc_4_1k_290_generative.voronoi.eta = eta;
se_swc_4_1k_290_generative.voronoi.gam = gam;
se_swc_4_1k_290_generative.voronoi.pow = pow;
se_swc_4_1k_290_generative.voronoi.nlvls = nlvls;
se_swc_4_1k_290_generative.voronoi.nreps = nreps;
save('/imaging/astle/users/da04/PhD/ann_jascha/data/se_swc_4_1k_290_generative.mat','se_swc_4_1k_290_generative','-v7.3');
%% quick visual
% plot minimum energy
h = figure; h.Position = [100 100 600 500];
bar(min(output.energy'));
xlabel('Generative model'); xticklabels(modeltypes); xtickangle(45);
ylabel('Energy');
ylim([0 1]); 
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 16;
box off; b.TickLength = [.02 .02];
sgtitle(sprintf('%s: network %g (%g threshold), %g parameters eta limits [%g %g] gam limits [%g %g]',...
    network_labels(group),net,thr,nreps*nlvls,eta(1),eta(2),gam(1),gam(2)));
% plot parameter space
model = 3;
figure;
h = scatter(squeeze(output.parameters(model,:,1)),squeeze(output.parameters(model,:,2)),60,output.energy(model,:));
xlabel('\eta'); ylabel('\gamma');
caxis([0 1]); sgtitle(sprintf('%s generative model - %s regularisation %g epoch=%g',...
    modeltypes(model),network_labels{group},parameters{group}(net,2),epoch));
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 16; box off; b.TickLength = [.02 .02];
c = colorbar; c.Label.String = 'Energy';