%% voronoi tesselation procedure
% written by danyal akarca, university of cambridge, 2021
function output = run_li_iii_generative_models(threshold,network,epoch,el,eu,gl,gu);
%% add relevant paths and requirements
addpath('/imaging/shared/users/ja02/CBUActors/SpatialRNNs/toolboxes/2019_03_03_BCT/');
addpath('/imaging/shared/users/ja02/CBUActors/SpatialRNNs/toolboxes/voronoi');
addpath('/imaging/astle/users/da04/PhD/toolboxes/sort_nat');
% change directory
cd('/imaging/shared/users/ja02/CBUActors/SpatialRNNs/Results/mazeGenI_L1_mtIII_1k');
% set the savedir
sdir = '/imaging/astle/users/da04/PhD/ann_jascha/data/l1_iii_generative_models';
%% load relevant rnn data
%%% load l1_3_1k %%%
% set the number of networks in here
nnet = 1001;
% set number of nodes
nnode = 100;
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
nets = zeros(nnet,tt,nnode,nnode);
wei = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        nets(i,t,:,:) = Training_History{t+1,1};
        wei(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('l1_3_1k network %g loaded',i-1));
end
%% binarise networks ready for use according to the proportional threshold
% initialise
Abin = zeros(nnet,tt,nnode,nnode);
% for the binarised set of networks
for net = 1:nnet;
    for t = 1:tt;
        a = abs(squeeze(nets(net,t,:,:)));
        b = threshold_proportional(a,threshold);
        c = b + b';
        Abin(net,t,:,:) = double(c>0);
    end
    % display
    disp(sprintf('l1_3_1k network %g threhsolded at %g',net-1,threshold));
end
%% initialise model information
% set number of models
nmodels = 13;
% set model type
modeltypes = string({'sptl',...
    'neighbors','matching',...
    'clu-avg','clu-min','clu-max','clu-diff','clu-prod',...
    'deg-avg','deg-min','deg-max','deg-diff','deg-prod'});
% set whether the model is based on powerlaw or exponentials
modelvar = [{'powerlaw'},{'powerlaw'}];
%% set up parameters for the tesselation
% set eta and gamma limits
eta = [el,eu];
gam = [gl,gu];
% parameters related to the optimization
pow = 2; % severity
nlvls = 5; % number of steps
nreps = 200; % number of repetitions/samples per step
%% run the tesselation procedure
% set the target network
Atgt = squeeze(Abin(network,epoch+1,:,:));
% take the euclidean (loaded earlier)
D = Cost_Matrix;
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
for model = 1:nmodels;
    % print text
    disp(sprintf('running l1_3_1k_rnn_thr%g_net%g_epo%g model %s...',threshold,network,epoch,modeltypes(model)));
    % run the model
    [E,K,N,P] = fcn_sample_networks(Atgt,Aseed,D,m,modeltypes(model),modelvar,nreps,nlvls,eta,gam,pow);
    % store the output
    output.energy(model,:) = E;
    output.ks(model,:,:) = K;
    output.networks(model,:,:) = N;
    output.parameters(model,:,:) = P;
end
%% save the subject
% change directory
cd(sdir);
% save file
save(sprintf('l1_3_1k_thr%g_net%g_epo%g_generative_model.mat',threshold,network,epoch),'output','-v7.3');
end
