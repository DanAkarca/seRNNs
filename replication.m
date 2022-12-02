%% explore all recurrent neural network data
% written dr danyal akarca

%% set pre-requisites and paths

% clear the workspace and command window
clear; clc;
% addpaths of prerequisites
addpath('/imaging/astle/users/da04/PhD/toolboxes/2019_03_03_BCT'); % bct
addpath('/imaging/astle/users/da04/PhD/toolboxes/MatlabToolbox-master/'); % iosr
addpath('/imaging/astle/users/da04/PhD/hd_gnm_generative_models/voronoi'); % voronoi
addpath('/imaging/astle/users/da04/PhD/toolboxes/computeCohen'); % cohen's d
addpath('/imaging/astle/users/da04/PhD/toolboxes/sort_nat'); % sorting files
addpath('/imaging/astle/users/da04/PhD/toolboxes/Colormaps/Colormaps (5)/Colormaps/'); % color palette
% set directories of the data
project = '/imaging/shared/users/ja02/CBUActors/SpatialRNNs/Results/';
cd(project);
% set paths of trained rnns
l1     = '/mazeGenI_L1_mtIII_1k'; % 1000 l1 (main manuscript)
seRNN  = '/mazeGenI_SE1_sWc_mtIV_1k'; % 1000 seRNNs (main manuscript)
Donly  = '/mazeGenI_SE1_mtII'; % spatial only variation (supplement)
Conly  = '/mazeGenI_SE1_sWcExcl_mtI'; % communicability only variation (supplement)
random = '/mazeGenI_SE1_sWc_mtIV_randI'; % random task variation (supplementary)
hard   = '/mazeGenII_SE1_sWc_mtI'; % hard task variation (supplementary)
% set labels
network_labels = string({'L1','seRNN','Donly','Conly','Random','Hard'});
% set number of nodes
nnode = 100;
% load pre-computed network statistics
load('/imaging/astle/users/da04/Postdoc/seRNNs/data/l1_neuron_statistics.mat');
load('/imaging/astle/users/da04/Postdoc/seRNNs/data/seRNN_neuron_statistics.mat');
% load pre-computed generative models
load('/imaging/astle/users/da04/Postdoc/seRNNs/data/l1_generative_models.mat');
load('/imaging/astle/users/da04/Postdoc/seRNNs/data/seRNN_generative_models.mat');

%% load all recurrent neural networks across regularisation strengths

%%% load L1 networks %%%
% change directory
cd(strcat(project,l1));
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
net_l1 = zeros(nnet,tt,nnode,nnode);
wei_l1 = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_l1 = zeros(nnet,tt,2); % acc, val acc
parameters_l1 = [];
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_l1(i,t,:,:) = Training_History{t+1,1};
        wei_l1(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_l1(i,t,:) = Training_History{t+1,5:6};
        parameters_l1(i,:) = [i Regulariser_Strength];
    end
    % display
    disp(sprintf('L1 network %g loaded',i-1));
end

%%% load seRNNs %%%
% change directory
cd(strcat(project,seRNN));
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
parameters_seRNN = [1:1001;linspace(0.001,0.3,1001)]';
% initialise
net_sRNN = zeros(nnet,tt,nnode,nnode);
wei_seRNN = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_seRNN = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_sRNN(i,t,:,:) = Training_History{t+1,1};
        wei_seRNN(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_seRNN(i,t,:) = Training_History{t+1,5:6};
        % keep one instance of the coordinates and cost matrix
        Dbox = Cost_Matrix;
        Cbox = Coordinates;
    end
    % display
    disp(sprintf('seRNN network %g loaded',i-1));
end

%%% load Donly networks %%%
% change directory
cd(strcat(project,Donly));
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
parameters_Donly = [1:101;linspace(0.00001,0.02,101)]';
% initialise
net_Donly = zeros(nnet,tt,nnode,nnode);
wei_Donly = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_Donly = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_Donly(i,t,:,:) = Training_History{t+1,1};
        wei_Donly(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_Donly(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('Donly network %g loaded',i-1));
end

%%% load Conly networks %%%
% change directory
cd(strcat(project,Conly));
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
parameters_Conly = [1:101;linspace(0.00001,0.02,101)]';
% initialise
net_Conly = zeros(nnet,tt,nnode,nnode);
wei_Conly = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_Conly = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_Conly(i,t,:,:) = Training_History{t+1,1};
        wei_Conly(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_Conly(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('Conly network %g loaded',i-1));
end

%%% load Random networks %%%
% change directory
cd(strcat(project,random));
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
parameters_random = [1:101;linspace(0.001,0.3,101)]';
% initialise
net_random = zeros(nnet,tt,nnode,nnode);
wei_random = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_random = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_random(i,t,:,:) = Training_History{t+1,1};
        wei_random(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_random(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('random network %g loaded',i-1));
end

%%% load Hard networks %%%
% change directory
cd(strcat(project,hard));
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
parameters_hard = [1:101;linspace(0.001,0.3,101)]';
% initialise
net_hard = zeros(nnet,tt,nnode,nnode);
wei_hard = zeros(nnet,tt,5); % weight count >0, >1e-7, >1e-3, mean wei, cost mean wei
acc_hard = zeros(nnet,tt,2); % acc, val acc
% loop over networks
for i = 1:nnet;
    % load data
    load(data(i));
    % loop over training
    for t = 1:tt;
        net_hard(i,t,:,:) = Training_History{t+1,1};
        wei_hard(i,t,:) = Training_History{t+1,[2:4 7:8]};
        acc_hard(i,t,:) = Training_History{t+1,5:6};
    end
    % display
    disp(sprintf('Hard network %g loaded',i-1));
end
% move back to the original directory
cd(project);

%% collect all recurrent neural network data into ordered cell arrays

% network data
ann_nets = {net_l1,net_sRNN,net_Donly,net_Conly,net_random,net_hard};
% compute the sample size of each network
nnets = [];
for i = 1:length(ann_nets);
    nnets(i) = size(ann_nets{i},1);
end
% accuracy data
acc_nets = {acc_l1,acc_seRNN,acc_Donly,acc_Conly,acc_random,acc_hard};
% parameters
parameters = {parameters_l1,parameters_seRNN,parameters_Donly,parameters_Conly,parameters_random,parameters_hard};
% display to see what index corresponds to what set
disp(network_labels');

%% visualise a hidden layer: Figure 1 & 2

% set group of networks e.g., 2
group = 2;
% set network e.g., 50 (low), 290 (moderate), 700 (high)
net = 290;
% time point e.g., epoch 6
epoch = 6;
% marker multiplier (e.g. 1,6,10,12); 
m = 6;
% set cost coordinates
c = Cbox;
% set distance
d = Dbox;

% visualise network
gr = ann_nets{group};
% take the matrix
a = abs(squeeze(gr(net,epoch,:,:)));
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
% visualise the correlation
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
sgtitle(sprintf('%s, %g, reg %.3g, t=%g, r=%.3g, p=%.3g',...
    network_labels{group},net,parameters{group}(net,2),epoch,r,p));

%% compare the accuracy between l1 and seRNNs: Figure 1

% set epoch
epoch = 9;
% set accuracy threshold
accthr = 0.9;
% set groups
ngroups = 2;
% form statistics
global_statistics = {l1_neuron_statistics.statistics seRNN_neuron_statistics.statistics};
% set palettes
pal = [0,176,178;242,98,121;111,86,149;178,162,150]./256;

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
    err = std(a,[],2,'omitnan')./sqrt(length(a)); % standard error
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

%% compare network statistics: Figure 2

% set epoch
epoch = 9;
% set accuracy threshold
accthr = 0.9;
% set groups
ngroups = 2;
% remove first network global statistics and accuracy
global_statistics = {l1_neuron_statistics.statistics(2:end,:,:) seRNN_neuron_statistics.statistics(2:end,:,:)};
accuracy_statistics = {acc_l1(2:end,:,:) acc_seRNN(2:end,:,:)};
% form labels
global_label = l1_neuron_statistics.labels;
% set number of statistics
nstat = 4;
% set palettes
pal = [0,176,178;242,98,121;111,86,149;178,162,150]./256;
% set any regularisation limits
rem = [];
% initialise
okeep = []; akeep = [];
% visualise
h = figure; h.Position = [100 100 1600 300];
for i = 1:nstat;
    subplot(1,nstat,i);
    for group = 1:ngroups;
        % get the size of the network
        w = squeeze(global_statistics{group}(:,:,1));
        % remove 
        w(rem,:) = [];
        % continue
        w = mean(w);
        w = w(:);
        w = 1.5*w;
        % take the ordered measure and keep
        o = squeeze(global_statistics{group}(:,:,i))';
        okeep(:,:,i,group) = o';
        % remove any networks
        okeep(rem,:,:,:) = [];
        % take accuracy
        a = (accuracy_statistics{group}(:,:,2)<accthr)';
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
        ylabel(global_label(i),'linewidth',10);
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

%% plot specific epoch comparisons statistical differences: Figure 2

% set the statistics
stat = 3;
% set the epoch
epoch = 9;
% take l1 networks
x = squeeze(okeep(:,epoch+1,stat,1));
indx = squeeze(akeep(:,epoch+1,stat,1));
x(find(indx)) = [];
% take seRNN networks
y = squeeze(okeep(:,epoch+1,stat,2));
indy = squeeze(akeep(:,epoch+1,stat,2));
y(find(indy)) = [];
% plot the histogram
h = figure; h.Position = [100 100 350 300];
histogram(x,...
    'facecolor',pal(1,:),...
    'edgecolor','w'); hold on; 
histogram(y,...
    'facecolor',pal(2,:),...
    'edgecolor','w');
box off;
xlabel(global_label(stat)); ylabel('Frequency');
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 16;
% run statistical comparisons
a = nan(1000,2);
a(1:length(x),1) = x;
a(1:length(y),2) = y;
[h p] = ttest(a(:,1),a(:,2));
d = computeCohen_d(a(:,1),a(:,2));

%% visualise genreative model differences: Figure 2, Supplementary 7

% set energy, accuracy data and removal indices
energydata = seRNN_generative_models.energy;
parametersgen = seRNN_generative_models.parameters;
accuracydata = acc_seRNN;
parametersdata = parameters_seRNN;
% set any networks to remove
rem = [1:100];
% set the original indices
allnets = [100:600];
% epoch 9 as this is what the models were trained on
epoch = 9;
% set the colour pallete
vu = viridis;
palette = [vu(255,:); 1 0.1992 0.1992; vu(180,:); vu(50,:)];
% take parameters and network
m = []; ind = [];
for net = 1:length(allnets);
    data = squeeze(energydata(net,:,:))';
    [m(net,:) ind(net,:)] = min(data);
end
% get accuracy
accuracy_use = squeeze(accuracydata(allnets,epoch+1,2));
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

% visualise the model fits across the rules
u = figure; u.Position = [100 100 800 500]; 
h = iosr.statistics.boxPlot(m,...
    'theme','colorall',...
    'symbolMarker','x',...
    'symbolColor',[.5 .5 .5]);
% set colors
palettei = [1 2 2 3 3 3 3 3 4 4 4 4 4];
for i = 1:13;
    h.handles.box(i).FaceColor = palette(palettei(i),:);
end
box off; 
b = gca; b.TickDir = 'out'; 
b.FontName = 'Arial'; b.FontSize = 24;
ylabel('Model fit'); xlabel('Generative model');
ylim([0 .9]);

%% visualise supplementary generative model findings: Supplementary 8

% set energy, accuracy data and removal indices
energydata = seRNN_generative_models.energy;
parametersgen = seRNN_generative_models.parameters;
accuracydata = acc_seRNN;
parametersdata = parameters_seRNN;
% set any networks to remove
rem = [];
% set the original indices
allnets = [100:600];
% epoch 9 as this is what the models were trained on
epoch = 9;
% set the colour pallete
vu = viridis;
palette = [vu(255,:); 1 0.1992 0.1992; vu(180,:); vu(50,:)];
% take parameters and network
m = []; ind = [];
for net = 1:length(allnets);
    data = squeeze(energydata(net,:,:))';
    [m(net,:) ind(net,:)] = min(data);
end
% get accuracy
accuracy_use = squeeze(accuracydata(allnets,epoch+1,2));
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

% visualise a specific model of regularisation
% set the model
model = 3; 
% plot
h = figure; h.Position = [100 100 350 250];
reg = 1:size(m,1);
[r p] = corr(reg',m(:,model));
u = scatter(parametersdata,m(:,model),100,'o',...
    'markerfacecolor',[255 51 51]./256,...
    'markeredgecolor','w',...
    'markerfacealpha',.5);
ylabel('Model fit'); xlabel('Regularisation');
ylim([0.05 0.38]);
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 18;

% correlation matrix
reg = 1:size(m,1);
data = [reg' m];
[r p] = corr(data);
h = figure; h.Position = [100 100 700 600];
imagesc(r); caxis([-1 1]);
set(gca,'XTick',[1:14],'YTick',[1:14]);
xticklabels(['Regularisation' l1_generative_models.models]);
yticklabels({'Reg',1:13});
xtickangle(45);
colormap(magma); c = colorbar; c.Label.String = 'r';
b = gca; b.TickLength = [0 0]; b.FontName = 'Arial'; b.FontSize = 18;

% plot correlation with reg
h = figure; h.Position = [100 100 450 250]; clear bar;
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

% statistically directly compare energy between se_swc and l1
% set model
model = 3;
x = min(squeeze(l1_generative_models.energy(:,model,:)),[],2);
y = min(squeeze(seRNN_generative_models.energy(:,model,:)),[],2);
[h p] = ttest(x,y);
d = computeCohen_d(x,y);

%% compute spatial-function configuration findings: Figure 3