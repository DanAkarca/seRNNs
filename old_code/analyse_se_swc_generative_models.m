%% generative modelling findings of se swc
% written by danyal akarca
%% set pre-requisites
clear; clc;
% change directory to the se swc folder
cd('/imaging/shared/users/ja02/CBUActors/SpatialRNNs/generative_models/se_swc');
% load observed data
% addpaths
addpath('/imaging/astle/users/da04/PhD/toolboxes/2019_03_03_BCT');
addpath('/imaging/astle/users/da04/PhD/toolboxes/MatlabToolbox-master');
addpath('/imaging/astle/users/da04/PhD/toolboxes/computeCohen');
addpath('/imaging/astle/users/da04/PhD/toolboxes/bluewhitered');
addpath('/imaging/astle/users/da04/PhD/toolboxes/stdshade');
addpath('/imaging/astle/users/da04/PhD/hd_gnm_generative_models/code/functions/');
% define model types
model_labels = string({'sptl',...
    'neighbors','matching',...
    'clu-avg','clu-min','clu-max','clu-diff','clu-prod',...
    'deg-avg','deg-min','deg-max','deg-diff','deg-prod'});
eqn = string({'KS_k','KS_c','KS_b','KS_d'});
%% set hyperparameters of the dataset
% number of models
nmodels = 13;
% define thresholds
thresholds = [0.01 0.05 0.10];
% number of thresholds
nthresholds = length(thresholds);
% threshold labels
threshold_labels = string({'1%','5%','10%'});
% define networks
nets = 1:70;
% number of networks
nnets = length(nets);
% network labels
network_labels = string(nets);
% time points
times = 2:2:10;
% number of time points
ntimes = length(times);
% time labels
time_labels = [2,4,6,8,10];
%% form a loop index of the cultures
% initialise
index = [];
step = 1;
% loop through and visualise
for threshold = 1:nthresholds;
    for network = 1:nnets;
        for time = 1:ntimes
            % assign the network indices
            index(step,1) = threshold;
            index(step,2) = network;
            index(step,3) = time;
            step = step + 1;
        end
    end
end
% form the new nsamp
nsamp = size(index,1);
%% load generative model data
% list all generative output files
h = dir('*.mat');
list = string({h.name})';
% structure the data that exist 
exist = []; step = 1;
for k = 1:size(list,1);
    % get the string
    u = list(k);
    % form the character
    c = char(u);
    % loop through the index matrix and find where there is a match
    for kk = 1:size(index,1);
        if strcmp(c,sprintf('se_swc_%g_%g_%g_generative_model.mat',...
                thresholds(index(kk,1)),nets(index(kk,2)),times(index(kk,3))));
            exist(step,:) = [index(kk,1),index(kk,2),index(kk,3)];
            step = step + 1;
        end
    end
end
% sort this into a sorted array rather than the list order
exist = sortrows(exist,[1 2 3]);
% compute number of existing networks
nexist = size(exist,1);
% get indices of existing in index
existi = ismember(index,exist,'row');
% update all observed networks and costs (to do)
% initialise
energy_sample = zeros(nexist,13,10000);
ks_sample = zeros(nexist,13,10000,4);
networks_sample = cell(nexist,1);
parameters_sample = zeros(nexist,13,10000,2);
errors = zeros(nexist,1);
% loop over pipelines, cultures and divs
step = 1;
for k = 1:nexist;   
    % load this network's generative model output
    load(sprintf('se_swc_%g_%g_%g_generative_model.mat',thresholds(exist(k,1)),nets(exist(k,2)),times(exist(k,3))));
    % assign
    energy_sample(step,:,:) = output.energy;
    ks_sample(step,:,:,:) = output.ks;
    networks_sample{step} = output.networks;
    parameters_sample(step,:,:,:) = output.parameters;
    % clear the variable
    clear output
    % display
    disp(sprintf('se swc network_%g_%g_%g loaded; threshold %s network %s t=%g',...
        exist(k,1),exist(k,2),exist(k,3),...
        threshold_labels(exist(k,1)),network_labels(exist(k,2)),time_labels(exist(k,3))));
    % upate step
    step = step + 1;
end
%% look at energy landscape for a selected rule and network
% select model and network
model = 3;
net = 5;
% take the measure
e = squeeze(energy_sample(net,model,:));
pipeline_p = squeeze(parameters_sample(net,model,:,:));
% visualise
h = figure;
if model == 1
    scatter(pipeline_p(:,1),e,100,e,'.');
    xlabel('\eta'); ylabel('Energy'); 
    ylim([0 1]);
    caxis([0 1]); c = colorbar; c.Label.String = 'Energy';
else
    % plot the energy landscape
    scatter(pipeline_p(:,1),pipeline_p(:,2),100,e,'.'); 
    xlabel('\eta'); ylabel('\gamma'); 
    caxis([0 1]); c = colorbar; c.Label.String = 'Energy';
end
b = gca; b.TickDir = 'out'; b.FontSize = 14; b.FontName = 'Arial';
sgtitle(sprintf('network %g. %s model, %s threshold. t=%g',...
    net,model_labels(model),threshold_labels(exist(net,2)),time_labels(exist(net,3))));
%% plot energy landscape over training
% select model and network
model = 9;
net = 3; % this network is the specific one in the set
% warning 
if ~ismember(net,exist(:,2));
    disp('Warning! Network is out of range of available networks.');
end
% find extant networks over time
ind = exist(:,2)==net;
% get these indices
fi = find(ind);
% find the network information
u = exist(ind,:);
% specially the times
tt = u(:,3);
% look at all the times
te = [1:ntimes]; 
% binary extant
ex = ismember(te,tt);
% actual extant
ax = te(ex);
% visualise energy landscapes over time
step = 0;
h = figure; h.Position = [100 100 1200 200];
for i = 1:ntimes;
    if ex(i); % if exists
        % keep track
        step = step+1;
        % subplots
        subplot(1,ntimes,te(i));
        % index the network 
        in = fi(step);
        % take the measures
        e = []; p = [];
        e = squeeze(energy_sample(in,model,:));
        pipeline_p = squeeze(parameters_sample(in,model,:,:));
        % visualise
        if model == 1
            scatter(pipeline_p(:,1),e,100,e,'.');
            xlabel('\eta'); ylabel('Energy'); ylim([0 1]); caxis([0 1]); c = colorbar; c.Label.String = 'Energy';
        else
            % plot the energy landscape
            scatter(pipeline_p(:,1),pipeline_p(:,2),100,e,'.'); 
            xlabel('\eta'); ylabel('\gamma'); 
            caxis([0 1]); c = colorbar; c.Label.String = 'Energy';
        end
        b = gca; b.TickDir = 'out'; b.FontSize = 14; b.FontName = 'Arial';
    end
    title(sprintf('t=%g',time_labels(te(i))));
end
sgtitle(sprintf('network %g. %s model, %s threshold.',...
    net,model_labels(model),threshold_labels(exist(net,2))));
%% look at energy landscape for all networks by group
% select model
model = 9;
% select the criteria for selection
criteria = exist(:,3)==2;
criteria_ind = find(criteria);
% take the measure
e = squeeze(energy_sample(:,model,:));
p = squeeze(parameters_sample(:,model,:,:));
% breakdown by group to visualise, if wished
e_select = e(:,:);
p_select = p(:,:,:);
nsamp_select = size(e_select,1);
% visualise
if model == 1
    h = figure;
    h.Position = [100 100 600 300];
    eta = squeeze(p(:,:,1));
    m = parula(10000);
    for net = 1:nsamp_select;
        % get colours
        pipeline_d = round(e_select(net,:) .* 10000);
        col = m(pipeline_d,:);
        % plot
        scatter(eta(net,:),e_select(net,:),20,col,'.'); ylabel('energy'); xlabel('eta'); ylim([0 1]);
        b = gca; b.TickDir = 'out';
        b.FontName = 'Arial';
        yticks([]); xticks([]); hold on;
    end
else
% plot the energy landscape
h = figure;
h.Position = [100 100 500 400];
for net = 1:nsamp_select
    scatter(squeeze(p_select(net,:,1)),squeeze(p_select(net,:,2)),20,e_select(net,:),'.'); hold on;
    %xlabel('eta'); ylabel('gamma');
end
caxis([0 1]);
c = colorbar; c.Label.String = 'energy';
b = gca; b.TickDir = 'out';
xticks([]); yticks([]);
end
% set any limits
xlim([-4 0]); ylim([0 4]);
%% look at ks landscape for all networks
% select model
model = 3;
% take the measure
ks = squeeze(ks_sample(:,model,:,:));
pipeline_p = squeeze(parameters_sample(:,model,:,:));
% plot the energy landscape
h = figure;
h.Position = [100 100 1400 220];
for net = 1:nsamp
    for j = 1:4;
        subplot(1,4,j);
        scatter(squeeze(pipeline_p(net,:,1)),squeeze(pipeline_p(net,:,2)),20,ks(net,:,j),'.'); hold on;
        xlabel('eta'); ylabel('gamma'); title(eqn(j));
        caxis([0 1]); c = colorbar;
    end
end
%% compute summary statistics over the sample
% define the top n parameters
nset = [1 10 50];
% initialise a matrix to keep the data, which will be subject by n by parameters
top_e = cell(length(nset),1);
top_e_mean = zeros(length(nset),13,nexist);
top_p = cell(length(nset),1);
top_p_mean = zeros(length(nset),13,nexist,2);
% compute the minimum
% run just top 2 sets for memory
for no = 1:3;
    % take the actual amount of top performing parameters
    n = nset(no);
    % 
    for model = 1:13;
        % take energies for this model
        pipeline_d = squeeze(energy_sample(:,model,:))';
        % rank them for each subject
        [v i] = sort(pipeline_d);
        % take top n energies and their indices
        n_e = v(1:n,:);
        n_i = i(1:n,:);
        % take the corresponding parameters
        u = zeros(nexist,n,2);
        for s = 1:nexist;
            % keep parameters
            u(s,:,:) = squeeze(parameters_sample(s,model,n_i(:,s),:));
        end
        % if top parameter only
        if n == 1
            % squeeze the matrices
            u = squeeze(u);
            % assign
            top_e{no}(model,:) = n_e';
            top_p{no}(model,:,:) = u;
            % and assign it to the mean
            top_e_mean(no,model,:) = n_e';
            top_p_mean(no,model,:,:) = u;
            
        else
            top_e{no}(model,:,:) = n_e';
            top_p{no}(model,:,:,:) = u;
            % keep a mean value too
            top_e_mean(no,model,:) = squeeze(mean(n_e',2));
            top_p_mean(no,model,:,:) = squeeze(mean(u,2));
        end
    end
    disp(sprintf('set %g of %g complete',no,length(nset)));
end
%% count what drives the energy 
% specify the model
model = 3;
% take the data
ks_data = squeeze(ks_sample(:,model,:,:));
% initialise
driver = [];
% loop over networks
for net = 1:nexist;
    % find the max ks statistics for this network
    [v i] = max(squeeze(ks_data(net,:,:))');
    % group data
    driver(net,:) = [sum(i==1),sum(i==2),sum(i==3),sum(i==4)];,
end
% form a percentage
driver = driver ./ 20000 * 100;
% visualise
figure;
iosr.statistics.boxPlot(driver,...
    'showViolin',logical(0),...
    'theme','colorall',...
    'symbolMarker','x',...
    'showScatter',logical(1),...
    'scatterColor',[.5 .5 .5],...
    'scatterAlpha',0.5,...
    'symbolColor',[.5 .5 .5],...
    'boxColor','k',...
    'boxAlpha',0.15); 
ylim([0 100]);
b = gca; b.TickDir = 'out'; b.FontName = 'Arial'; b.FontSize = 14;
xticklabels({'degree','clustering','betweenness','edge length'});
ylabel('max(KS)'); yticklabels({'0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'});
%% visualise summary statistics across a specific group
% select which set of nset to view
set = 1;
% plot over pipelines
e_select = nan(nmodels,nexist,nthresholds);
for threshold = 1:nthresholds;
    criteria = exist(:,1)==threshold;
    % take specific energy values
    k = squeeze(top_e_mean(set,:,criteria));
    % keep
    e_select(:,1:size(k,2),threshold) = k;
end
% set the new model order
i = [2:13 1];
e_select = e_select(i,:,:);
% permute the order
e_select = permute(e_select,[2 3 1]);
% iosr boxplot
h = figure;
h.Position = [100 100 600 300];
iosr.statistics.boxPlot(e_select(i,:,:),...
    'showViolin',logical(0),...
    'theme','colorall',...
    'symbolMarker','x',...
    'symbolColor','k',...
    'boxColor',{'r','r','g','g','g','g','g','b','b','b','b','b','y'},...
    'boxAlpha',0.2);
ylim([0 1]); ylabel('Energy'); xlabel('Threshold');
xticklabels(threshold_labels);
b = gca; 
b.XAxis.TickDirection = 'out';
b.YAxis.TickDirection = 'out';
b.FontName = 'Arial';
b.FontSize = 14;
% plot over divs
e_select = nan(nmodels,nexist,ntimes);
for time = 1:ntimes;
    criteria = exist(:,3)==time;
    % take specific energy values
    k = squeeze(top_e_mean(set,:,criteria));
    % keep
    e_select(:,1:size(k,2),time) = k;
end
% set the new order
i = [2:13 1];
e_select = e_select(i,:,:);
% permute the model order
e_select = permute(e_select,[2 3 1]);
% iosr boxplot
h = figure;
h.Position = [100 100 600 300];
iosr.statistics.boxPlot(e_select(i,:,:),...
    'showViolin',logical(0),...
    'theme','colorall',...
    'symbolMarker','x',...
    'symbolColor','k',...
    'boxColor',{'r','r','g','g','g','g','g','b','b','b','b','b','y'},...
    'boxAlpha',0.2);
ylim([0 1]); ylabel('Energy'); xlabel('Training time');
xticklabels(time_labels);
b = gca; 
b.XAxis.TickDirection = 'out';
b.YAxis.TickDirection = 'out';
b.FontName = 'Arial';
b.FontSize = 14;
%% visualise summary statistics between specific ranges of regularisation
% set range of regularisation
reg = [50 69];
% find these networks
netrange = exist(:,2)>=reg(1)&exist(:,2)<=reg(2);
% plot over divs
e_select = nan(nmodels,nexist,ntimes);
for time = 1:ntimes;
    criteria = netrange & exist(:,3)==time;
    % take specific energy values
    k = squeeze(top_e_mean(set,:,criteria));
    % keep
    e_select(:,1:size(k,2),time) = k;
end
% set the new order
i = [2:13 1];
e_select = e_select(i,:,:);
% permute the model order
e_select = permute(e_select,[2 3 1]);
% iosr boxplot
h = figure;
h.Position = [100 100 600 300];
iosr.statistics.boxPlot(e_select(i,:,:),...
    'showViolin',logical(0),...
    'theme','colorall',...
    'symbolMarker','x',...
    'symbolColor','k',...
    'boxColor',{'r','r','g','g','g','g','g','b','b','b','b','b','y'},...
    'boxAlpha',0.2);
ylim([0 1]); ylabel('Energy'); xlabel('Training time');
xticklabels(time_labels);
b = gca; 
b.XAxis.TickDirection = 'out';
b.YAxis.TickDirection = 'out';
b.FontName = 'Arial';
b.FontSize = 14;