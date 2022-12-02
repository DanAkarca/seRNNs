%% generative modelling findings of se swc
% written by danyal akarca
%% load data for rat_primary_cortex matching
clear; clc;
% change directory to the se swc folder
cd('/imaging/shared/users/ja02/CBUActors/SpatialRNNs/generative_models/se_swc_2')
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
thresholds = [0.01];
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
time_labels = string({'2','4','6','8','10'});
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
        if strcmp(c,sprintf('se_swc_II_%g_%g_%g_generative_model.mat',...
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
    load(sprintf('se_swc_II_%g_%g_%g_generative_model.mat',thresholds(exist(k,1)),nets(exist(k,2)),times(exist(k,3))));
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
net = 7;
% take the measure
e = squeeze(energy_sample(net,model,:));
pipeline_p = squeeze(parameters_sample(net,model,:,:));
% visualise
h = figure;
if model == 1
    scatter(pipeline_p(:,1),e,100,e,'.');
    xlabel('\eta'); ylabel('energy'); 
    ylim([0 1]);
    caxis([0 1]); c = colorbar; c.Label.String = 'energy';
else
    % plot the energy landscape
    scatter(pipeline_p(:,1),pipeline_p(:,2),100,e,'.'); 
    xlabel('\eta'); ylabel('\gamma'); 
    caxis([0 1]); c = colorbar; c.Label.String = 'energy';
end
b = gca; b.TickDir = 'out';
%% look at energy landscape for all networks by group
% select model
model = 3;
% select the criteria for selection
criteria = exist(:,3)==5;
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
xlim([-7 0]); ylim([0 7]);
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
    'boxColor','c',...
    'boxAlpha',0.15); 
ylim([0 100]);
b = gca; b.TickDir = 'out';
xticklabels({'degree','clustering','betweenness','edge length'});
ylabel('max(KS)'); yticklabels({'0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'});
%% plot all
% select which set of nset to view
set = 1;
% take data and plot
e_select = squeeze(top_e_mean(set,:,:));
% visualise
h = figure;
h.Position = [100 100 600 300];
iosr.statistics.boxPlot(e_select,...
    'showViolin',logical(0),...
    'theme','colorall',...
    'symbolMarker','x',...
    'symbolColor','k',...
    'boxAlpha',0.2);
ylim([0 1]); ylabel('Energy'); xlabel('Threshold');
xticklabels(model_labels); xtickangle(45);
b = gca; 
b.XAxis.TickDirection = 'out';
b.YAxis.TickDirection = 'out';
b.FontName = 'Arial';
b.FontSize = 14;
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
iosr.statistics.boxPlot(e_select,...
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
iosr.statistics.boxPlot(e_select,...
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
%% compute energy over time for specific models or all
% model
models = [1:13];
e_pick = e_select(:,:,models);
% squeeze over all rules
e_time = permute(e_pick,[2 3 1]);
e_time = e_time(:,:)';
% anova
[p anovatab stats] = anova1(e_time);
comp = multcompare(stats);
%% group by generative rule
% take the energy values over time
% plot over divs
e_select = nan(nmodels,nexist);
for time = 1:ntimes;
    % set the crieria
    criteria = exist(:,1)==1 & exist(:,3)==time;
    % take specific energy values
    k = squeeze(top_e_mean(set,:,criteria));
    e_select(:,1:size(k,2),time) = k;
end
% set the new order
i = [2:13 1];
e_select = e_select(i,:,:);
% permute the model order
e_select = permute(e_select,[2 3 1]);
% visualise summary energy across divs across rule types: note, they've already been reordered above
e_div_rules = [];
e_div_rules(:,:,1) = squeeze(mean(e_select(:,:,[1:2]),3));
e_div_rules(:,:,2) = squeeze(mean(e_select(:,:,[3:7]),3));
e_div_rules(:,:,3) = squeeze(mean(e_select(:,:,[8:12]),3));
e_div_rules(:,:,4) = squeeze(e_select(:,:,13));
% visualise
h = figure;
h.Position = [100 100 900 450];
u = iosr.statistics.boxPlot(e_div_rules,...
    'showViolin',logical(1),...
    'showScatter',logical(1),...
    'scatterAlpha',.8,...
    'theme','colorall',...
    'symbolMarker','x',...
    'symbolColor',[.5 .5 .5],...
    'boxColor',{'r','g','b','y'},...
    'boxAlpha',0.2);
u.scatterColor = {[.5 .5 .5],[.5 .5 .5],[.5 .5 .5],[.5 .5 .5]}';
ylim([0 1]); ylabel('Energy'); xlabel('Training time');
xticklabels(time_labels);
b = gca; 
b.XAxis.TickDirection = 'out';
b.YAxis.TickDirection = 'out';
b.FontName = 'Arial';
b.FontSize = 14;
%% compute statistics between rules for each time point
% initialise
p = []; d = []; stats = {}; anovatab = {}; compare = {};
% loop over divs
for time = 1:ntimes;
    % take the data for this div
    data = squeeze(e_div_rules(:,time,:));
    % run an anova
    [p(time),anovatab{time},stats{time}] = anova1(data,{'homophily','clustering','degree','spatial'},'off');
    % run a tukey-kramer
    compare{time} = multcompare(stats{time},'display','off');
    % compute pairwise cohen d
    for i = 1:4;
        for j = 1:4;
            d(time,i,j) = computeCohen_d(data(:,i),data(:,j));
        end
    end
end
%% compute statistics between time points for each rule
% initialise
p = []; d = []; stats = {}; anovatab = {}; compare = {};
% loop over rules
for rule = 1:4
    % take the data for this rule
    data = squeeze(e_div_rules(:,:,rule));
    % run an anova
    [p(rule),anovatab{rule},stats{rule}] = anova1(data,time_labels,'off');
    % run a tukey-kramer
    compare{rule} = multcompare(stats{rule},'display','off');
    % compute pairwise cohen d
    for i = 1:3;
        for j = 1:3;
            d(rule,i,j) = computeCohen_d(data(:,i),data(:,j));
        end
    end
end
%% take local measures and compute the correlation matrix but loop over time points and models
% set pipeline 
pipeline = 1;
% set how many measures are computed
nmeasures = 6;
% set up to how many divs are considered 
ntime = 3;
% initialise correlation matrices
corr_local_observed = nan(ntime,nexist,nmeasures,nmeasures);
corr_local_simulated = nan(ntime,nexist,nmodels,nmeasures,nmeasures);
corr_local_together = nan(ntime,nexist,nmodels,nmeasures*2,nmeasures*2);
% initalise the topological organization dissimilairty
todissim = nan(ntimes,nexist,nmodels);
% labels
var_labels = string({...
    'obs degree','obs clustering','obs betweenn','obs length','obs eff','obs match',...
    'sim degree','sim clustering','sim between','sim length','sim eff','sim match'});
% loop over divs
for time = 1:ntime
    % index the networks
    index_loop = find(exist(:,1)==pipeline&exist(:,3)==time);
    % compute the sample size for this loop
    nsamp = size(index_loop,1);
    % take the observed networks for this loop
    o_networks = {}; c_networks = {};
    o_networks = exist_o_networks(index_loop);
    c_networks = exist_c_networks(index_loop);
    % take the simulation data for these networks
    E = energy_sample(index_loop,:,:);
    Ps = parameters_sample(index_loop,:,:,:);
    N = networks_sample(index_loop);    
    % compute observed local statistics for each network
    for i = 1:nsamp;
        % take the observed network
        w = o_networks{i};
        % take the cost
        d = c_networks{i};
        % compute the number of nodes
        nnode = size(w,1);
        % initalise array
        observed = zeros(nnode,nmeasures);
        % compute local observed statistics
        observed(:,1) = degrees_und(w)';
        observed(:,2) = clustering_coef_bu(w);
        observed(:,3) = betweenness_bin(w)';
        observed(:,4) = sum(w.*d)';
        observed(:,5) = efficiency_bin(w,1);
        observed(:,6) = mean(matching_ind(w)+matching_ind(w)')';
        % keep
        corr_local_observed(time,i,:,:) = corr(observed);
        for model = 1:nmodels
            % loop over models
            disp(sprintf(...
                'evaluating %g of %g: %s %s %s culture %g %s generative model...',...
                i,nsamp,type_labels(type),pipeline_labels(pipeline),div_labels(time),i,model_labels(model))); 
            % take the optimal network simulation parameters, given the set model, for this network
            e_net = squeeze(E(i,model,:));
            [~,i_low] = min(e_net);
            % take the end simulated network
            si = squeeze(N{i}(model,:,i_low));
            s = zeros(nnode);
            s(si) = 1;
            s = s + s';
            % initalise the array
            simulated = zeros(nnode,nmeasures);
            % compute local simulated statistics
            simulated(:,1) = degrees_und(s)';
            simulated(:,2) = clustering_coef_bu(s);
            simulated(:,3) = betweenness_bin(s)';
            simulated(:,4) = sum(s.*d)';
            simulated(:,5) = efficiency_bin(s,1);
            simulated(:,6) = mean(matching_ind(s)+matching_ind(s)')';
            % compute the correlation of the simulated
            corr_local_simulated(time,i,model,:,:) = corr(simulated);
            % form a matrix together and correlate these
            corr_local_together(time,i,model,:,:) = corr([observed simulated]);
            % compute the topological organization dissimilarity
            todissim(time,i,model) = topological_organization_dissimilarity(corr(observed),corr(simulated));
        end
    end
end
%% visualise the topological dissimilarity
%  alter the order
todissim_permute = permute(todissim,[2 1 3]);
% make zeros into nans
todissim_permute(todissim_permute==0)=NaN;
% plot
h = figure; h.Position = [100 100 900 300];
h = iosr.statistics.boxPlot(todissim_permute(:,:,[2:13 1]),...
    'showViolin',logical(0),...
    'theme','colorall',...
    'showScatter',logical(0),...
    'scatterMarker','x',...
    'scatterColor',[.5 .5 .5],...
    'symbolColor',[.5 .5 .5],...
    'symbolMarker','x',...
    'boxColor',{'r','r','g','g','g','g','g','b','b','b','b','b','y'},...
    'boxAlpha',0.2);
ylabel('{\itTFdissimilarity}'); xlabel('Days {\itin vitro}');
xticklabels({'14','21','28'});
ylim([0 0.25]);
b = gca; 
b.XAxis.TickDirection = 'out';
b.YAxis.TickDirection = 'out';
b.FontName = 'Arial';
b.FontSize = 14;
%% visualise a specific comparison in terms of a culture and averaged over cultures
time = 1;
culture = 5;
model = 1;
% take the specific data
specific_obs = squeeze(corr_local_observed(time,culture,:,:));
specific_sim = squeeze(corr_local_simulated(time,culture,model,:,:));
% visualise the specific culture
h = figure; h.Position = [100 100 800 300];
subplot(1,2,1); imagesc(specific_obs); title('observed'); caxis([-1 1]); colorbar; xticks([]); yticks([]);
subplot(1,2,2); imagesc(specific_sim); title('simulated'); caxis([-1 1]); colorbar; xticks([]); yticks([]);
sgtitle(sprintf('culture %g, div %g, model %g, tod %g',culture,time,model,squeeze(squeeze(todissim(time,culture,model)))));
% take the averaged data
averaged_obs = squeeze(mean(corr_local_observed(time,:,:,:),2));
averaged_sim = squeeze(mean(corr_local_simulated(time,:,model,:,:),2));
% visualise the averaged data
h = figure; h.Position = [100 100 800 300];
subplot(1,2,1); imagesc(averaged_obs); title('observed'); caxis([-1 1]); colorbar; xticks([]); yticks([]);
subplot(1,2,2); imagesc(averaged_sim); title('simulated'); caxis([-1 1]); colorbar; xticks([]); yticks([]);
sgtitle(sprintf('averaged div %g, model %g, tod %g',time,model,squeeze(mean(todissim(time,:,model),2))));
xticks([]); yticks([]);
%% plot a specific plot
g = figure; g.Position = [100 100 900 500];
plotind = [1 6 11; 2 7 12; 3 8 13; 4 9 14; 5 10 15];
for time = 1:ntime
    % observed
    subplot(3,5,plotind(1,time)); imagesc(squeeze(mean(corr_local_observed(time,:,:,:),2))); caxis([-1 1]); xticks([]); yticks([]);
    % matching
    subplot(3,5,plotind(2,time)); imagesc(squeeze(mean(corr_local_simulated(time,:,3,:,:),2))); caxis([-1 1]); xticks([]); yticks([]);
    % clu-avg
    subplot(3,5,plotind(3,time)); imagesc(squeeze(mean(corr_local_simulated(time,:,4,:,:),2))); caxis([-1 1]); xticks([]); yticks([]);
    % deg-avg
    subplot(3,5,plotind(4,time)); imagesc(squeeze(mean(corr_local_simulated(time,:,9,:,:),2))); caxis([-1 1]); xticks([]); yticks([]);
    % sptl
    subplot(3,5,plotind(5,time)); imagesc(squeeze(mean(corr_local_simulated(time,:,1,:,:),2))); caxis([-1 1]); xticks([]); yticks([]);
end
%% correlate energy and the to matrix
% pick the set
set = 1; 
% take the energy
ep = squeeze(top_e_mean(set,:,:))';
% take only those we have calculated the todissim of
epto = zeros(3,12,13);
% form into the same size to form correlations
for time = 1:3;
    epto(time,:,:) = ep(index(:,1)==pipeline&index(:,3)==time,:);
end
% form indices for rules colours
ind = [ones(3*12*1,1);2*ones(3*12*2,1);3*ones(3*12*5,1);4*ones(3*12*5,1)];
col = {'y','r','g','b'};
% across all models
[rall pall] = corr(epto(:),todissim(:));
% take data
x = epto(:); y = todissim(:);
% keep individual correlations based on rules
rrule  = []; prule = [];
% visualise
figure;
for rule = 1:4;
    [rrule(rule) prule(rule)] = corr(x(ind==rule),y(ind==rule));
    h = scatter(x(ind==rule),y(ind==rule),70,'o',...
        'markerfacecolor',col{rule},...
        'markerfacealpha',0.5,...
        'markeredgecolor',[.5 .5 .5]);
    hold on;
end
sgtitle('all models'); xlabel('energy'); ylabel('TOdissimilarity');
xlim([0 0.6]);
ylim([0 0.3]);
b = gca; b.TickDir = 'out'; b.FontName = 'Arial';
%% compute generative model statistics over time 
% initialise
matching_K     = {};              
matching_Fk    = {};            
matching_Ksum  = {}; 
matching_Fksum = {};
matching_P     = {};              
matching_Psum  = {};
matching_A     = {};       
% loop over networks
for uh = 1:nsamp;
    tic;
    % take the observed network
    w = o_networks{uh};
    % compute the number of nodes
    nnode = size(w,1);
    % take the optimal network simulation parameters, given the set model, for this network
    e_net = squeeze(E(uh,model,:));
    [~,i_low] = min(e_net);
    p_low = squeeze(Ps(uh,model,i_low,:));
    % take the end simulated network
    s = squeeze(N{uh}(model,:,i_low));
    % take the cost
    D = c_networks{uh};
    % run the generative model for homophily
    A = zeros(nnode);
    % set target
    Atgt        = w;
    % set the parameter for this subject
    params      = p_low';
    % number of bi-directional connections
    m           = nnz(Atgt)/2;
    % model var
    modelvar    = [{'powerlaw'},{'powerlaw'}];
    % minimum edge
    epsilon     = 1e-5;
    % run initial matching
    n           = length(D);
    nparams     = size(params,1);
    b           = zeros(m,nparams);
    K           = matching_ind(A);
    K           = K + K';
    % keep the K,Fk,P,A at each iteration, under the current parameter
    Kall        = [];
    Fkall       = [];
    Pall        = [];
    Aall        = [];
    % save the first K
    Kall(1,:,:)  = K;
    % save the first A
    Aall(1,:,:)  = A;
    % display
    disp(sprintf('running generative model %g for %g connections across %g sensors...',uh,m,nnode));
    for iparam = 1:nparams
        eta = params(iparam,1);
        gam = params(iparam,2);
        K = K + epsilon;
        n = length(D);
        mseed = nnz(A)/2;
        mv1 = modelvar{1};
        mv2 = modelvar{2};
        switch mv1
            case 'powerlaw'
                Fd = D.^eta;
            case 'exponential'
                Fd = exp(eta*D);
        end
        switch mv2
            case 'powerlaw'
                Fk = K.^gam;
            case 'exponential'
                Fk = exp(gam*K);
        end
        Ff = Fd.*Fk.*~A;
        [u,v] = find(triu(ones(n),1));
        indx = (v - 1)*n + u;
        P = Ff(indx);
        % save the first parameterised K in Fk
        Fkall(1,:,:)  = Fk;
        % save the first probabilities
        Ff(isinf(Ff)) = 0;
        Pall(1,:,:)   = Ff;
        step = 2; % added in
        for ii = (mseed + 1):m
            C = [0; cumsum(P)];
            r = sum(rand*C(end) >= C);
            uu = u(r);
            vv = v(r);
            A(uu,vv) = 1;
            A(vv,uu) = 1;
            updateuu = find(A*A(:,uu));
            updateuu(updateuu == uu) = [];
            updateuu(updateuu == vv) = [];
            updatevv = find(A*A(:,vv));
            updatevv(updatevv == uu) = [];
            updatevv(updatevv == vv) = [];
            c1 = [A(:,uu)', A(uu,:)];
            for i = 1:length(updateuu)
                j = updateuu(i);
                c2 = [A(:,j)' A(j,:)];
                use = ~(~c1&~c2);
                use(uu) = 0;  use(uu+n) = 0;
                use(j) = 0;  use(j+n) = 0;
                ncon = sum(c1(use))+sum(c2(use));
                if (ncon==0)
                    K(uu,j) = epsilon;
                    K(j,uu) = epsilon;
                else;
                    K(uu,j) = (2*(sum(c1(use)&c2(use))/ncon)) + epsilon;
                    K(j,uu) = K(uu,j);
                end
            end
            c1 = [A(:,vv)', A(vv,:)];
            for i = 1:length(updatevv)
                j = updatevv(i);
                c2 = [A(:,j)' A(j,:)];
                use = ~(~c1&~c2);
                use(vv) = 0;  use(vv+n) = 0;
                use(j) = 0;  use(j+n) = 0;
                ncon = sum(c1(use))+sum(c2(use));
                if (ncon==0)
                    K(vv,j) = epsilon;
                    K(j,vv) = epsilon;
                else
                    K(vv,j) = (2*(sum(c1(use)&c2(use))/ncon)) + epsilon;
                    K(j,vv) = K(vv,j);
                end
            end
            switch mv2
                case 'powerlaw'
                    Fk = K.^gam;
                case 'exponential'
                    Fk = exp(gam*K);
            end
            Kall(step,:,:)  = K;                     % added in (K)
            Fkall(step,:,:) = Fk;                    % added in (Fk)
            Aall(step,:,:)  = A;                     % added in (A)
            Ff = Fd.*Fk.*~A;
            P = Ff(indx);
            % remove infinite values (self connections)
            Ff(isinf(Ff))   = 0;
            Pall(step,:,:)  = Ff;                    % added in (P)
            % change the step
            step = step+1;
        end
        b(:,iparam)        = find(triu(A,1));
        matching_K{uh}     = Kall;                  % added in (K)
        matching_Fk{uh}    = Fkall;                 % added in (Fk)
        matching_Ksum{uh}  = squeeze(sum(Kall,2));  % added in (Ksum)
        matching_Fksum{uh} = squeeze(sum(Fkall,2)); % added in (FkSum)
        matching_P{uh}     = Pall;                  % added in (Psum)
        matching_Psum{uh}  = squeeze(sum(Pall,2));  % added in (Psum)
        matching_A{uh}     = Aall;                  % added in (A)
    end
    time = toc;
    disp(sprintf('generative model complete: network %g took %.3g seconds',uh,time));
    end
%% explore correlations between observed and simulated measurements
% initialise
concatenated_data = {};
corr_concatenated_data = zeros(nsamp,4+2*nmeasures,4+2*nmeasures);
% labels
var_labels = string({'costs','parameterised costs','values','parameterised values',...
    'obs degree','obs clustering','obs betweenn','obs length','obs eff','obs match',...
    'sim degree','sim clustering','sim between','sim length','sim eff','sim match'});
% concatenate the data by looping over subjects
for i = 1:nsamp;
    % get the costs for this network
    d = c_networks{i};
    % get the eta for this network
    e_net = squeeze(E(i,model,:));
    [~,i_low] = min(e_net);
    eta = squeeze(Ps(i,model,i_low,1));
    % compute the Fd
    Fd = d.^eta;
    Fd(isinf(Fd))=0;
   % place in all the data for this network
   concatenated_data{i}(:,1) = sum(d)';
   concatenated_data{i}(:,2) = sum(Fd)';
   concatenated_data{i}(:,3) = sum(matching_Ksum{i})';
   concatenated_data{i}(:,4) = sum(matching_Fksum{i})';
   concatenated_data{i}(:,5:4+nmeasures) = local_observed{i};
   concatenated_data{i}(:,5+nmeasures:4+2*nmeasures) = local_simulated{i};
   % correlate these
   corr_concatenated_data(i,:,:) = corr(concatenated_data{i});
end
% visualise the average
mean_corr_concatenated = squeeze(mean(corr_concatenated_data,1));
% plot
figure;
imagesc(mean_corr_concatenated); colorbar; caxis([-1 1]);
%% compute a correlation matrix with the observed statistics and the parameters
% determine the set
set = 1;
% determine the model 
model = 3;
% observed statistics
observed = zeros(size(index,1),6);
for network = 1:size(index,1);
    % take matrix
    A = all_o_networks{network};
    D = all_c_networks{network};
    % compute statistics
    observed(network,1) = density_und(A);
    observed(network,2) = mean(degrees_und(A));
    observed(network,3) = mean(clustering_coef_bu(A));
    observed(network,4) = mean(betweenness_bin(A));
    observed(network,5) = mean(A.*D,'all');
    observed(network,6) = efficiency_bin(A,0);
    [~,observed(network,7)] = modularity_und(A);
end
% wiring data for matching
wiring_data = [squeeze(top_p_mean(set,model,:,:))];
% add in the generative model parameters for homophily
data = [wiring_data observed];
% labels
observed_labels = string({'density','degree','clustering','betweenness','edge length','efficiency','modularity'});
wiring_labels = string({'eta','gamma'});
all_labels = string({'eta','gamma','density','degree','clustering','betweenness','edge length','efficiency','modularity'});
% correlate
[r p] = corr(data);
% visualise
figure;
imagesc(r); caxis([-1 1]); xticklabels(all_labels); xtickangle(45); yticklabels(all_labels); c = colorbar; c.Label.String = 'r';
colormap(flip(bluewhitered));
b = gca; b.FontName = 'Arial'; b.TickDir = 'out';
% pls between observed and simulated
[xl,yl,xs,ys,beta,var,mse] = plsregress(normalize(wiring_data),normalize(observed));
% observed
robs = corr(xs(:,1),ys(:,1));
% permute scores
nperm = 1000;
n = size(observed,1);
r = [];
for i = 1:nperm;
    wiring_data_perm = wiring_data(randperm(n),:);
    [~,~,xs_perm,ys_perm] = plsregress(normalize(wiring_data_perm),normalize(observed));
    r(i) = corr(xs_perm(:,1),ys_perm(:,1));
end
% permuted p value
pperm = 1/sum(robs>r);
% plot the key measures
comp = 1;
h = figure; h.Position = [100 100 1500 250];
sgtitle('pls');
% visualise pls
subplot(1,5,1);
bar(var); xticklabels({'predictor','response'}); ylabel('% variance explained'); 
b = gca; b.FontName = 'Arial'; b.TickDir = 'out';
subplot(1,5,2); 
scatter(xs(:,comp),ys(:,comp),150,'.'); xlabel('parameter score'); ylabel('network score'); ylim([-100 100]); xlim([-0.4 0.4]); 
b = gca; b.FontName = 'Arial'; b.TickDir = 'out';
subplot(1,5,3);
histogram(r); hold on; xline(robs,'linewidth',3); xlabel('r'); ylabel('frequency'); 
b = gca; b.FontName = 'Arial'; b.TickDir = 'out';
subplot(1,5,4);
bar(-xl(:,comp)); xticklabels(wiring_labels); xtickangle(45); ylabel('parameter loading'); ylim([-12 12]); 
b = gca; b.FontName = 'Arial'; b.TickDir = 'out';
subplot(1,5,5);
bar(-yl(:,comp)); xticklabels(observed_labels); xtickangle(45); ylabel('network loading'); 
b = gca; b.FontName = 'Arial'; b.TickDir = 'out';
% correlate the first pca of each
[coef,score,~,~,exp] = pca(normalize(wiring_data));
[coefy,scorey,~,~,expy] = pca(normalize(observed));
% get the stats
[r p] = corr(score(:,comp),scorey(:,comp));
% visualise pca
h = figure; h.Position = [100 100 1500 250];
sgtitle('pca');
subplot(1,4,1);
bar([exp(1) expy(1)]); xticklabels({'predictor','response'}); ylabel('% variance explained');
subplot(1,4,2);
scatter(score(:,comp),scorey(:,comp),150,'.'); xlabel('parameter score'); ylabel('network score'); ylim([-7 7]); xlim([-7 7]);
subplot(1,4,3);
bar(coef(:,comp)); xticklabels(wiring_labels); xtickangle(45); ylabel('network loading'); ylim([-1 1]);
subplot(1,4,4);
bar(coefy(:,comp)); xticklabels(observed_labels); xtickangle(45); ylabel('parameter loading');
%% evaluate parameter movements over time, pipelines and statistics
% select which set of nset to view
set = 1;
% take the parameters
p = top_p{set};
% set the model to visualise
model = 3;
% take parameters over time
h = figure; h.Position = [100 100 1200 250];
div_data = zeros(nnet*ncultures*nset(set),2,ntime);
for time = 1:ntime;
    subplot(1,ntime,time);
    % take the data
    if set == 1
        p_select = p(:,index(:,3)==time,:);
        data = squeeze(p_select(model,:,:));
    else
        p_select = p(:,index(:,3)==time,:,:);
        data = squeeze(p_select(model,:,:,:));
        data = reshape(data,[size(data,1)*size(data,2),2]);
    end
    % keep the data
    div_data(:,:,time) = data;
    % run correlations
    [r pv] = corr(data(:,1),data(:,2));
    % visualise
    h = scatter(data(:,1),data(:,2),30,'o','markerfacecolor','b','markerfacealpha',0.5,'markeredgecolor',[170 170 170]./256);
    title(sprintf('%s, r=%.3g, p=%.3g',div_labels(time),r,pv));
    sgtitle(sprintf('top %g simulations',nset(set)));
    xlim([-3.7 0.2]); ylim([-0.1 1.1]);
    xlabel('eta'); ylabel('gamma');
    b = gca;
    b.XAxis.TickDirection = 'out';
    b.YAxis.TickDirection = 'out';
    b.FontName = 'Arial';
    hold on;
end
% look at parameter distributions over time
figure;
iosr.statistics.boxPlot(div_data,...
    'showViolin',logical(1),...
    'theme','colorall',...
    'symbolMarker','x',...
    'symbolColor','k');
xticklabels({'eta','gamma'});
b = gca; 
b.XAxis.TickDirection = 'out';
b.YAxis.TickDirection = 'out';
% take over pipelines
h = figure; h.Position = [100 100 900 250];
pipeline_data = zeros(48*nset(set),2,3);
for pipeline = 1:3;
    subplot(1,3,pipeline);
    % take the data
    if set == 1
        p_select = p(:,index(:,1)==pipeline,:);
        data = squeeze(p_select(model,:,:));
    else
        p_select = p(:,index(:,1)==pipeline,:,:);
        data = squeeze(p_select(model,:,:,:));
        data = reshape(data,[size(data,1)*size(data,2),2]);
    end
    % keep this data
    pipeline_data(:,:,pipeline) = data;
    % run correlations
    [r pv] = corr(data(:,1),data(:,2));
    % visualise
    h = scatter(data(:,1),data(:,2),30,'o','markerfacecolor','k','markerfacealpha',0.5,'markeredgecolor',[170 170 170]./256);
    title(sprintf('%s, r=%.3g, p=%.3g',pipeline_labels(pipeline),r,pv));
    sgtitle(sprintf('top %g simulations',nset(set)));
    xlim([-3.7 0.2]); ylim([-0.1 1.1]);
    xlabel('eta'); ylabel('gamma');
    b = gca;
    b.XAxis.TickDirection = 'out';
    b.YAxis.TickDirection = 'out';
    b.FontName = 'Arial';
    hold on;
end
% run statistics between divs
div_p = []; div_d = [];
eta = squeeze(div_data(:,1,:)); 
[~,div_p(1,1)] = ttest(eta(:,1),eta(:,2)); 
[~,div_p(1,2)] = ttest(eta(:,1),eta(:,3));
[~,div_p(1,3)] = ttest(eta(:,1),eta(:,4));
[~,div_p(1,4)] = ttest(eta(:,2),eta(:,3));
[~,div_p(1,5)] = ttest(eta(:,2),eta(:,4));
[~,div_p(1,6)] = ttest(eta(:,3),eta(:,4));
div_d(1,1) = computeCohen_d(eta(:,1),eta(:,2)); 
div_d(1,2) = computeCohen_d(eta(:,1),eta(:,3));
div_d(1,3) = computeCohen_d(eta(:,1),eta(:,4));
div_d(1,4) = computeCohen_d(eta(:,2),eta(:,3));
div_d(1,5) = computeCohen_d(eta(:,2),eta(:,4));
div_d(1,6) = computeCohen_d(eta(:,3),eta(:,4));
gam = squeeze(div_data(:,2,:));
[~,div_p(2,1)] = ttest2(gam(:,1),gam(:,2)); 
[~,div_p(2,2)] = ttest2(gam(:,1),gam(:,3));
[~,div_p(2,3)] = ttest2(gam(:,1),gam(:,4));
[~,div_p(2,4)] = ttest2(gam(:,2),gam(:,3));
[~,div_p(2,5)] = ttest2(gam(:,2),gam(:,4));
[~,div_p(2,6)] = ttest2(gam(:,3),gam(:,4));
div_d(2,1) = computeCohen_d(gam(:,1),gam(:,2)); 
div_d(2,2) = computeCohen_d(gam(:,1),gam(:,3));
div_d(2,3) = computeCohen_d(gam(:,1),gam(:,4));
div_d(2,4) = computeCohen_d(gam(:,2),gam(:,3));
div_d(2,5) = computeCohen_d(gam(:,2),gam(:,4));
div_d(2,6) = computeCohen_d(gam(:,3),gam(:,4));
% run statistics between pipelines
pipeline_p = []; pipeline_d = [];
eta = squeeze(pipeline_data(:,1,:)); 
[~,pipeline_p(1,1)] = ttest(eta(:,1),eta(:,2)); 
[~,pipeline_p(1,2)] = ttest(eta(:,1),eta(:,3));
[~,pipeline_p(1,3)] = ttest(eta(:,2),eta(:,3));
pipeline_d(1,1) = computeCohen_d(eta(:,1),eta(:,2),'paired'); 
pipeline_d(1,2) = computeCohen_d(eta(:,1),eta(:,3),'paired');
pipeline_d(1,3) = computeCohen_d(eta(:,2),eta(:,3),'paired');
gam = squeeze(pipeline_data(:,2,:));
[~,pipeline_p(2,1)] = ttest(gam(:,1),gam(:,2));
[~,pipeline_p(2,2)] = ttest(gam(:,1),gam(:,3));
[~,pipeline_p(2,3)] = ttest(gam(:,2),gam(:,3));
pipeline_d(2,1) = computeCohen_d(gam(:,1),gam(:,2),'paired'); 
pipeline_d(2,2) = computeCohen_d(gam(:,1),gam(:,3),'paired');
pipeline_d(2,3) = computeCohen_d(gam(:,2),gam(:,3),'paired');