%% load activity recordings
% set network range
network = [0:100];
% set epoch strings
epochset = string({...
    '01','02','03','04','05','06','07','08','09','10'});
% initalise
activity = cell(100,10);
% set directory
se_swc_4_dir = '/imaging/shared/ja02/CBUActors/SpatialRNNs/Results/mazeGenI_SE1_sWC_mtlIV/Activity_Recordings/';
l1_3_dir = '/imaging/shared/users/ja02/CBUActors/SpatialRNNs/Results/mazeGenI_L1_mtIII/Activity_Recordings';
% change directory
cd(l1_3_dir);
% loop
for net = 1:length(network);
    for epoch = 1:length(epochset);
        str = sprintf('SN_%g_%s_recording.mat',network(net),epochset(epoch));
        load(str);
        % form an array
        unitarray = cell2mat(recorded_data(2:end,2));
        % intialise
        activity{net,epoch} = zeros(100,640,50,2);
        % loop over units
        for unit = 1:100;
            % get unit indices, not that they are zero indexed
            ind = unitarray==unit-1;
            % get index +1 because we index back
            numind = find(ind)+1;
            % initalise an array
            pre_act_unit = zeros(sum(ind),50);
            post_act_unit = zeros(sum(ind),50);
            % loop over each unit over trials
            for trial = 1:sum(ind);
                pre_act_unit(trial,:) = cell2mat(recorded_data(numind(trial),7));
                post_act_unit(trial,:) = cell2mat(recorded_data(numind(trial),8));
            end
            activity{net,epoch}(unit,:,:,1) = pre_act_unit;
            activity{net,epoch}(unit,:,:,2) = post_act_unit;
            % display
            disp(sprintf('network %g, epoch %g, unit %g complete',net,epoch,unit));
        end
    end
end
% save
l1_3_activity = activity;
%{
cd('/imaging/astle/users/da04/PhD/ann_jascha/data');
save('l1_3_activity.mat','l1_3_activity','-v7.3');
%}