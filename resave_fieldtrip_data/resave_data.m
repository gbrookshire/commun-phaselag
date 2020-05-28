data_dir = '../data/SubjectCMC/';

ft_defaults

% find the interesting epochs of data
cfg = [];
cfg.trialfun = 'trialfun_left';
cfg.dataset = [data_dir 'SubjectCMC.ds'];
cfg = ft_definetrial(cfg);

% detect EOG artifacts in the MEG data
cfg.continuous                = 'yes';
cfg.artfctdef.eog.padding     = 0;
cfg.artfctdef.eog.bpfilter    = 'no';
cfg.artfctdef.eog.detrend     = 'yes';
cfg.artfctdef.eog.hilbert     = 'no';
cfg.artfctdef.eog.rectify     = 'yes';
cfg.artfctdef.eog.cutoff      = 2.5;
cfg.artfctdef.eog.interactive = 'no';
cfg = ft_artifact_eog(cfg);

% detect jump artifacts in the MEG data
cfg.artfctdef.jump.interactive = 'no';
cfg.padding                    = 5;
cfg = ft_artifact_jump(cfg);

% detect muscle artifacts in the MEG data
cfg.artfctdef.muscle.cutoff      = 8;
cfg.artfctdef.muscle.interactive = 'no';
cfg = ft_artifact_muscle(cfg);

% reject the epochs that contain artifacts
cfg.artfctdef.reject          = 'complete';
cfg = ft_rejectartifact(cfg);

% preprocess the MEG data
cfg.demean                    = 'yes';
cfg.dftfilter                 = 'yes';
cfg.channel                   = {'all' '-EMGlft' '-EMGrgt'}; %{'MEG'};
cfg.continuous                = 'yes';
meg = ft_preprocessing(cfg);

% Preprocess the EMG data
cfg              = [];
cfg.dataset      = meg.cfg.dataset;
cfg.trl          = meg.cfg.trl;
cfg.continuous   = 'yes';
cfg.demean       = 'yes';
cfg.dftfilter    = 'yes';
cfg.channel      = {'EMGlft' 'EMGrgt'};
cfg.hpfilter     = 'yes';
cfg.hpfreq       = 10;
cfg.rectify      = 'yes';
emg = ft_preprocessing(cfg);

% Combine them
data = ft_appenddata([], meg, emg);

% Make the time variable the same for every trial, to facilitate reading
% the data in mne-python.
for k = 1:length(data.time)
    data.time{k} = data.time{1};
end

% Save the data
hdr = meg.hdr;
save([data_dir 'hdr.mat'], 'hdr')
save([data_dir 'data.mat'], 'data')

% Plot it
figure
subplot(2,1,1);
plot(data.time{1},data.trial{1}(77,:));
axis tight;
legend(data.label(77));

subplot(2,1,2);
plot(data.time{1},data.trial{1}(152:153,:));
axis tight;
legend(data.label(152:153));

% Compute coherence (method 1)
cfg            = [];
cfg.output     = 'fourier';
cfg.method     = 'mtmfft';
cfg.foilim     = [5 100];
cfg.tapsmofrq  = 5;
cfg.keeptrials = 'yes';
cfg.channel    = {'MEG' 'EMGlft' 'EMGrgt'};
freqfourier    = ft_freqanalysis(cfg, data);
cfg            = [];
cfg.method     = 'coh';
cfg.channelcmb = {'MEG' 'EMG'};
fdfourier      = ft_connectivityanalysis(cfg, freqfourier);

% Compute coherence (method 2)
cfg            = [];
cfg.output     = 'powandcsd';
cfg.method     = 'mtmfft';
cfg.foilim     = [5 100];
cfg.tapsmofrq  = 5;
cfg.keeptrials = 'yes';
cfg.channel    = {'MEG' 'EMGlft' 'EMGrgt'};
cfg.channelcmb = {'MEG' 'EMGlft'; 'MEG' 'EMGrgt'};
freq           = ft_freqanalysis(cfg, data);
cfg            = [];
cfg.method     = 'coh';
cfg.channelcmb = {'MEG' 'EMG'};
fd             = ft_connectivityanalysis(cfg, freq);

% Plot the coherence
cfg                  = [];
cfg.parameter        = 'cohspctrm';
cfg.xlim             = [5 80];
cfg.refchannel       = 'EMGlft';
cfg.layout           = 'CTF151_helmet.mat';
cfg.showlabels       = 'yes';
figure; ft_multiplotER(cfg, fd)

cfg.channel = 'MRC21';
figure; ft_singleplotER(cfg, fd);

cfg                  = [];
cfg.parameter        = 'cohspctrm';
cfg.xlim             = [15 20];
cfg.zlim             = [0 0.1];
cfg.refchannel       = 'EMGlft';
cfg.layout           = 'CTF151_helmet.mat';
figure; ft_topoplotER(cfg, fd)

