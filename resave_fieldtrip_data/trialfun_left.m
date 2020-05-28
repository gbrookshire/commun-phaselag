function trl = trialfun_left(cfg)

% read in the triggers and create a trial-matrix
% consisting of 1-second data segments, in which 
% left ECR-muscle is active.

event = ft_read_event(cfg.dataset);
trig  = [event(find(strcmp('backpanel trigger', {event.type}))).value];
indx  = [event(find(strcmp('backpanel trigger', {event.type}))).sample];

%left-condition
sel = [find(trig==1028):find(trig==1029)];

trig = trig(sel);
indx = indx(sel);
 
trl = [];
for j = 1:length(trig)-1
  trg1 = trig(j);
  trg2 = trig(j+1);
  if trg1<=100 & trg2==2080,
    trlok      = [[indx(j)+1:1200:indx(j+1)-1200]' [indx(j)+1200:1200:indx(j+1)]'];
    trlok(:,3) = [0:-1200:-1200*(size(trlok,1)-1)]';
    trl        = [trl; trlok];
  end
end
