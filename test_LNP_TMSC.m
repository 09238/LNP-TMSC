clear;
clc;


load WebKB.mat   %dn

gt = Y;
cls_num = length(unique(gt)); %cluster number
num_views = length(X); % view number


%% Data preprocessing
for v = 1:num_views
    X{v}=NormalizeData(X{v}');   
end


%% Function selection
n = 1; 
%n = 2;


%% Parameter selection
lambda =0.0009;
gamma = 0.8;
tau = 100000;


[F,O] = LNP_TMSC(X,lambda,tau,gamma,n);

%% Spectral Clustering
S = 0;
for k=1:num_views
    S = S + abs(F{k})+abs(F{k}');
end
C = SpectralClustering(S,cls_num);
[~, nmi, ~] = compute_nmi(gt,C);
ACC = Accuracy(C,double(gt));
[f1,pre,rec] = compute_f(gt,C);
[ARI,~,~,~]=RandIndex(gt,C);
save('result.mat','ACC','ARI','nmi','f1','pre','rec');

