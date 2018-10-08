M = csvread('wind_data_NoLabel.csv');
Z = zscore(M)

x1 = Z(:,1);
x2 = Z(:,2);
x3 = Z(:,3);
x4 = Z(:,4);
x5 = Z(:,5);
x6 = Z(:,6);
x7 = Z(:,7);
x8 = Z(:,8);
x9 = Z(:,9);
x10 = Z(:,10);
x11 = Z(:,11);
x12 = Z(:,12);
x13 = Z(:,13);
x14 = Z(:,14);
x15 = Z(:,15);
x16 = Z(:,16);

Y = Z(:,17);
v = [1:5023];
% 
% figure
% subplot(2,1,1)
% plot(v,M(:,1),v, M(:,2), v, M(:,3), v, M(:,4), v, M(:,5), v, M(:,6), v, M(:,17));
% title('Data before standardization');
% legend('x1','x2','x3','x4','x5','x6','winding temp');
% xlabel('time');
% ylabel('value');
% subplot(2,1,2)
% plot(v,x1,v, x2, v, x3, v, x4, v, x5, v, x6, v, Y);
% title('Data after standardization');
% legend('x1','x2','x3','x4','x5','x6','winding temp');
% xlabel('time');
% ylabel('value');

figure
t = [1:5023];
plot(t, M(:,17));
title('Generator winding temperature trend');
xlabel('time');
ylabel('temperature (F)');


X = [x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16];
[eigenvectors,PCs,eigenvalues]=princomp(X);
variance=eigenvalues./sum(eigenvalues);

figure; 
plot([1:16],variance,'*-');
ylabel('ratio of total variance explained by mode');
xlabel('mode number');
title('Variance Explained by each Mode');

% plot the first three modes (eigenvestors and PCs)
figure; 
for i=1:2
subplot(2,2,2*i-1)
plot(eigenvectors(:,i));
%ylim([-1 1]);
xlabel('x');
title(['eigenvector',num2str(i)]);

subplot(2,2,2*i)
plot(PCs(:,i));
xlabel('time');
title(['PC',num2str(i)]);
end


Xnew = [PCs(:,1) PCs(:,2)];
Xnew2 = Xnew';
Y2 =Y';

[a SE PVAL INMODEL STATS NEXTSTEP HISTORY]=stepwisefit(Xnew,Y);

if 0
    
n = 5023;
Nin= 4000

trainX= Xnew(1:Nin,1:2);
trainY= Y(1:Nin,1);

testX= Xnew(Nin+1:end,1:2);
testY= Y(Nin+1:end,1);
    
% Fit model and predict:
mdl = stepwiselm(trainX,trainY);
predY = predict(mdl,Xnew);
% Plot predictions against known response
rmsemlp = sqrt(mse(Y - predY));
Rmlp = corrcoef(Y,predY);
figure
t=[1:5023];
plot(t,Y,'b-',t,predY,'r-');
xlabel('time');
ylabel('winding temp');

title (['RMSE =',num2str(rmsemlp), ', target (blue) vs predicted (red)']);

figure
sz = 30;
xline=[min([Y;predY]):max([Y;predY])]; 
plot(Y,predY,'bo'); hold on
plot(xline,xline,'k-','LineWidth',1);
% scatter (Y, predY,sz);
% hold on;
% plot (v,v);
xlabel('Y');
ylabel('predicted Y');
 title ('MLR with r = 0.870');

end

if 1
rt = [1 2 3 4 5 6 7 8 9 10];    
re = [0.1205 0.1159 0.0934 0.0898 0.0807 0.0873 0.0774 0.0841 0.0765 0.0707];
ree = [0.0707 0.0765 0.0841 0.0774 0.0873 0.0807 0.0898 0.0934 0.1159 0.1205];
sz = 80;
RGB = [172 102 255]/256 ;
figure
scatter(rt,ree,sz,RGB, 'filled');
xlabel('No of lags');
ylabel('RMSE');
title ('RMSE with different predictand lags');
    
end

if 0
inputDelays = 1:1;
feedbackDelays = 1:8
m=11;
net = narxnet(inputDelays,feedbackDelays,m);

[inputs,inputStates,layerStates,targets] = preparets(net,con2seq(Xnew2),{},con2seq(Y2));

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio   = 15/100;
net.divideParam.testRatio  = 15/100;

[net,tr] = train(net,inputs,targets,inputStates,layerStates);

outputs = net(inputs,inputStates,layerStates);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs);

tt = cell2mat(targets);
y = cell2mat(outputs);

rmse = sqrt(mse(tt - y));

netc = closeloop(net);
netc.name = [net.name ' - Closed Loop'];

[xc,xic,aic,tc] = preparets(netc,con2seq(Xnew2),{},con2seq(Y2));
yc = netc(xc,xic,aic);
perfc = perform(netc,tc,yc);
t2 = cell2mat(tc);
y2 = cell2mat(yc);

rmse2 = sqrt(mse(t2 - y2));

t = [1:5021];

Rnn = corrcoef(y2,t2);

% figure
% xline=[min([y2;t2]):max([y2;t2])]; 
% plot(t2,y2,'bo'); hold on
% plot(xline,xline,'k-','LineWidth',1);
% xlabel('Y');
% ylabel('predicted Y');
%  title ('NN model with r = 0.995');

% if m == 1 
%  figure
% end
% 
% subplot(3,4,m);
% plot(t,y2(m,:),'b-', t,t2(m,:),'r-');
% title(['no. of neurons =',num2str(m)]);
% xlabel('time');
% ylabel('winding temp');
% hold on;

% 
% plot(t,y,'r-',t, tt,'b-');
% title (['RMSE =',num2str(rmse), ', target (blue) vs predicted (red)']);
% ylabel('winding temp');
% 
% figure
% plot(t,y2,'r-',t, t2,'b-');
% title (['RMSE =',num2str(rmse2), ', target (blue) vs predicted (red)']);
% xlabel('time');
% ylabel('winding temp');


end 

% figure
% autocorr(Y)
% xlabel('Lag')
% ylabel('Y autocorrelation')
% title('Y autocorrelation')

Nin=3000
xdata_in= Xnew(1:Nin,1:2);
ydata_out= Y(1:Nin,1);

xdata_test= Xnew(Nin+1:end,1:2);
ydata_test= Y(Nin+1:end,1);


if 0 % change to 1 if you want to run this part of the script

no_runs=20;   % note that it takes several minutes to run, so you might test it first with smaller number of runs
 % here can choose the number of neurons
inputDelays = 1:0;
feedbackDelays = 1:0;
hiddenLayerSize = 5;

for kk=1:no_runs
% pick 50 random points from x and ydata
index = randi(Nin,[1 Nin]);
xdata1=xdata_in(index,:);
ydata1=ydata_out(index,:);


% apply MLP NN model

net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize);

[inputs,inputStates,layerStates,targets] = preparets(net,con2seq(xdata1'),{},con2seq(ydata1'));

[net,tr] = train(net,inputs,targets,inputStates,layerStates);

ymodel(kk,:) = net(inputs,inputStates,layerStates);

netc = closeloop(net);
netc.name = [net.name ' - Closed Loop'];
[xc,xic,aic,tc] = preparets(netc,con2seq(xdata1'),{},con2seq(ydata1'));
ymodelc(kk,:) = netc(xc,xic,aic);


t = cell2mat(targets);
ym = cell2mat(ymodel(kk,:));
ymc = cell2mat(ymodelc(kk,:));

% find RMSE between this output and test data
rmse(kk) = sqrt(mse(ym - t));
rmsec(kk) = sqrt(mse(ymc - t));
end

% calculate ensamble mean and RMSE
ymodel_mean=mean(ym);
rmse(kk+1) = sqrt(mse(ym - t));

RMSE(1)=rmse(kk+1);

ymodel_meanc=mean(ymc);
rmsec(kk+1) = sqrt(mse(ymc - t));

RMSEC(1)=rmsec(kk+1);

% plot RMSE
figure;
plot([1:no_runs],rmse(1:no_runs),'bo',[no_runs+1],rmse(no_runs+1),'ro');
xlabel('model run');
ylabel('RMSE');

figure;
plot([1:no_runs],rmsec(1:no_runs),'bo',[no_runs+1],rmsec(no_runs+1),'ro');
xlabel('model run');
ylabel('RMSEC');

% 
% [r p]=corrcoef([ydata_test ymodel' ymodel_mean']);
% 
% figure;
% plot(ydata_test,ymodel_mean,'bo'); 
% title(['MLP model mean w/ 10 hidden neurons r= ' num2str(r(1,end),'%2.2f')]);

end







 
