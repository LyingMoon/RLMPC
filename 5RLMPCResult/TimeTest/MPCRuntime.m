clear all;
load 001MPC4.mat;

predictionTime = 0.5;
Cts = 0.1;
Ts = 0.01;
NNPredictStep = predictionTime/Cts;
mpcPredictStep = predictionTime/Ts;
diff = mpcPredictStep/NNPredictStep;

N = 400000;
% Randomize the training set
state=(rand(N,4)-0.5);
x1=[state(1:N/2,1)*3;state(N/2+1:N,1)*1];
x2=[state(1:N/2,2);state(N/2+1:N,2)*0.4];
x3=[state(1:N/4,3)*8;state(1+N/4:N,3)*1];
x4=[state(1:N/4,4)*8;state(1+N/4:N,4)*1];

refx1=[(rand(N/4,NNPredictStep+1)-0.5)*3;(rand(3*N/4,NNPredictStep+1)-0.5)];
refx1mpc = zeros(N,mpcPredictStep);

% Use the prediction shrinkage technique
for i = 1:N
    for j = 1:mpcPredictStep
    refx1mpc(i,j) = (refx1(i,floor((j-1)/diff)+2) - refx1(i,floor((j-1)/diff)+1)) ... 
        *(rem(j-1,diff)/diff) + refx1(i,floor((j-1)/diff)+1);
    end
end

uValue=zeros(1,N);

% tic and toc will evaluate the execution time required
tic
for k=1:N
    x=mpcstate(mpc1,[x1(k),x2(k),x3(k),x4(k)]);
    uValue(:,k)=mpcmove(mpc1,x,[],[refx1mpc(k,:);zeros(3,mpcPredictStep)]');
end
toc

INPUT=[x1,x2,x3,x4,refx1];
OUTPUT=uValue';
% Do remember to store the two variables INPUT and OUTPUT in .mat format,
% in the example code, they are named 001INPUT3.mat and 001OUTPUT3.mat