clear all

% This is the offline data gathered from NNMPC, under the condition of
% fre=1 and amp=1, and initial state [0,0,0,0]
% If you want to plot this, put datav1-x4 in this folder
% ALL SIMULATION
% load datav1.csv;
% load datax1.csv;
% load datax2.csv;
% load datax3.csv;
% load datax4.csv;


load AdBd001Qube.mat;
load 001MPC4.mat
predictionTime = 0.5;
Cts = 0.1;
Ts = 0.01;
NNPredictStep = predictionTime/Cts;
mpcPredictStep = predictionTime/Ts;
diff = mpcPredictStep/NNPredictStep;

refx1= zeros(1,NNPredictStep+1);

refx1mpc= zeros(1,mpcPredictStep);

refx1smpc = zeros(1,mpcPredictStep);

N = 700;

state = zeros(4,N+1);
stateS = zeros(4,N+1);

% Initialize the initial state for different comparsion 
state(1,1) = 0;%(rand(4,1)-1)*0.5;
stateS(1,1) = state(1,1);

state(2,1) = 0;%(rand(4,1)-1)*0.5;
stateS(2,1) = state(2,1);

% Two cost function that compares SMPC and MPC
cost = 0; %Cost of MPC
costs = 0; %Cost of SMPC

% Change those parameters to compare SMPC and MPC
feq=1;
weight = 1;
uValue = zeros(1,N+1);
uValueS = zeros(1,N+1);

for i = 1:N
    
    for p = 1:mpcPredictStep
        refx1mpc(1,p) = weight*sin((i+p-1)*Ts/feq);
    end
    for k = 1:(NNPredictStep+1)
        refx1(1,k) = weight*sin((i+diff*k-diff)*Ts/feq);
    end
    for j = 1:mpcPredictStep
    refx1smpc(1,j) = (refx1(1,floor((j-1)/diff)+2) - refx1(1,floor((j-1)/diff)+1)) ... 
        *(rem(j-1,diff)/diff) + refx1(1,floor((j-1)/diff)+1);
    end
    x=mpcstate(mpc1,state(:,i)');
    xs=mpcstate(mpc1,stateS(:,i)');
    uValue(1,i)=mpcmove(mpc1,x,[],[refx1mpc(1,:);zeros(3,mpcPredictStep)]');
    uValueS(1,i)=mpcmove(mpc1,xs,[],[refx1smpc(1,:);zeros(3,mpcPredictStep)]');
    state(:,i+1) = Ad*state(:,i) + Bd*(uValue(1,i));
    stateS(:,i+1) = Ad*stateS(:,i) + Bd*(uValueS(1,i));
    cost = cost + costfun(state(:,i+1),refx1mpc(1,2),uValue(1,i));
    costs = costs + costfun(stateS(:,i+1),refx1mpc(1,2),uValueS(1,i));
end

time=linspace(0,N,N)*Ts;
figure;

state=state(:,1:700);
stateS=stateS(:,1:700);
uValue=uValue(:,1:700);
uValueS=uValueS(:,1:700);
datax1=datax1(:,1:700);
datax2=datax2(:,1:700);
datax3=datax3(:,1:700);
datax4=datax4(:,1:700);
datav1=datav1(:,1:700);

subplot(5,1,1);
plot(time,state(1,:),'go-','LineWidth',2);
hold on;
plot(time,stateS(1,:),'b.--','LineWidth',1.7,'MarkerSize',9);
hold on
plot(time,datax1,'r--','LineWidth',1.5);
hold off
%xlabel('time(s)')
ylabel('$x_{1}$(rad)','Interpreter','latex','FontSize',12)
legend('MPC','SMPC','NNMPC','FontSize',7)
%title('Simulation','FontSize',14);

subplot(5,1,2);
plot(time,state(2,:),'go-','LineWidth',2);
hold on;
plot(time,stateS(2,:),'b.--','LineWidth',1.7,'MarkerSize',9);
hold on
plot(time,datax2,'r--','LineWidth',1.5);
hold off
%xlabel('time(s)')
ylabel('$x_{2}$(rad)','Interpreter','latex','FontSize',12)
%title('Pendulum Angle (rad)');

subplot(5,1,3);
plot(time,state(3,:),'go-','LineWidth',2);
hold on;
plot(time,stateS(3,:),'b.--','LineWidth',1.7,'MarkerSize',9);
hold on
plot(time,datax3,'r--','LineWidth',1.5);
hold off
%xlabel('time(s)')
ylabel('$x_{3}$(rad/s)','Interpreter','latex','FontSize',12)
%title('Rotary Angular Velocity (rad/s)');

subplot(5,1,4);
plot(time,state(4,:),'go-','LineWidth',2);
hold on;
plot(time,stateS(4,:),'b.--','LineWidth',1.7,'MarkerSize',9);
hold on
plot(time,datax4,'r--','LineWidth',1.5);
hold off
%xlabel('time(s)')
ylabel('$x_{4}$(rad/s)','Interpreter','latex','FontSize',12)
%title('Pendulum Angular Velocity (rad/s)');

subplot(5,1,5);
plot(time,uValue(1,:),'go-','LineWidth',2);
hold on;
plot(time,uValueS(1,:),'b.--','LineWidth',1.7,'MarkerSize',9);
hold on
plot(time,datav1,'r--','LineWidth',1.5);
hold off
xlabel('time(s)','FontSize',12)
ylabel('$v_{1}$(V)','Interpreter','latex','FontSize',12)
%title('Voltage Input (V)');





