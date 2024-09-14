clear all;

% The data gathered from RL + MPC
load rlmpcv1.csv;
load rlmpcx1.csv;
load rlmpcx2.csv;
load rlmpcx3.csv;
load rlmpcx4.csv;

% The data gathered from Warm Start RL
load rlinimpcv1.csv;
load rlinimpcx1.csv;
load rlinimpcx2.csv;
load rlinimpcx3.csv;
load rlinimpcx4.csv;

% The data gathered from real world MPC
load MPCRW.mat

% If you would like to compared the NNMPC,
% Uncomment following lines and change the code in the plot part
% load nnmpcv1.csv;
% load nnmpcx1.csv;
% load nnmpcx2.csv;
% load nnmpcx3.csv;
% load nnmpcx4.csv;

time=linspace(700,1399,700)*0.01;
state=States';

subplot(5,1,1);
plot(time,state(1,700:1399),'go-','LineWidth',2);
hold on;
plot(time,rlinimpcx1,'b.--','LineWidth',1.7,'MarkerSize',9);
hold on
plot(time,rlmpcx1,'r--','LineWidth',1.5);
hold off
ylabel('$x_{1}$(rad)','Interpreter','latex','FontSize',12)
legend('MPC','Warm Start RL','RL + MPC','FontSize',7)

subplot(5,1,2);
plot(time,state(2,700:1399),'go-','LineWidth',2);
hold on;
plot(time,rlinimpcx2,'b.--','LineWidth',1.7,'MarkerSize',9);
hold on
plot(time,rlmpcx2,'r--','LineWidth',1.5);
hold off
ylabel('$x_{2}$(rad)','Interpreter','latex','FontSize',12)

subplot(5,1,3);
plot(time,state(3,700:1399),'go-','LineWidth',2);
hold on;
plot(time,rlinimpcx3,'b.--','LineWidth',1.7,'MarkerSize',9);
hold on
plot(time,rlmpcx3,'r--','LineWidth',1.5);
hold off
ylabel('$x_{3}$(rad/s)','Interpreter','latex','FontSize',12)

subplot(5,1,4);
plot(time,state(4,700:1399),'go-','LineWidth',2);
hold on;
plot(time,rlinimpcx4,'b.--','LineWidth',1.7,'MarkerSize',9);
hold on
plot(time,rlmpcx4,'r--','LineWidth',1.5);
hold off
ylabel('$x_{4}$(rad/s)','Interpreter','latex','FontSize',12)

subplot(5,1,5);
plot(time,-Voltage(700:1399),'go-','LineWidth',2);
hold on;
plot(time,rlinimpcv1,'b.--','LineWidth',1.7,'MarkerSize',9);
hold on
plot(time,rlmpcv1,'r--','LineWidth',1.5);
hold off
xlabel('time(s)','FontSize',12)
ylabel('$v_{1}$(V)','Interpreter','latex','FontSize',12)


