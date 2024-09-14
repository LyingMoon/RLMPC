predictionTime = 0.15;
Cts = 0.05;
Ts = 0.01;
NNPredictStep =3;%predictionTime/Cts;
mpcPredictStep = predictionTime/Ts;
diff = mpcPredictStep/NNPredictStep;

refx1= zeros(1,NNPredictStep+1);

refx1mpc= zeros(1,mpcPredictStep);

refx1smpc = zeros(1,mpcPredictStep);

weight=1;
i=150;
feq=0.2;

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

refmpc=refx1mpc;
refsmpc=refx1smpc;
Timempc=linspace(i*Ts,(i+mpcPredictStep-1)*Ts,mpcPredictStep);
TimeCP=[i*Ts,i*Ts+0.05,i*Ts+0.1,i*Ts+0.15];

plot(Timempc,refmpc,'m--x','LineWidth',1.5,'MarkerSize',8)
hold on
plot(Timempc,refsmpc,"b--x",'LineWidth',1.5,'MarkerSize',8)
hold on
plot(TimeCP,refx1,"ro",'MarkerSize',12,'LineWidth',1.5)
xlabel('Time(s)')
ylabel('$f_{ref}(t)$','Interpreter','latex')
legend('$W$','$W_{ext}$','$C$','Interpreter','latex')
hold off
