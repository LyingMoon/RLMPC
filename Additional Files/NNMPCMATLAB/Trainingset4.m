%MPCmove, generate training set


x1List=[-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1];
x2List=[-0.3,-0.25,-0.175,-0.125,-0.0625,-0.03,0,0.03,0.0625,0.125,0.175,0.25,0.3];
x3List=[-3,-2,-1,-0.5,-0.25,0,0.25,0.5,1,2,3];
x4List=[-3,-2,-1,-0.5,-0.25,0,0.25,0.5,1,2,3];

[x1,x2,x3,x4]=ndgrid(x1List,x2List,x3List,x4List);
N=length(x1(:));
x1=x1(:);
x2=x2(:);
x3=x3(:);
x4=x4(:);

refSignal=rand(N,120)*2.5-1.25;

tic
for (k=1:N)
    x=mpcstate(mpc1,[x1(k),x2(k),x3(k),x4(k)]);
    
    uValue(k)=mpcmove(mpc1,x,[],[refSignal(k,:);zeros(3,120)]');
end
toc

INPUT4=[x1,x2,x3,x4,refSignal];
OUTPUT4=uValue(:);