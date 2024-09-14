%% Motor
% Resistance
Rm = 8.4;
% Current-torque (N-m/A)
kt = 0.042;
% Back-emf constant (V-s/rad)
km = 0.042;
%
%% Rotary Arm
% Mass (kg)
mr = 0.095;
% Total length (m)
r = 0.085;
% Moment of inertia about pivot (kg-m^2)
Jr = mr*r^2/3;
% Equivalent Viscous Damping Coefficient (N-m-s/rad)
br = 1e-3; % damping tuned heuristically to match QUBE-Sero 2 response
%
%% Pendulum Link
% Mass (kg)
mp = 0.024;
% Total length (m)
Lp = 0.129;
% Pendulum center of mass (m)
l = Lp/2;
% Moment of inertia about pivot (kg-m^2)
Jp = mp*Lp^2/3;
% Equivalent Viscous Damping Coefficient (N-m-s/rad)
bp = 5e-5; % damping tuned heuristically to match QUBE-Sero 2 response
% Gravity Constant
g = 9.81;

%% State Space Model
% Find Total Inertia
Jt = Jr*Jp - mp^2*r^2*l^2;
% 
% State Space Representation
A = [0 0 1 0;
     0 0 0 1;
     0 mp^2*l^2*r*g/Jt  -br*Jp/Jt   -mp*l*r*bp/Jt 
     0  mp*g*l*Jr/Jt    -mp*l*r*br/Jt   -Jr*bp/Jt];
%
B = [0; 0; Jp/Jt; mp*l*r/Jt];
% Assume Full State feedback
C = eye(4,4);
D = zeros(4,1);
% 
% Add actuator dynamics
A(3,3) = A(3,3) - km*km/Rm*B(3);
A(4,3) = A(4,3) - km*km/Rm*B(4);
B = km * B / Rm;

% Sample time = Ts (second)
Ts=0.01;

% Set up the discrete state space model
[Ad,Bd] = c2d(A,B,Ts);

sys = ss(Ad,Bd,C,D,0.01);

% Then Open the MPC designer, where you can change the weight and different
% parameters of MPC
mpcDesigner
