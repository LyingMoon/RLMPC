%% Motor
% Resistance
Rm = 9.7; % nominal value = 8.4;
% Current-torque (N-m/A)
kt = 0.042;
% Back-emf constant (V-s/rad)
km = kt;
%
%% Rotary Arm
% Mass (kg)
mr = 0.095*1.1;
% Total length (m)
r = 0.085;
% Moment of inertia about pivot (kg-m^2)
Jr = mr*r^2/3;
% Equivalent Viscous Damping Coefficient (N-m-s/rad)
br = 1e-3*2; % damping tuned heuristically to match QUBE-Servo 2 response
% 
%% Pendulum Link
% Mass (kg)
mp = 0.024*1.1;
% Total length (m)
Lp = 0.129;
% Pendulum center of mass (m)
l = Lp/2;
% Moment of inertia about pivot (kg-m^2)
Jp = mp*Lp^2/3;
% Equivalent Viscous Damping Coefficient (N-m-s/rad)
bp = 5e-5*2; % damping tuned heuristically to match QUBE-Sero 2 response
% Gravity Constant
g = 9.81;
