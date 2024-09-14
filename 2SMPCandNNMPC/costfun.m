function cost = costfun(state,ref1,input)
%COSTFUN Summary of this function goes here
%   Detailed explanation goes here


cost= - 5*(state(1)-ref1)^2 - 5*(state(2))^2 - 0.5*(input(1))^2;

end

