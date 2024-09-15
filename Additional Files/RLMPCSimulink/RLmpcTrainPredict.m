% Load pre-defined agent (false) or train new agent (true)
doTraining = true;

% load QUBE-Servo 2 Pendulum parameters (with defined parameter variation)
run("Copy_of_qube2_rotpen_param.m");
% initial pendulum angle (rad)
alpha0 = pi; % set to pi to start pendulum in inverted position, 0 for down position
% Maximum voltage (V)
Vmax = 2;
% max rotary arm angle +/- 60 deg
theta_max = 2;%120*pi/180;
% inverted pendulum angle balance threshold (rad)
alpha_bal_threshold = 60*pi/180;
% reward QR weights
q11 = 5; q22 = 5; q33 = 0; q44 = 0; UR = 0.1; UVR=0; B = -100; % agent 9
lowlimit=-inf*ones(124,1);
upplimit=-lowlimit;
% observation signals "rlNumericSpec" creates action/observative data of a given dimension and signal limits
obsInfo = rlNumericSpec([124 1],'LowerLimit',lowlimit,'UpperLimit',upplimit);
obsInfo.Name = 'observations';
%obsInfo.Description = 'theta,alpha,theta_dot,alpha_dot,r1,r2,r3,r4';
numObs = obsInfo.Dimension(1);
% action signals - constrained to motor limit
actInfo = rlNumericSpec([1 1],'LowerLimit',-Vmax,'UpperLimit',Vmax);
actInfo.Name = 'Motor Voltage';
% numActions = actInfo.Dimension(1);
% create environment object
mdl = 'MPCRLpredict';
agentBlk = [mdl '/RLmpc/RL Agent'];
env = rlSimulinkEnv(mdl,agentBlk,obsInfo,actInfo);
% reset function used to randomize initial position of pendulum
env.ResetFcn = @(in)localResetFcn(in);
% Sampling rate
Ts = 0.005;
% Simulation duration
Tf = 7;
% Fix the random generator seed for reproducibility.
rng(0);

statePath = [
    imageInputLayer([numObs 1 1],'Normalization','none','Name','observation')
    fullyConnectedLayer(128,'Name','CriticStateFC1')
    reluLayer('Name', 'CriticRelu1')
    fullyConnectedLayer(200,'Name','CriticStateFC2')];
actionPath = [
    imageInputLayer([1 1 1],'Normalization','none','Name','action')
    fullyConnectedLayer(200,'Name','CriticActionFC1')];
commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','CriticOutput')];
% 
criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
%  
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');
% plot network
plot(criticNetwork)
% 
% Specify options for the critic representation .
criticOpts = rlRepresentationOptions('LearnRate',1e-3,'GradientThreshold',1);
% 
% Create critic representation using the specified deep neural network 
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'observation'},'Action',{'action'},criticOpts);
% 
% Create the actor, first create a deep neural network with one input, 
% the observation, and one output, the action.
% Construct actor similarly to the critic. 
actorNetwork = [
    imageInputLayer([numObs 1 1],'Normalization','none','Name','observation')
    fullyConnectedLayer(128,'Name','ActorFC1')
    reluLayer('Name','ActorRelu1')
    fullyConnectedLayer(200,'Name','ActorFC2')
    reluLayer('Name','ActorRelu2')
    fullyConnectedLayer(1,'Name','ActorFC3')
    tanhLayer('Name','ActorTanh')
    scalingLayer('Name','ActorScaling','Scale',max(actInfo.UpperLimit))];
% 
actorOpts = rlRepresentationOptions('LearnRate',5e-04,'GradientThreshold',1);
% 
actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'observation'},'Action',{'ActorScaling'},actorOpts);
% specify agent options
agentOpts = rlDDPGAgentOptions(...
    'SampleTime',Ts,...
    'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',1e6,...
    'DiscountFactor',0.99,...
    'MiniBatchSize',128);
% For continuous action signals, it is important to set the noise variance 
% appropriately to encourage exploration. It is common to have 
% Variance*sqrt(Ts) be between 1% and 10% of your action range
agentOpts.NoiseOptions.Variance = 0.2; 
agentOpts.NoiseOptions.VarianceDecayRate = 1e-5;
% 
% create the DDPG agent using the specified actor representation, critic 
% representation and agent options.
agent = rlDDPGAgent(actor,critic,agentOpts);

% Maximum # of steps for training
maxsteps = ceil(Tf/Ts);
% Setup up training parametesr
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',3000, ...
    'MaxStepsPerEpisode',maxsteps, ...
    'ScoreAveragingWindowLength',5, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',-35);
% 
if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
    % Save trained agent
    save('QubeIPBalDDPGAgentCost3.mat','agent');
else
    % Load pretrained agent.
    load('QubeIPBalDDPG09.mat','agent');
    % load('QubeIPBalDDPGNominalParam.mat','agent');
end


function in = localResetFcn(in)
    % randomize initial inverted position angle
    blk = sprintf('MPCRLpredict/IC0');
    blk2 = sprintf('MPCRLpredict/ic22');
    blk3 = sprintf('MPCRLpredict/ic33');
    % initialize angle of pendulum +/- 10 deg about vertical upright position
    ic0 = ( 60 * ( rand()-0.5) + 180 ) * pi/180;
    in = setBlockParameter(in,blk,'Value',num2str(ic0));
    ic22=rand()-0.5;
    in = setBlockParameter(in,blk2,'Value',num2str(ic22));
    ic33=rand();
    in = setBlockParameter(in,blk3,'Value',num2str(ic33));
end