import gym
import sys
from yaml import load
from yaml import CLoader as Loader, CDumper as Dumper

sys.path.append("../")
config = load(open(sys.argv[1],'r'),Loader)

env = gym.make(config['environment'])
from networks.QNetworks import DQN_fc
from networks.trainer import Trainer
from models.DQN import DQN
from models.FixedDQN import FixedDQN
import reward_policies

import os
if not os.path.exists(config['app_params']['outputdir']):
    os.makedirs(config['app_params']['outputdir'])

QNetwork = DQN_fc(env.observation_space.shape,env.action_space.n).cuda()
QTrainer = Trainer(QNetwork,config)
if not config['train']:
    QTrainer.load(os.path.join(config['app_params']['outputdir'],config['app_params']['weightFile']))

QTargetNetwork = DQN_fc(env.observation_space.shape,env.action_space.n).cuda()
QTargetTrainer = Trainer(QTargetNetwork,config)
agent = FixedDQN(config,env,QTrainer,QTargetTrainer,reward_policies.identity)
if config['train']:
    agent.pretrain()
    agent.train()
else:
    agent.run()