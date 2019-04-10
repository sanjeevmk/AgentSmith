import gym
import sys
from yaml import load
from yaml import CLoader as Loader, CDumper as Dumper

sys.path.append("../")
config = load(open(sys.argv[1],'r'),Loader)
from utils import utils

env = gym.make(config['environment'])
from networks.QNetworks import DQN_image
from networks.trainer import ImageTrainer
from models.DQN import DQN
from models.FixedDQN import FixedDQN
import reward_policies

import os
if not os.path.exists(config['app_params']['outputdir']):
    os.makedirs(config['app_params']['outputdir'])

QNetwork = DQN_image(config['task_params']['stack_size'],env.action_space.n).cuda()
QTrainer = ImageTrainer(QNetwork,config)
if not config['train']:
    QTrainer.load(os.path.join(config['app_params']['outputdir'],config['app_params']['weightFile']))

def stateProcessor(image):
    image = utils.rgb2gray(image)
    image = utils.resize(image)
    image = utils.scale01(image)
    return image

QTargetNetwork = DQN_image(config['task_params']['stack_size'],env.action_space.n).cuda()
QTargetTrainer = ImageTrainer(QTargetNetwork,config)
agent = FixedDQN(config,env,QTrainer,QTargetTrainer,reward_policies.identity,preprocessFunc=stateProcessor)
if config['train']:
    agent.pretrain()
    agent.train()
else:
    agent.run()