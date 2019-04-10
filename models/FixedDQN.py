from collections import deque
import random
import numpy as np
import os
class FixedDQN:
    def __init__(self,config,env,QTrainer,QTargetTrainer,reward_policy,preprocessFunc=None):
        self.QTrainer = QTrainer 
        self.QTargetTrainer = QTargetTrainer
        self.buffer = deque(maxlen=config['dqn_params']['memory_size'])
        self.learning_rate = config['network_params']['lr']
        self.batch_size = config['dqn_params']['batch_size']
        self.pretrain_length = config['dqn_params']['pretrain_length']
        self.env = env
        self.num_actions = self.env.action_space.n
        self.env_state_shape = self.env.observation_space.shape
        self.num_episodes = config['dqn_params']['num_episodes']
        self.steps_per_episode = config['dqn_params']['steps_per_episode']
        self.explore_probability = config['dqn_params']['explore_probability']
        self.gamma = config['dqn_params']['gamma']
        self.render = config['ui_params']['render']
        self.reward_policy = reward_policy
        self.targetScore = config['task_params']['target_score']
        self.targetConsistency = config['task_params']['consistency']
        self.output_weight = os.path.join(config['app_params']['outputdir'],config['app_params']['weightFile'])
        self.explore_decay = config['dqn_params']['explore_decay']
        self.reward_history = deque(maxlen=self.targetConsistency)
        self.tau = config['fixeddqn_params']['tau']
        self.stack_size = config['task_params']['stack_size']
        self.stateType = config['task_params']['state_type']
        self.stateHistory = deque(maxlen=self.stack_size)
        self.preprocessFunc = preprocessFunc
        self.globalStep = 0

    def pretrain(self):
        state = self.env.reset()
        if self.preprocessFunc is not None:
            state = self.preprocessFunc(state)
        for i in range(self.stack_size):
            self.stateHistory.append(state)

        for i in range(self.pretrain_length):
            action = random.choice(range(self.num_actions))
            next_state,reward,terminal,_ = self.env.step(action)
            if terminal:
                next_state = np.zeros(next_state.shape)
            if self.preprocessFunc is not None:
                next_state = self.preprocessFunc(next_state)
            if self.stack_size>0:
                state = np.stack(self.stateHistory)
            self.stateHistory.append(next_state)
            if self.stack_size>0:
                next_state = np.stack(self.stateHistory)
            self.buffer.append([state,action,next_state,reward,terminal])
            state = next_state

    def getAction(self,state):
        toss = random.uniform(0,1)
        if toss < self.explore_probability:
            action = random.choice(range(self.num_actions))
        else:
            action = self.QTrainer.predict(state)
            action = np.argmax(action)
        return action

    def updateQ(self,sample):
        states = np.array([s[0] for s in sample])
        actions = [s[1] for s in sample]
        next_states = np.array([s[2] for s in sample])
        rewards = [s[3] for s in sample]
        terminals = [s[4] for s in sample]
        QNext = self.QTargetTrainer.predict(next_states)
        targets = self.QTrainer.predict(states)
        for i in range(self.batch_size):
            target = rewards[i]
            if not terminals[i]:
                target = rewards[i] + self.gamma*np.max(QNext[i,:])
            targets[i,actions[i]] = target
        return self.QTrainer.train(states,targets)

    def isSolved(self,episodeNumber,totalScore):
        if np.mean(self.reward_history) > self.targetScore:
            return True
        else:
            return False
        if (episodeNumber+1)%self.targetConsistency!=0:
            return False
        return totalScore/self.targetConsistency > self.targetScore

    def train_one_episode(self):
        state = self.env.reset()
        if self.preprocessFunc is not None:
            state = self.preprocessFunc(state)
        for i in range(self.stack_size):
            self.stateHistory.append(state)
        if self.stack_size>0:
            state = np.stack(self.stateHistory)
        stepCount = 0
        terminal = False
        episodeRewards = 0.0
        while stepCount < self.steps_per_episode and not terminal:
            action = self.getAction(state)
            next_state,reward,terminal,_ = self.env.step(action)
            if terminal:
                next_state = np.zeros(next_state.shape)
            episodeRewards += reward
            reward = self.reward_policy(next_state,reward)
            if self.preprocessFunc is not None:
                next_state = self.preprocessFunc(next_state)
            if self.stack_size>0:
                state = np.stack(self.stateHistory)
            self.stateHistory.append(next_state)
            if self.stack_size>0:
                next_state = np.stack(self.stateHistory)
            self.buffer.append((state,action,next_state,reward,terminal))
            state = next_state

            if self.globalStep%self.tau==0 and self.tau!=-1:
                self.QTargetTrainer.copy(self.QTrainer.network)
            stepCount += 1
            self.globalStep += 1
            sample = random.sample(self.buffer,self.batch_size)
            loss = self.updateQ(sample)
            self.explore_probability = max(0.001,self.explore_probability*self.explore_decay)
        if self.render:
            self.env.render()
        return loss,episodeRewards
        
    def train(self):
        totalScore = 0
        for i in range(self.num_episodes):
            if self.tau!=-1:
                self.QTargetTrainer.copy(self.QTrainer.network)
            if self.render:
                self.env.render()
            if self.isSolved(i,totalScore):
                break
            if i%self.targetConsistency==0:
                totalScore = 0
            QLoss,episodeRewards = self.train_one_episode()
            self.reward_history.append(episodeRewards)
            totalScore += episodeRewards
            print("Episode: {} Rewards: {} Explore: {} Average: {} Loss: {}".format(i,episodeRewards,self.explore_probability,np.mean(self.reward_history),QLoss))

        self.env.close()
        self.QTrainer.save(self.output_weight)
    
    def run(self):
        done = False
        state = self.env.reset()
        while not done:
            self.env.render()
            action = self.QTrainer.predict(state)
            action = np.argmax(action)
            next_state,reward,done,_ = self.env.step(action)
            state = next_state
        self.env.close()
