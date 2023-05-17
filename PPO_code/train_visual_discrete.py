import numpy as np
import gym
from utils import *
from agent import *
from config_discrete import *

def train(env, agent, num_episode, eps_init, eps_decay, eps_min, max_t, num_frame=2, constant=0):
    rewards_log = []
    average_log = []
    state_history = []
    action_history = []
    done_history = []
    reward_history = []

    for i in range(1, n_episode+1): 
        
        frame = env.reset()
        frame = preprocess(frame, constant)
        state_deque = deque(maxlen=num_frame)
        for _ in range(num_frame):
            state_deque.append(frame)
        state = np.stack(state_deque, axis=0)
        state = np.expand_dims(state, axis=0)

        episodic_reward = 0
        done = False
        t = 0
        
        if len(state_history) == 0:
            state_history.append(state)
        else:
            state_history[-1] = state

       while not done and t < max_t:
            action = agent.act(state)
            frame, reward, done, _ = env.step(action)
            frame = preprocess(frame, constant)
            state_deque.append(frame)
            next_state = np.stack(state_deque, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            
            episodic_reward += reward
            action_history.append(action)
            done_history.append(done)
            reward_history.append(reward * scale)
            state = next_state.copy()
            episodic_reward += reward

        if i % update_frequency == 0:
            states, actions, log_probs, rewards, dones = agent.process_data(state_history, action_history, reward_history, done_history, 64)
            for _ in range(n_update):
                agent.learn(states, actions, log_probs, rewards, dones)
            state_history = []
            action_history = []
            done_history = []
            reward_history = []
        
        rewards_log.append(episodic_reward)
        average_log.append(np.mean(rewards_log[-100:]))

    return rewards_log

if __name__ == '__main__':
    env = gym.make(VISUAL_ENV_NAME)
    agent = Agent_discrete(state_size=env.observation_space.shape[0], 
                           action_size=env.action_space.n, 
                           lr=LEARNING_RATE, 
                           beta=BETA, 
                           eps=EPS, 
                           tau=TAU, 
                           gamma=GAMMA, 
                           device=DEVICE,
                           hidden=HIDDEN_DISCRETE,
                           share=SHARE, 
                           mode=MODE, 
                           use_critic=CRITIC, 
                           normalize=NORMALIZE)
    rewards_log, _ = train(agent=agent, 
                           env=env, 
                           n_episode=RAM_NUM_EPISODE, 
                           n_update=N_UPDATE, 
                           update_frequency=UPDATE_FREQUENCY, 
                           max_t=MAX_T, 
                           scale=SCALE)
    np.save('{}_rewards.npy'.format(VISUAL_ENV_NAME), rewards_log)