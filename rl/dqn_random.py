from common.gym_env.GazeboGymWrapper import UpTheStairsEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import rospy
import rospkg
import time
import yaml
from datetime import datetime
import numpy as np
import os

class RL_Node:
    def __init__(self, episode_length, seed=42):
        self.max_episode_length = episode_length
        rospack = rospkg.RosPack()
        self.path_to_pkg = rospack.get_path('up_the_stairs')

        now = datetime.now()
        folder_name = 'dqn_random_' + now.strftime("%m_%d_%Y_%H_%M_%S") 

        with open(path_to_pkg + '/rl/dqn_random.yaml', 'r') as file:
            OPTIONS = yaml.safe_load(file)

        np.random.seed(seed)
        magnitudes = np.random.randint(low=OPTIONS['train']['parameters']['min_magnitude'],
                                       high=OPTIONS['train']['parameters']['max_magnitude'], 
                                       size=OPTIONS['train']['number_prototypes'])
        magnitudes = magnitudes + np.random.uniform(low=0.0, 
                                                    high=1.0, 
                                                    size=OPTIONS['train']['number_prototypes'])
        
        angles = np.random.randint(low=OPTIONS['train']['parameters']['min_angle'],
                                   high=OPTIONS['train']['parameters']['max_angle'], 
                                   size=OPTIONS['train']['number_prototypes'])
        
        prototypes = np.vstack((magnitudes, angles)).T

        self.env = UpTheStairsEnv(episode_length, prototypes)
        self.model = DQN("MlpPolicy", self.env, verbose=1, learning_starts=3000, seed=seed)


        self.log_path = self.path_to_pkg + '/log_gym/' + folder_name + "/"
        print("episode length:", episode_length)

    def start_learn(self, total_timesteps=8000, eval_frequency=100):
        # self.eval_callback = EvalCallback(self.env, best_model_save_path=self.log_path,
        #                              log_path=self.log_path, eval_freq=eval_frequency,
        #                              deterministic=False, render=False)
        self.checkpoint_callback = CheckpointCallback(save_freq=eval_frequency, save_path=self.log_path,
                                         name_prefix='dqn_random')
        print("total_timesteps:", total_timesteps)
        self.start_time = time.time()
        self.model.learn(total_timesteps=total_timesteps, callback=self.checkpoint_callback) 
        self.end_time = time.time()
        rospy.logerr("Total training time:" + str(self.end_time - self.start_time))
        self.model.save(self.log_path + "model")

    def shutdown_hook(self):
        self.end_time = time.time()
        rospy.logerr("Total training time:" + str(self.end_time - self.start_time))
        rospy.loginfo('Shutdown requested.')
        rospy.loginfo("Saving data before shutting down...")
        self.model.save(self.log_path + "model")

    def show_run(self, folder_collection, runs, seeds):
        for seed, folder in zip(seeds, folder_collection):
            np.random.seed(seed)

            with open(path_to_pkg + '/rl/dqn_random.yaml', 'r') as file:
                OPTIONS = yaml.safe_load(file)

            magnitudes = np.random.randint(low=OPTIONS['train']['parameters']['min_magnitude'],
                                           high=OPTIONS['train']['parameters']['max_magnitude'], 
                                           size=OPTIONS['train']['number_prototypes'])
            magnitudes = magnitudes + np.random.uniform(low=0.0, 
                                                        high=1.0, 
                                                        size=OPTIONS['train']['number_prototypes'])

            angles = np.random.randint(low=OPTIONS['train']['parameters']['min_angle'],
                                       high=OPTIONS['train']['parameters']['max_angle'], 
                                       size=OPTIONS['train']['number_prototypes'])

            prototypes = np.vstack((magnitudes, angles)).T

            self.env = UpTheStairsEnv(self.max_episode_length, prototypes)



            models = [model for model in os.listdir(os.path.join(self.path_to_pkg,'log_gym',folder)) if model.endswith(".zip") and '_' in model]
            sorted_by = [int(model.split('_')[-2]) for model in models ]
            index_sort = np.argsort(sorted_by)

            timesteps = []
            results = []
            ep_lengths = []

            for idx in index_sort: #timesteps
                timesteps.append(sorted_by[idx])
                self.model = DQN.load(os.path.join(self.path_to_pkg,'log_gym',folder, models[idx]),
                                      custom_objects={'action_space': self.env.action_space,
                                                      'observation_space': self.env.observation_space})

                print("Total parameters: ", sum(p.numel() for p in self.model.q_net.parameters()))
                single_run_ep_length = []
                single_run_results = []

                for i in range(runs): # multiple runs per timestep
                    rospy.loginfo("Starting timestep " + str(sorted_by[idx]) + " run " + str(i + 1) + "/" + str(runs))

                    obs, _ = self.env.reset() 
                    accumulated_reward = 0
                    episode_length = 0

                    while True:
                        action, _states = self.model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = self.env.step(action)

                        episode_length += 1
                        accumulated_reward += reward
                        if terminated:
                            rospy.loginfo("Run ended with reward: "+ str(accumulated_reward))
                            rospy.loginfo("Episode length "+str(episode_length))
                            single_run_ep_length.append(episode_length)
                            single_run_results.append(accumulated_reward)
                            break
                            obs = self.env.reset()

                results.append(single_run_results)
                ep_lengths.append(single_run_ep_length)

            # Store eval
            timesteps = np.array(timesteps)
            results = np.array(results)
            ep_lengths = np.array(ep_lengths)

            np.savez(os.path.join(self.path_to_pkg,'log_gym',folder,'evaluations'), 
                     timesteps=timesteps, 
                     results=results, 
                     ep_lengths=ep_lengths)


if __name__ == '__main__':
    rospy.init_node('elsa_dqn_random') 
    rate = rospy.Rate(100)
    rospack = rospkg.RosPack()
    path_to_pkg = rospack.get_path('up_the_stairs')
    with open(path_to_pkg + '/rl/dqn_random.yaml', 'r') as file:
        OPTIONS = yaml.safe_load(file)

    
    
    if OPTIONS['mode'] == 'train':
        for seed in OPTIONS['train']['seeds']:
            node = RL_Node(OPTIONS['episode_length'], seed)
            rospy.on_shutdown(node.shutdown_hook)
            node.start_learn(total_timesteps=OPTIONS['train']['total_timesteps'], 
                             eval_frequency=OPTIONS['train']['eval_frequency'])
    elif OPTIONS['mode'] == 'eval':
        node = RL_Node(OPTIONS['episode_length'], OPTIONS['run']['seed'])
        node.show_run(OPTIONS['run']['model_folder'],
                      OPTIONS['run']['runs'],
                      OPTIONS['run']['seeds'])

    