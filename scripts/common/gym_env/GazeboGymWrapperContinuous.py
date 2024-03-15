import gym
from gym import spaces
import rospy
from gazebo_msgs.srv import SetModelState, GetModelState
from up_the_stairs.srv import Jump, JumpRequest, BackToStart
import numpy as np
from rospy_message_converter import message_converter

class UpTheStairsContinuousEnv(gym.Env):
    """Custom Environment that follows gym interface for stair climbing task with continuous actions
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, episode_length):
        super(UpTheStairsContinuousEnv, self).__init__()

        # Initialize Gazebo Services
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.set_model_state.wait_for_service()

        self.get_model_state = rospy.ServiceProxy( '/gazebo/get_model_state', GetModelState)
        self.get_model_state.wait_for_service()

        self.back_to_start = rospy.ServiceProxy( '/up_the_stairs/back_to_start', BackToStart)
        self.back_to_start.wait_for_service()

        self.jumper = rospy.ServiceProxy('/up_the_stairs/jumper', Jump)
        self.jumper.wait_for_service()


        self.n_steps = 0

        self.EPISODE_LENGTH = episode_length

        self.action_space = spaces.Box(low=np.array([50.0, 1.0]), high=np.array([1000.0, 89.0]))  # n_prototypes is the number of actions

        n_dimensions = 4
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(n_dimensions, ), dtype=np.float16)

    def step(self, action):
        # Execute one time step within the environment
        terminated = False
        truncated = False
        model_state = self.get_model_state('bd_droid', 'world')
        pose_pre_jump = model_state.pose
        jump_ex = JumpRequest()
        jump_ex.magnitude=action[0]
        jump_ex.angle=action[1]
        self.jumper(jump_ex)
        model_state = self.get_model_state('bd_droid', 'world')
        pose_post_jump = model_state.pose

        _, _, s2, effect = self.effect_calculator(action, pose_pre_jump, pose_post_jump)

        observation = np.append(s2[:3], [self.distance_to_step_sensor(s2)])
        reward = self.custom_reward(effect)
        if self.n_steps == self.EPISODE_LENGTH:
            terminated = True
            truncated = True
            info = {"TimeLimit.truncated": True}
            self.n_steps = 0
        else:
            terminated = self.check_goal(s2)
            if terminated:
                self.n_steps = 0
            else:
                self.n_steps += 1
            info = {"TimeLimit.truncated": False}

        # Must return: observation, reward, done, info
        return observation, reward, terminated, truncated, info

    def reset(self, seed=0, options=None):
        # Reset the state of the environment to an initial state
        self.back_to_start()
        rospy.sleep(1)
        self.n_steps = 0
        observation = np.array([0.0, 0.25, 0.05, 0.35])
        return observation, {}  # return initial observation

    def render(self, mode='human'):
        '''
        render is automatic if Gazebo gui is running
        '''
        pass

    def close(self):
        # Override close if additional cleanup is necessary
        pass 

    def custom_reward(self, effect):
        if effect[2] > 0.05:
            return 1 
        elif effect[2] < -0.05:
            return effect[2] / 0.3
        
        return 0 
        
    def check_goal(self, state):
        if state[2] >= 5.2 and 6.2 <= state[1]:
            return True 
        # Check if Robot is on the stairs 
        if 0.0 < state[1] < 10:
            if -2.46 < state[0] < 2.47:
                if 0.0 < state[2]:
                    return False
                else:
                    return True
            else: 
                return True
        else: 
            return True

    def effect_calculator(self, behaviour, s_1, s_2):
        dict_s1 = message_converter.convert_ros_message_to_dictionary(s_1)
        dict_s2 = message_converter.convert_ros_message_to_dictionary(s_2)
        s1 = np.array(list(dict_s1['position'].values()) + list(dict_s1['orientation'].values()))
        s2 = np.array(list(dict_s2['position'].values()) + list(dict_s2['orientation'].values()))
        effect = s2 - s1
        return behaviour, s1, s2, effect
        
    

    def action_prototypes(self, action):
        return self.prototypes[action]
    

    def distance_to_step_sensor(self, state):
        y_robot = state[1]
        step_pos = 0.6

        while y_robot > step_pos:
            if step_pos > 6.2:
                step_pos = 10
                break
            step_pos += 0.4
        
        if step_pos == 10:
            return 10
        else:
            return step_pos - y_robot

