import rospy
from gazebo_msgs.srv import SetModelState, GetModelState
from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3
from gazebo_msgs.msg import ModelState
from up_the_stairs.srv import Rollout, RolloutGaussian, RolloutGaussianResponse, RolloutResponse, Jump, JumpRequest, SetStart
import random
from rospy_message_converter import message_converter
import numpy as np
import pandas as pd
from datetime import datetime

class UpTheStairsData:
    def __init__(self):
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.set_model_state.wait_for_service()

        self.get_model_state = rospy.ServiceProxy( '/gazebo/get_model_state', GetModelState)
        self.get_model_state.wait_for_service()

        self.set_start = rospy.ServiceProxy( '/up_the_stairs/set_start', SetStart)
        self.set_start.wait_for_service()

        self.jumper = rospy.ServiceProxy('/up_the_stairs/jumper', Jump)
        self.jumper.wait_for_service()

        self.collected_data_df = pd.DataFrame(columns=['magnitude', 'angle',
                                                  'env_x', 'env_y', 'env_z',
                                                  'env_rot_x', 'env_rot_y', 'env_rot_z', 'env_rot_w',
                                                  'dx', 'dy', 'dz', 'd_rot_x', 'd_rot_y', 'd_rot_z', 'd_rot_w', 'robot_in_map'])

        self.FILENAME = ''

        self.storage_path = None

    def robot_to_start(self):
        success = True
        try:
            model_state_msg = ModelState()
            model_state_msg.model_name='bd_droid'
            model_state_msg.reference_frame='world'
            point_msg = Point(x=0.0,y=0.25,z=0.05)
            quat_msg = Quaternion(x=0.0,y=0.0,z=0.0,w=1.0)
            model_state_msg.pose = Pose(position=point_msg,orientation=quat_msg)
            vector_msg = Vector3(x=0.0,y=0.0,z=0.0)
            model_state_msg.twist=Twist(linear=vector_msg, angular=vector_msg)
            self.set_model_state(model_state_msg)
        except rospy.ServiceException as e:
                rospy.logerr('set_model_state service call failed: {0}'.format(e))
                success = False
        rospy.loginfo('Models reset')
        return success
    
    def pick_start(self):
        x = (random.random()-0.5)
        y = random.random() * 2
        z = 1.8
        point_msg = Point(x=x,y=y,z=z)
        quat_msg = Quaternion(x=0.0,y=0.0,z=0.0,w=1.0)

        return Pose(position=point_msg, orientation=quat_msg)
    
    def robot_in_map(self):
        model_state = self.get_model_state('bd_droid', 'world')
        pose = model_state.pose
        reset = True
        # Check if Robot is on the stairs 
        if 0.0 < pose.position.y < 4.8:#2.16:
            if -2.46 < pose.position.x < 2.47:
                if 0.0 < pose.position.z < 3.2:#1.25:
                    reset = False
                    rospy.loginfo("In Box")
                    return True
        if reset:
            rospy.loginfo("Outside of Box!")
            return False

    def rollout(self, request):
        request.storage_path = request.storage_path
        now = datetime.now()
        self.FILENAME = now.strftime("%m_%d_%Y__%H_%M_%S") 
        with open(request.storage_path + self.FILENAME + '.txt', 'w') as f:
            f.writelines('uniform \ntrys: ' + str(request.trys) +
                      '\nmax_f: ' + str(request.max_force) +
                      '\nmin_f: ' + str(request.min_force) +
                      '\nmax_angle: ' + str(request.max_angle) +
                      '\nmin_angle: ' + str(request.min_angle)) 
        jumps = 0
        self.collected_data_df = self.collected_data_df[0:0]
        while jumps < request.trys:
            force = request.min_force + (random.random() * (request.max_force - request.min_force))
            angle = random.randint(request.min_angle,request.max_angle)
            rospy.logwarn("Force: " + str(force) + " Angle: " + str(angle))
            # if not self.robot_in_map():
            pose_start = self.pick_start()
            self.set_start(pose_start)
            rospy.sleep(1)
            model_state = self.get_model_state('bd_droid', 'world')
            pose_pre_jump = model_state.pose
            jump_ex = JumpRequest()
            jump_ex.magnitude=force
            jump_ex.angle=angle
            self.jumper(jump_ex)
            model_state = self.get_model_state('bd_droid', 'world')
            pose_post_jump = model_state.pose
            effect = self.effect_calculator([force, angle], pose_pre_jump, pose_post_jump)
            jumps += 1
            rospy.logerr(str(jumps) + " out of " + str(request.trys) + " done...")
        
        if request.storage_path != '':
            self.collected_data_df.to_csv(request.storage_path + self.FILENAME + '.csv')
            

        return True

    def rollout_gaussian(self, request):
        self.storage_path = request.storage_path
        now = datetime.now()
        self.FILENAME = now.strftime("%m_%d_%Y__%H_%M_%S") 
        with open(request.storage_path + self.FILENAME + '.txt', 'w') as f:
            f.writelines('Gaussian \ntrys: ' + str(request.trys) +
                      '\nmean_f: ' + str(request.mean_force) +
                      '\nstd_f: ' + str(request.std_force) +
                      '\nmean_angle: ' + str(request.mean_angle) +
                      '\nstd_angle: ' + str(request.std_angle)) 
        jumps = 0
        self.collected_data_df = self.collected_data_df[0:0]
        while jumps < request.trys:
            force = np.random.normal(request.mean_force, request.std_force)
            angle = np.random.normal(request.mean_angle, request.std_angle)
            # if not self.robot_in_map():
            self.robot_to_start()
            model_state = self.get_model_state('bd_droid', 'world')
            pose_pre_jump = model_state.pose
            jump_ex = JumpRequest()
            jump_ex.magnitude=force
            jump_ex.angle=angle
            self.jumper(jump_ex)
            model_state = self.get_model_state('bd_droid', 'world')
            pose_post_jump = model_state.pose
            effect = self.effect_calculator([force, angle], pose_pre_jump, pose_post_jump)
            jumps += 1
            rospy.logerr(str(jumps) + " out of " + str(request.trys) + " done...")
        
        if request.storage_path != '':
            self.collected_data_df.to_csv(request.storage_path + self.FILENAME + '.csv')
            

        return True

    def callback(self, request):
        rospy.logerr("Starting Rollout with try count:"+str(request.trys))
        done = self.rollout(request)
        return RolloutResponse(done)
    
    def callback_gaussian(self, request):
        rospy.logerr("Starting Rollout with try count:"+str(request.trys))
        done = self.rollout_gaussian(request)
        return RolloutGaussianResponse(done)
    
    def effect_calculator(self, behaviour, s_1, s_2):
        affordance = behaviour
        dict_s1 = message_converter.convert_ros_message_to_dictionary(s_1)
        dict_s2 = message_converter.convert_ros_message_to_dictionary(s_2)
        s1_list = list(dict_s1['position'].values()) + list(dict_s1['orientation'].values())
        s2_list = list(dict_s2['position'].values()) + list(dict_s2['orientation'].values())
        affordance = affordance + s1_list + list(np.array(s2_list) - np.array(s1_list)) + [self.robot_in_map()] # (b, env, effect)
        rospy.logerr(affordance)
        # rospy.logwarn(self.collected_data_df)
        self.collected_data_df.loc[len(self.collected_data_df)] = affordance

    
    # def shutdown_hook(self):
    #     rospy.loginfo('Shutdown requested.')
    #     if self.storage_path != '':
    #         rospy.loginfo("Saving data before shutting down...")
    #         self.collected_data_df.to_csv(self.storage_path + self.FILENAME + '.csv')
        

        
def start_rollout_server():
    rospy.init_node('elsa_up_the_stairs') 
    roller = UpTheStairsData()
    rate = rospy.Rate(100)
    rospy.Service('up_the_stairs/rollout', Rollout, roller.callback)
    rospy.Service('up_the_stairs/rollout_gaussian', RolloutGaussian, roller.callback_gaussian)
    rospy.loginfo("Up the stairs service ready")
    # rospy.on_shutdown(roller.shutdown_hook)
    rospy.spin()


if __name__ == '__main__':
    start_rollout_server()  