import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from nav_msgs.msg import Path
from std_msgs.msg import Header
import argparse
import numpy as np
from datetime import datetime

class RecordTrajectoryNode:
    def __init__(self):
        parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')          
        parser.add_argument('-f', '--out_folder', type=str)
        parser.add_argument('-l', '--trajectory_length', type=int)
        args = parser.parse_args()
        now = datetime.now()
        self.FILENAME = 'traj-' + now.strftime("%m_%d_%Y__%H_%M_%S") 
        self.TRAJECTORY_LENGTH = args.trajectory_length
        self.OUTFOLDER = args.out_folder
        self.trajectories = np.empty((0,self.TRAJECTORY_LENGTH,3))

     
    def add_trajectory(self, msg):
        trajectory = np.empty((0,3))
        for i, pose in enumerate(msg.poses):
            if i == self.TRAJECTORY_LENGTH:
                break
            trajectory = np.append(trajectory, np.array([[pose.pose.position.x, 
                                                          pose.pose.position.y,
                                                          pose.pose.position.z]]), axis=0)
        
        self.trajectories = np.append(self.trajectories, np.expand_dims(trajectory, axis=0), axis=0)

    def shutdown_hook(self):
        rospy.loginfo("Saving trajectories before shutting down...")
        np.save(self.OUTFOLDER + '/' + self.FILENAME, self.trajectories)
        rospy.loginfo("Complete. Shutting down...")



if __name__ == '__main__':
    rospy.init_node('elsa_record_trajectories') 
    jp_node = RecordTrajectoryNode()
    rate = rospy.Rate(100)
    rospy.Subscriber('/up_the_stairs/trajectory', Path, callback=jp_node.add_trajectory)
    rospy.loginfo("Recording trajectories ...")
    # Register the shutdown hook
    rospy.on_shutdown(jp_node.shutdown_hook)
    rospy.spin()
    


    