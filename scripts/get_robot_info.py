import rospy
import pandas as pd
from gazebo_msgs.msg import ModelStates


def callback(message, trajectory):
    trajectory.loc[len(trajectory)] = [message.pose[2].position.x, 
                                      message.pose[2].position.y,
                                      message.pose[2].position.z]
    print(trajectory)
    


if __name__ == '__main__':
    rospy.init_node('get_gazebo_rob_info') 
    rate = rospy.Rate(100)
    trajectory = pd.DataFrame(columns=['x', 'y', 'z'])
    rospy.Subscriber('/gazebo/model_states',data_class=ModelStates, callback=callback)
    rospy.spin()