import rospy
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3
from gazebo_msgs.msg import ModelState
from up_the_stairs.srv import BackToStart, BackToStartResponse

class BackToStartNode:
    def __init__(self):
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.set_model_state.wait_for_service()
    
    def callback(self, request):
        robot_name = 'bd_droid'
        success = True
        try:
            model_state_msg = ModelState()
            model_state_msg.model_name= robot_name
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
        return BackToStartResponse(True)
        

if __name__ == '__main__':
    rospy.init_node('elsa_reset') 
    node = BackToStartNode()
    rate = rospy.Rate(100)
    service = rospy.Service('/up_the_stairs/back_to_start', BackToStart, node.callback)
    rospy.loginfo("Up the stairs reset service ready")
    rospy.spin()
    


    