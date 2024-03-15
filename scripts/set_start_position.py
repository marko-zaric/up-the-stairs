import rospy
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3
from gazebo_msgs.msg import ModelState
from up_the_stairs.srv import SetStart, SetStartResponse

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
            model_state_msg.pose = request.start_pose
            vector_msg = Vector3(x=0.0,y=0.0,z=0.0)
            model_state_msg.twist=Twist(linear=vector_msg, angular=vector_msg)
            self.set_model_state(model_state_msg)
        except rospy.ServiceException as e:
                rospy.logerr('set_model_state service call failed: {0}'.format(e))
                success = False
        rospy.loginfo('Start Position set!') 
        return SetStartResponse(success)
        

if __name__ == '__main__':
    rospy.init_node('elsa_up_the_stairs_set_start') 
    node = BackToStartNode()
    rate = rospy.Rate(100)
    service = rospy.Service('/up_the_stairs/set_start', SetStart, node.callback)
    rospy.loginfo("Up the stairs reset service ready")
    rospy.spin()
    


    