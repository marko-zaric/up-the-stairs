import rospy
from gazebo_msgs.srv import GetWorldProperties, ApplyBodyWrench, GetLinkProperties, GetModelState
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Wrench, Point, PoseStamped, WrenchStamped
from std_msgs.msg import Header
from nav_msgs.msg import Path
from up_the_stairs.srv import Jump, JumpResponse
import numpy as np
from std_msgs.msg import Header
import tf2_ros
from scipy.spatial.transform import Rotation as R

class JumperNode:
    def __init__(self):
        self.record_trj = False
        self.trajectory = []
        self.trajectory_min1 = []
        self.trajectory_min2 = []

        self.gazebo_world = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        self.gazebo_world.wait_for_service()

        self.apply_body_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
        self.apply_body_wrench.wait_for_service()

        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.get_model_state.wait_for_service()
        
        self.get_robot_properties = rospy.ServiceProxy('/gazebo/get_link_properties', GetLinkProperties)
        self.get_robot_properties.wait_for_service()

        self.traj_pub = rospy.Publisher('/up_the_stairs/trajectory', Path)
        self.traj_pub_minus1 = rospy.Publisher('/up_the_stairs/trajectory_min_1', Path)
        self.traj_pub_minus2 = rospy.Publisher('/up_the_stairs/trajectory_min_2', Path)

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.force_arrows = rospy.Publisher('/up_the_stairs/force_arrow', WrenchStamped, queue_size=10)
    

    def execute_jump(self, request):
        robot_name = 'bd_droid'
        response_model_state = self.get_model_state(model_name = robot_name)
        robot_pose = response_model_state.pose.position
        self.record_trj = True
        self.compute_force_vector(robot_name, robot_pose, request.magnitude, request.angle)   
        diff_state = 1000
        previous_state = self.get_model_state(model_name = robot_name).pose.position
        while diff_state > 10**-3:
            rospy.sleep(1)
            new_state = self.get_model_state(model_name = robot_name).pose.position
            diff_state = np.abs(previous_state.x - new_state.x) + np.abs(previous_state.y - new_state.y) +np.abs(previous_state.z - new_state.z)
            previous_state = new_state

        self.record_trj = False
        h = Header()
        h.stamp = rospy.Time.now()
        h.frame_id = 'world'
        path_msg = Path(header=h, poses=self.trajectory)
        self.traj_pub.publish(path_msg)
        if len(self.trajectory_min1) > 0:
            path_msg = Path(header=h, poses=self.trajectory_min1)
            self.traj_pub_minus1.publish(path_msg)
        if len(self.trajectory_min2) > 0:
            path_msg = Path(header=h, poses=self.trajectory_min2)
            self.traj_pub_minus2.publish(path_msg)
        self.trajectory_min2 = self.trajectory_min1
        self.trajectory_min1 = self.trajectory
        self.trajectory = []
        return JumpResponse(True)
    
    def execute_jump2(self, force, angle):
        robot_name = 'bd_droid'
        response_model_state = self.get_model_state(model_name = robot_name)
        robot_pose = response_model_state.pose.position
        self.compute_force_vector(robot_name, robot_pose, force, angle)   
        return JumpResponse(True)



    def compute_force_vector(self, robot_name, robot_pose, force_val, angle):
        
        link_name = robot_name + '::' +robot_name

        robot_properties = self.get_robot_properties(link_name)

        q = np.ones(3)

        q[0] = 0
        q[1] = np.cos(angle*(np.pi/180))
        q[2] = np.sin(angle*(np.pi/180))


        f = q * robot_properties.mass
        f = f/np.linalg.norm(f)
        

        f *= force_val

        # projecting on components
        wrench_msg = Wrench()


        wrench_msg.force.x = f[0]
        wrench_msg.force.y = f[1]
        wrench_msg.force.z = f[2]

        wrench_msg.torque.x = 0
        wrench_msg.torque.y = 0
        wrench_msg.torque.z = 0

        wrench_stamped_msg = WrenchStamped()
        wrench_stamped_msg.wrench = wrench_msg
        wrench_stamped_msg.header.frame_id = robot_name

        dummy_point = Point()
        dummy_point.x = 0.0
        dummy_point.y = 0.0
        dummy_point.z = 0.0

        self.force_arrows.publish(wrench_stamped_msg)
        self.apply_body_wrench(body_name= link_name,
                               reference_frame= 'world',
                               reference_point=dummy_point,
                               wrench=wrench_msg,
                               duration=rospy.Duration(0.01)
                               )
        
    def get_trajectory(self, msg):
        if self.record_trj == True:
            h = Header()
            h.stamp = rospy.Time.now()
            h.frame_id = 'world'
            robot_idx = msg.name.index('bd_droid')
            pose_stamped = PoseStamped(header=h, pose=msg.pose[robot_idx])
            self.trajectory.append(pose_stamped)


if __name__ == '__main__':
    rospy.init_node('elsa_jumper') 
    jp_node = JumperNode()
    rate = rospy.Rate(100)
    service = rospy.Service('/up_the_stairs/jumper', Jump, jp_node.execute_jump)
    rospy.Subscriber('/gazebo/model_states',data_class=ModelStates, callback=jp_node.get_trajectory)
    rospy.loginfo("Up the stairs jumper service ready")
    rospy.spin()
    


    