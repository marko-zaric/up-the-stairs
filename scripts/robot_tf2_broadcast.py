#!/usr/bin/env python  
import rospy

# Because of transformations
import tf_conversions

import tf2_ros
import geometry_msgs.msg
from gazebo_msgs.srv import GetModelState
from visualization_msgs.msg import Marker
from std_msgs.msg import Header


def handle_pose(pose, robot_name, publisher_model):
    '''
    The handler function for the turtle pose message broadcasts this turtle's translation and rotation, 
    and publishes it as a transform from frame "world" to frame "turtleX". 
    '''
    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()
    '''
    Now, we create a Transform object and give it the appropriate metadata.

        1. We need to give the transform being published a timestamp, we'll just stamp it with the current time, ros::Time::now().

        2. Then, we need to set the name of the parent frame of the link we're creating, in this case "world"
        3. Finally, we need to set the name of the child node of the link we're creating, in this case this is the name of the turtle itself. 
    '''
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "world"
    t.child_frame_id = robot_name
    # Here we copy the information from the 3D turtle pose into the 3D transform. 
    t.transform.translation.x = pose.position.x
    t.transform.translation.y = pose.position.y
    t.transform.translation.z = pose.position.z
    # q = tf_conversions.transformations.quaternion_from_euler(0, 0, msg.theta)
    t.transform.rotation.x = pose.orientation.x
    t.transform.rotation.y = pose.orientation.y
    t.transform.rotation.z = pose.orientation.z
    t.transform.rotation.w = pose.orientation.w

    # This is where the real work is done. Sending a transform with a TransformBroadcaster requires passing in just the transform itself. 
    br.sendTransform(t)

    # Send marker
    marker = Marker()
    h = Header()
    h.frame_id = "world"
    h.stamp = rospy.Time.now()
    marker.header = h
    marker.id = 0
    marker.type = Marker.CUBE #Marker.MESH_RESOURCE
    marker.action = 0
    marker.pose.position.x = pose.position.x
    marker.pose.position.y = pose.position.y
    marker.pose.position.z = pose.position.z
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.71
    marker.pose.orientation.w = 0.71
    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.a = 1.0
    marker.color.r = 66/255
    marker.color.g = 245/255
    marker.color.b = 212/255
    #marker.mesh_resource = "package://up_the_stairs/models/bd_droid/untitled.dae"
    publisher_model.publish(marker)    


if __name__ == '__main__':
    rospy.init_node('tf2_up_the_stairs')
    robot_name = rospy.get_param('~robot_name')
    publisher_model = rospy.Publisher('robot_model', Marker, queue_size=10)
    get_model_state = rospy.ServiceProxy( '/gazebo/get_model_state', GetModelState)
    rospy.wait_for_service('/gazebo/get_model_state')
    while not rospy.is_shutdown():
        try:
            model_state = get_model_state(robot_name, 'world')
            handle_pose(model_state.pose, robot_name, publisher_model)
        except Exception as e:
            rospy.logwarn(e)
        rospy.sleep(1)
  