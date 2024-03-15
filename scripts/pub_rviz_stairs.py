#!/usr/bin/env python
# license removed for brevity
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header

def talker():
    pub = rospy.Publisher('stairs', MarkerArray, queue_size=10)
    rospy.init_node('stairs_model', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        marker_list = []
        for i in range(4):
            marker = Marker()
            h = Header()
            h.frame_id = "world"
            h.stamp = rospy.Time.now()
            marker.header = h
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = 0
            marker.pose.position.x = 0 #-2.5
            marker.pose.position.y = 0.8 + 0.4*i
            marker.pose.position.z = 0.15 + 0.15*i
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 2.5 #5
            marker.scale.y = 0.4
            marker.scale.z = 0.3*(i+1)
            marker.color.a = 1.0
            marker.color.r = 0.8
            marker.color.g = 0.8
            marker.color.b = 0.8
            marker_list.append(marker)
        pub.publish(MarkerArray(marker_list))
        rate.sleep()
        
if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass