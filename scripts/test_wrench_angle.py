import rospy
from up_the_stairs.srv import Jump, JumpRequest, BackToStart

class AngleTester:
    def __init__(self):
        self.set_start = rospy.ServiceProxy( '/up_the_stairs/back_to_start', BackToStart)
        self.set_start.wait_for_service()

        self.jumper = rospy.ServiceProxy('/up_the_stairs/jumper', Jump)
        self.jumper.wait_for_service()
        

        force = 10000
        for angle in range(180):
            rospy.logwarn("Force: " + str(force) + " Angle: " + str(angle))
            self.set_start()
            rospy.sleep(1)
            jump_ex = JumpRequest()
            jump_ex.magnitude=force
            jump_ex.angle=angle
            self.jumper(jump_ex)
    

if __name__ == '__main__':
    rospy.init_node('elsa_angle_tester') 
    AngleTester()
    rate = rospy.Rate(100)
    rospy.loginfo("Going through angles ...")
    rospy.spin()