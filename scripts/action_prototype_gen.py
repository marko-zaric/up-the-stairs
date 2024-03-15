import rospy
import rospkg
import numpy as np
import pandas as pd
from datetime import datetime
from up_the_stairs.srv import ActionPrototypeGen, ActionPrototypeGenRequest
import os
import yaml
from actprogen import effect_based_prototypes

class MotionPrototypeGeneratorNode:
    def __init__(self):
        self.data = None
        self.FILENAME = ''
        self.action_prototypes = None
        rospack = rospkg.RosPack()
        self.path_to_pkg = rospack.get_path('up_the_stairs')
        


    def callback(self, request):
        self.name_prototype_file = request.name_file
        self.env_name = request.env_name
        # load config file
        with open(self.path_to_pkg + '/data/prototypes/'+ request.env_name + '/config.yaml', 'r') as file:
                CONFIG = yaml.safe_load(file)

        # load data
        self.data = pd.read_csv(os.path.join(self.path_to_pkg, 'data/motion_samples', request.env_name, CONFIG['data']))
        self.data = self.data[self.data[CONFIG['exclude-condition']] == True]
        self.data = self.data.sample(n=CONFIG['number-samples'],random_state=42)
        
        # generate motion prototypes
        self.prototype_generator = effect_based_prototypes.EffectActionPrototypes(self.data, CONFIG['motion-dimensions'])
        self.prototype_generator.generate(effect_dimensions=CONFIG['cluster-by'])
        rospy.loginfo("Final prototypes: ")
        rospy.loginfo(self.prototype_generator.action_prototypes)
        np.save(os.path.join(self.path_to_pkg,'data/prototypes',self.env_name, self.name_prototype_file), self.prototype_generator.action_prototypes)
        for key in self.prototype_generator.prototypes_per_label:
             np.save(os.path.join(self.path_to_pkg,'data/prototypes',self.env_name, str(key)), self.prototype_generator.prototypes_per_label[key])
        rospy.loginfo("Prototypes stored as: " + request.name_file + ".npy for environment " + request.env_name) 
        return 1

        
def start_prototype_server():
    rospy.init_node('elsa_action_prototypes') 
    MP_GEN = MotionPrototypeGeneratorNode()
    rospy.Rate(100)
    rospy.Service('up_the_stairs/action_prototype_generator', ActionPrototypeGen, MP_GEN.callback)
    rospy.loginfo("Action prototype generator ready!")
    rospy.spin()


if __name__ == '__main__':
    start_prototype_server()  