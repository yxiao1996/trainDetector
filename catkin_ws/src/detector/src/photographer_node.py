#!/usr/bin/env python
import rospy
from std_msgs.msg import String #Imports msg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class photographer_node(object):
    def __init__(self):
        # Save the name of the node
        self.node_name = rospy.get_name()
        
        rospy.loginfo("[%s] Initialzing." %(self.node_name))
        
        self.active = True
        self.state = "init"
        self.bridge = CvBridge()

        self.save_dir = rospy.get_param('~save_dir')
        self.img_count = 0

        # Setup publishers
        #self.pub_topic_a = rospy.Publisher("~topic_a",String, queue_size=1)
        # Setup subscriber
        self.sub_image = rospy.Subscriber("~image_raw", Image, self.cbImage, queue_size=1)
        #self.sub_switch = rospy.Subscriber("~switch", BoolStamped, self.cbSwitch)
        # Read parameters
        self.pub_timestep = self.setupParameter("~pub_timestep",0.05)
        # Create a timer that calls the processImg function every 1.0 second
        self.timer = rospy.Timer(rospy.Duration.from_sec(self.pub_timestep),self.processImg)

        rospy.loginfo("[%s] Initialzed." %(self.node_name))

    def setupParameter(self,param_name,default_value):
        value = rospy.get_param(param_name,default_value)
        rospy.set_param(param_name,value) #Write to parameter server for transparancy
        rospy.loginfo("[%s] %s = %s " %(self.node_name,param_name,value))
        return value

    def cbImage(self,msg):
        if not self.active:
            return
        self.rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        #self.rgb[:, :, [0, 2]] = self.rgb[:, :, [2, 0]]
        #rgb[:, :, [0, 2]] = rgb[:, :, [2, 0]]
        self.state = "default"


    def cbSwitch(self, msg):
        next_state = msg.data
        self.active = next_state

    def processImg(self,event):
        if not self.active:
            return
        if self.state == "init":
            return
        gray = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2GRAY)
            
        cv2.imshow('photographer', self.rgb)
        key = cv2.waitKey(10)
        if key == ord('s'):
            cv2.imwrite(self.save_dir+'/'+str(self.img_count)+'.jpg', self.rgb)
            self.img_count += 1

    def on_shutdown(self):
        rospy.loginfo("[%s] Shutting down." %(self.node_name))

if __name__ == '__main__':
    # Initialize the node with rospy
    rospy.init_node('photographer_node', anonymous=False)

    # Create the NodeName object
    node = photographer_node()

    # Setup proper shutdown behavior 
    rospy.on_shutdown(node.on_shutdown)
    
    # Keep it spinning to keep the node alive
    rospy.spin()