#!/usr/bin/env python3

import rospy 
from human_following.msg import track
import numpy as np
from leg_tracker.msg import Person, PersonArray, Leg, LegArray
import time


class human_tracker():
    def __init__(self):
        self.pubs=list()
        self.subs1=[rospy.Subscriber('tracker', track, self.track_callback)]
        self.subs2=[rospy.Subscriber('people_tracked', PersonArray, self.person_callback)]
        self.camera_id_none,self.lidar_id,self.stop =True,True,True
        self.camera_ok,self.lidar_ok=False,False
        self.time_count=0

        self.track_id_save=None
        # self.pubs['cmd_vel'] = rospy.Publisher("cmd_vel", Twist,queue_size=1)

    def track_callback(self, data):
        

        axis,number,clss=data.axis,data.number,data.clss
        merge=list()
        for i in range(len(clss)):
            merge.append(clss[i])
            merge.append(number[i])
            y=round(pow(pow(axis[2*i+1]+0.1,2)-pow(axis[2*i],2),0.5),3)
            merge.append((y,round(-axis[2*i],3),round(axis[2*i+1]+0.1,3)))
        ### merge [[0,1,(x,y,z)]]
        merge=np.array(merge,dtype=object)
        merge=np.reshape(merge,(int(len(merge)/3),3))

        if self.camera_id_none: 
            # self.camera_ok=False 
            self.camera_track(merge)
            
     
        else: 
            # self.camera_ok=True
            self.camera_cmd(merge)
        # self.lidarcamera()

    def camera_track(self, list_):
        print("find id")
        
        if len(list_)!=0:
            axis_z_list=[list_[i][2][2] for i in range(len(list_))]
            min_dist=np.where(np.array(axis_z_list)-min(axis_z_list)==0)[0][0]
            self.target_c=list_[min_dist][1]
            self.camera_id_none=False
            
            # if self.time_count:
            self.track_id_save=[self.target_c]
            self.time_count=0
            print("robot's track_id is {}".format(self.track_id_save[0]))
        
        else: 
            self.target_c=None
        self.time_count+=1
        print(self.time_count)

    def camera_cmd(self, list_):
        # print("track id {}".format(self.track_id_save[0]))
        self.camera_mode=True
        id_list=[list_[i][1] for i in range(len(list_))]

        if len(np.where(np.array(id_list)==self.track_id_save)[0])!=0: 
            self.camera_mode=False
        
        elif self.time_count==10: 
            self.camera_id_none=True
            
        if self.camera_mode: 
            self.camera_id_none=True
        


    def person_callback(self, data):
        if self.lidar_id: self.lidar_ok=False,self.lidar_track(data.people)            
        else: self.lidar_ok=True, self.lidar_cmd(data.people)
      
    def lidar_track(self, list_):
        # self.lidarcamera()

        id_list=[list_[i].id for i in range(len(list_))]
        if len(id_list)!=0:
            dis=[pow(list_[i].pose.position.x**2+list_[i].pose.position.y**2,0.5) for i in range (0,len(list_))]
            min_dist=np.where(np.array(dis)-min(dis)==0)[0][0]
            self.track_l=list_[min_dist].id
            self.lidar_id=False

        else: 
            self.track_l=None

    def lidar_cmd(self, list_):
        self.lidar_mode=True
        id_list=[ list_[i].id for i in range(0,len(list_))]
        if len(np.where(np.array(id_list)==self.track_l)[0])==0: self.lidar_mode=False
        if self.lidar_mode: self.lidar_id=True

    def lidarcamera(self):
        if self.camera_ok and self.lidar_ok: print("camera+lidar")
        elif self.camera_ok and not self.lidar_ok: print("only camera")  
        elif not self.camera_ok and self.lidar_ok: print("only lidar") 
        else: print("None")
    
if __name__ == '__main__':
    rospy.init_node('sub_yolo', anonymous=True)
    cls_=human_tracker()
    rospy.spin()