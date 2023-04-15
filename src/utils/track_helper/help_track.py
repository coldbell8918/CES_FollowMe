import numpy as np

def min_dist_cal(x):
    """
    Description: finding min depth 
    Args:
        data (list): depth list
    return : index
    """
    return np.where(np.array(x)-min(x)==0)[0][0] 

def make_target_function(list_,type,sensor,const):
    
    """
    Description: make targety
    Args:
        data (human_following/track): detected human list
        type : 'axis', 'id'

    return : 'id' =>id; 'axis' => axis(x,y,z)
    """

    axis_depth_list=[list_[i][2][2] for i in range(len(list_))]
    min_dist=min_dist_cal(axis_depth_list)
    
    if min_dist<const:
        target_id= list_[min_dist][1]
        if type =='id':
            return target_id 

        elif type=='axis':
            target_axis= list_[min_dist][2]
            if sensor=='cam':

                return target_id,[target_axis[1],target_axis[0],target_axis[2]],min_dist

            elif sensor =='lidar':
                
                return target_id,[target_axis[1],target_axis[0],target_axis[2]]
    else:
        if type =='id':
            return None

        elif type=='axis':
            target_axis= list_[min_dist][2]
            if sensor=='cam':

                return None,None,None

            elif sensor =='lidar':
                
                return  None,None
        

    
            