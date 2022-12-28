import numpy as np

def min_dist_cal(x):
    """
    Description: finding min depth 
    Args:
        data (list): depth list
    return : index
    """
    return np.where(np.array(x)-min(x)==0)[0][0] 

def make_target_function(list_,type,sensor):
    """
    Description: make target
    Args:
        data (human_following/track): detected human list
        type : 'axis', 'id'

    return : 'id' =>id; 'axis' => axis(x,y,z)
    """
    
    if type =='id':
        axis_depth_list=[list_[i][2][2] for i in range(len(list_))]
        min_dist=min_dist_cal(axis_depth_list)
        target_id= list_[min_dist][1]
        return target_id 

    elif type=='axis':
        axis_depth_list=[list_[i][2][2] for i in range(len(list_))]
        index=min_dist_cal(axis_depth_list)
        target_id= list_[index][1]
        target_axis= list_[index][2]
        if sensor=='cam':

            return target_id,[target_axis[1],target_axis[0],target_axis[2]],index

        elif sensor =='lidar':
            
            return target_id,[target_axis[1],target_axis[0],target_axis[2]]
            