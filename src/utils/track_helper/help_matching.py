import numpy as np
from .orb_provider import orb_match

def matching_data(list_,target_id,target,target_img,imgs,length,type,const,sensor,match_ratio,match_type):
    """
    Description: Find target data from list_ -> len : 0 means there is no matching data, 1 means find data ; index : target data's index from list_
    Args:
        list_ (data_transform): all([[0,id,(x,y,depth)],..])
        target : id or axis
        length : 'len','index'
        type : 'axis', 'id'
        const : constant(float or int) ;default 0.8
    return :  len, index
    """

    id_list=[list_[i][1] for i in range(len(list_))]

    if type=='id':
        id_list=[list_[i][1] for i in range(len(list_))]
        check=np.where(np.array(id_list)==target_id)[0]
        if length=='len':
            return len(check)
        elif length=='index':
            return list_[check[0]][1],list_[check[0]][2],check[0]

    elif type=='axis':

        axis_lists=np.array([list_[i][2] for i in range(len(list_))])
        target=np.array([target])
        

        if sensor =='cam':
            match_count,distances=orb_match(target_img,imgs,match_ratio,match_type)
    
            norm=np.linalg.norm(axis_lists-target,axis=1)
            calculate=np.min(np.array(distances)/(np.array(match_count)))
            check=np.where(calculate)[0]
            
            if check==[]:
                if length=='len':
                    return 0
                elif length=='index':
                    return None
            else:    
                if length=='len':
                    return len(check)
                elif length=='id_axis':
                    return list_[check[0]][1],list_[check[0]][2],check[0]
        
        elif sensor=='lidar':
            norm=np.min(np.linalg.norm(axis_lists-target,axis=0))
            check=np.where(norm<const)[0]
            if check==[]:
                if length=='len':
                    return 0
                elif length=='index':
                    return None
            else:    
                if length=='len':
                    return len(check)
                elif length=='id_axis':
                    return list_[check[0]][1],list_[check[0]][2],check[0]
        

    
def cam_aov_check(target_id,target_axis,cam_aov):
    """
    Description: camera angle of view (input data is only lidar type)
    Args:
        target_id : target_id
        target_axis : [x,y,depth] 
        cam_aov : default 0.5
    return :  target_id,target_axis
    """
    theta=np.arcsin(target_axis[0]/(target_axis[2]+0.0001)) ## x/depth #to prevent zero division +0.00001
    if theta.all()<(90-cam_aov/2) and theta.all()>(cam_aov/2+90):
        return None,None
    else:
        return target_id,target_axis