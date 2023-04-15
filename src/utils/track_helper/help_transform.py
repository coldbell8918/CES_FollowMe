import numpy as np

def data_transform(data,cam_lidar,type):
    """
    Description: transform data to 2D array
    Args:
        data (human_following/track): detected human list , (leg_tracker/PersonArray): detected human array
        cam_lidar (string) : 'cam','lidar'
        type (string) : 'all','axis'
    return : all([[0,id,(x,y,depth)],..],crop_imgs(at cam)) , axis([[x,y,depth],...])  
    """
    merge=list()
    if type=='all':
        if cam_lidar=='cam':
            crowd=data.persons
            crop_imgs=list()
            for i in range(len(crowd)):
                merge.append(0)
                merge.append(crowd[i].id)
                y=round(crowd[i].pose.x,3) ## taken y
                depth=round(crowd[i].pose.y,3) ## taken depth
                x=round(pow(depth**2-y**2,0.5),3)
                merge.append((x,y,depth))
                
                height=crowd[i].shape[0]
                weight=crowd[i].shape[1]
                dim=crowd[i].shape[2]
                image =crowd[i].crops
                image=[i for i in image]
                image=np.array(image).astype(np.uint8)  
                image=image.reshape(height,weight,dim)
                crop_imgs.append(image)

            merge=np.array(merge,dtype=object)
            return np.reshape(merge,(int(len(merge)/3),3)), crop_imgs  ### merge=[[0,1,(x,y,depth)]] 


        elif cam_lidar=='lidar':
            crowd=data.people
            for i in range(len(crowd)):
                merge.append(0)
                merge.append(crowd[i].id)
                x=round(crowd[i].pose.position.x,3)
                y=round(crowd[i].pose.position.y,3)
                depth=round(pow(pow(y,2)+pow(x,2),0.5),3)
                merge.append((x,y,depth))
            merge=np.array(merge,dtype=object)
            return np.reshape(merge,(int(len(merge)/3),3))  ### merge [[0,1,(x,y,z)]]

    elif type=='axis':
        if cam_lidar=='cam':
            crowd=data.persons
            if len(crowd)!=0:
                for i in range(len(data)):
                    x=round(crowd[i].pose.x,3)
                    y=round(crowd[i].pose.y,3)
                    depth=round(pow(x**2+y**2,0.5),3)
                    merge.append((x,y,depth))
            else:
                return [[None,None,None]]
                
        elif cam_lidar=='lidar':
            crowd=data.people
            if len(crowd)!=0:
                for i in range(len(data)):
                    x=round(crowd[i].pose.position.x,3)
                    y=round(crowd[i].pose.position.y,3)
                    depth=round(pow(pow(y,2)+pow(x,2),0.5),3)
                    merge.append((x,y,depth))
            else:
                return [[None,None,None]]

        merge=np.array(merge)
        return np.reshape(merge,(int(len(merge)/3),3)) 

