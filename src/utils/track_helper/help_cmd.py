def cmd_decision(linear,angular,x,y,depth): 
    """
    Description: cmd decision 
    Args:
        linear : linear_velocity 
        angular : angular_velocity
        x : x_distance from the target in x_tf
        y : y_distance from the target in y_tf
        depth : depth from the target
    """
    if depth>0.5: ## meter 
        if y>0:
            if y<0.3: # robot and human front
                if x<0.6 and x>0.4:
                    return 0,0

                elif x<=0.4 and x==0:
                    return -linear,0

                else:            
                    return linear, 0
            else:
                # if depth<1:
                #     return 0,0
                # else:
                return 0,angular

        else:
            if y>-0.3: # robot and human front
                if x<0.6 and x>0.4:
                    return 0,0          

                elif x<=0.4 and x==0:                        
                    return -linear,0

                else:
                    return linear,0
            else:
                # if depth<1:
                #     return 0,0
                # else:
                return 0, angular
    else:
        if depth <0.4:
            return 0,0
            
        else:
            return -linear,0