import numpy as np
import cv2
from matplotlib import pyplot as plt # this lets you draw inline pictures in the notebooks

def orb_match(target_img,image_list,ratio,match_type):
    NoneType=type(None)
    """
        Description: orb launching
        Args:
            target_img : array_img data
            image_list : array_imgs data 
            ratio : default 0.5
            match_type : match (euclidean distance), knnMatch (clustering)
        return : counted match numbers, matrix distances 
    """
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING) # explain of parameter :see this link https://leechamin.tistory.com/330
    _,target_descriptors= orb.detectAndCompute(target_img,None)
    
    match_count=[]
    distances=[]
    image_descriptors=None
    for i in range(len(image_list)):
        image=image_list[i]
        _, image_descriptors = orb.detectAndCompute(image,None)
        if type(target_descriptors)!=NoneType and type(image_descriptors)!=NoneType:
            if match_type=='match':
                matches = matcher.match(target_descriptors,image_descriptors)
            elif match_type=='knnMatch':
                matches = matcher.knnMatch(target_descriptors, image_descriptors, 2)
            try:
                good_matches = [first for first,second in matches \
                                    if first.distance < second.distance *1]
                match_count.append(len(good_matches)+1) ## prevent zero division
                dis=0
                for i in good_matches:
                    dis+=i.distance
                distances.append(dis) 
            except:
                match_count.append(1)
                distances.append(100000)
        else:
            match_count.append(1)
            distances.append(100000)

    return match_count,distances

### reference 
### https://bkshin.tistory.com/entry/OpenCV-29-%EC%98%AC%EB%B0%94%EB%A5%B8-%EB%A7%A4%EC%B9%AD%EC%A0%90-%EC%B0%BE%EA%B8%B0
### https://towardsdatascience.com/improving-your-image-matching-results-by-14-with-one-line-of-code-b72ae9ca2b73