import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage import img_as_float, img_as_uint, img_as_int, img_as_ubyte
from skimage.transform import resize
import imutils
from fastai.vision import *
from skimage import measure
from skimage import filters
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import sys
from fastai.vision import Image
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
from matplotlib import pyplot as plt
import numpy as np
import PIL
from IPython.display import clear_output
import time
import bisect as bi
import IPython.display as disp
import fastai
import tkinter as tki

from helper_functions import *

###FUNCTIONS###

def put_classifier_text(_frame, main_class, second_class):
    frame = _frame.copy()
    main_size=cv2.getTextSize("{}: {}".format(main_class[0],main_class[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    second_size=cv2.getTextSize("{}: {}".format(second_class[0], second_class[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)
    
    cv2.putText(frame,"{}: {}".format(main_class[0],main_class[1]),(frame.shape[1]-main_size[0][0]-5,25),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    cv2.putText(frame,"{}: {}".format(second_class[0], second_class[1]),(frame.shape[1]-second_size[0][0]-5,45),cv2.FONT_HERSHEY_SIMPLEX,0.4,(170,170,170),2)
    return frame

def inference(frame):
    '''
    All the steps, one by one:
    
    frame_rgb = frame[...,::-1] #convert bgr to rgb

    t = Image(pil2tensor(PIL.Image.fromarray(frame_rgb).convert("RGB"), np.float32).div_(255))
    prediction = learn.predict(t)
    p = prediction[1].squeeze() #prediction data

    mask = np.array(p).astype(np.uint8)
    '''
    return np.array(
        learn.predict(Image(pil2tensor(PIL.Image.fromarray(frame[...,::-1]).convert("RGB"), np.float32).div_(255)))[1].squeeze()
        ).astype(np.uint8)

def inference_classification(frame):
    '''
    All the steps, one by one:
    
    frame_rgb = frame[...,::-1] #convert bgr to rgb

    t = Image(pil2tensor(PIL.Image.fromarray(frame_rgb).convert("RGB"), np.float32).div_(255))
    prediction = learn.predict(t)
    p = prediction[1].squeeze() #prediction data

    mask = np.array(p).astype(np.uint8)
    '''
    pred_class,pred_idx,outputs = classifier.predict(Image(pil2tensor(PIL.Image.fromarray(frame[...,::-1]).convert("RGB"), np.float32).div_(255)))#[1].squeeze()
    
    main = "{0:.2f}".format(float("{0:.2f}".format(max(outputs)*100.0)))
    second = "{0:.2f}".format(float(np.sort(outputs)[-2]*100.0))
    outputs = np.array(outputs).tolist()
#     outputs = np.sort(outputs).tolist()

#     cv2.putText(frame,"{}: {}".format(,main),(5,25),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
#     cv2.putText(frame,"{}: {}".format(classifier_class_dict[outputs.index(outputs[-2])],second),(5,55),cv2.FONT_HERSHEY_COMPLEX,0.9,(170,170,170),2)
    
    return (classifier_class_dict[outputs.index(np.max(outputs))], main), (classifier_class_dict[outputs.index(np.sort(outputs)[-2])], second)

def put_classifier_text(frame, main_class, second_class):
    main_size=cv2.getTextSize("{}: {}".format(main_class[0],main_class[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    second_size=cv2.getTextSize("{}: {}".format(second_class[0], second_class[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)
    
    cv2.putText(frame,"{}: {}".format(main_class[0],main_class[1]),(frame.shape[1]-main_size[0][0]-5,25),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    cv2.putText(frame,"{}: {}".format(second_class[0], second_class[1]),(frame.shape[1]-second_size[0][0]-5,45),cv2.FONT_HERSHEY_SIMPLEX,0.4,(170,170,170),2)
    return frame
    
    
def unwrap_image(mask):
    
    ####### Find centroid of bifurcation#####
    mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    labels = get_blobs(mask,imshow=False)
    thresh = labels[3]
    thresh = img_as_uint(thresh*100)

    if len(np.where( thresh > 0 )[0]):
        # calculate moments of binary image
        M = cv2.moments(thresh)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX = 125
        cY = 150
        
    ###### endof find centre ############
    
    value = np.sqrt(((mask.shape[0]/2.0)**2.0)+((mask.shape[1]/2.0)**2.0))
    polar_image = cv2.linearPolar(mask.astype(np.float32),(cX,cY), value, cv2.WARP_FILL_OUTLIERS).astype(np.uint8)[:,:,0]
    
    return polar_image

def find_posterior_region(mask, prev_posterior_angles=None, posterior_region_ctr=0):
    if len(mask.shape)>2:
        raise Exception('Mask should be grayscale image.')
    
    posterior_region_ctr+=1
    
    # Find anterior and posterior parts of trachea
    histogram = np.sum((mask==1).astype(np.uint8), axis=1)#row-wise sums
    region, posterior_angle = longest_contiguous_region(histogram < 5, histogram)
    
    if prev_posterior_angles is not None:
        #Average previous angles
        prev_posterior_angles[posterior_region_ctr%len(prev_posterior_angles)]=posterior_angle
        posterior = int(np.average(prev_posterior_angles))
        
        return posterior, prev_posterior_angles, posterior_region_ctr

    return posterior_angle, None, None

def get_posterior_corrected_frame(mask, posterior):
    temp = mask[0:posterior,0:]
    return np.concatenate((mask[posterior:,0:],temp))



def correlate_blobs(new_blobs, prev_blobs, percent_overlap_thresh=0.75, child_area_thresh=1.5, orphan_min_area=50):
    """
    Link parent blobs from prev_blobs to children blobs from new_blobs
    constraints:

    0) children can only have one parent. parents can have multiple children

    1) children must have a large enough intersection with their parent

    2) children must be smaller than their parents (they are allowed to be larger up to a certain threshold)
    if a potential child has a strong correlation to a given potential parent (large intersection), it is only
    considered a child if child_area < parent_area*thresh (where thresh is >= 1)
    else if a potential child has a strong correlation to a given potential parent, and it doesn't meet the area criteria,
    a new track is created for it and it is parentless (TODO: consider going "back" in the tree and grouping previous blobs
    and considering them as a single blob so that the area criteria is met. rn thinking not needed)

    Args:
        new_blobs: list in which each channel is a grayscale image
                   that contains a single blob present in current 
                   frame (see get_blobs_single_class() )
        prev_blobs: list in which each channel is a grayscale image
                   that contains a single blob present in previous
                   frame (see get_blobs_single_class() )
        percent_overlap_thresh (default: 0.75): minimum amount that
                   child must overlap with parent to be considered
                   a child
        child_area_thresh (default: 1.5): child can be up to
                   child_area_thresh times bigger than its parent
                   to be considered a child
        orphan_min_area (default: 50): minimum area for a parentless
                   blob to be added to orphan list. All orphans with
                   less than orphan_min_area will be disregarded
                   
    Returns:
        family, orphans
        
        family: list of tuples of (child_blob, parent_blob) i.e. the
                actual frames are tuple[0] and tuple[1]
                If a parent has no children, entry will be (zero_arr, parent_blob)
                len of family is num_parent_blob
                
        orphans: list of tuples of (orphan_blob, None) i.e. the actual
                frame is tuple[0], and tuple[1] is None
                len of orphans is the number of orphans in current frame
                   
    """
    parent_to_children = {} #key is the parent ID, value is the child ID
    orphans = [] # IDs of blobs without parents
    
    #Make all children empty lists
    for p in range(0, len(prev_blobs)):
        parent_to_children[p] = [] 
    
    #Populate parent_to_child dict (children pick their parents)
    for c in range(0, len(new_blobs)):
        nblob = new_blobs[c]
   
        max_intercept = 0
        max_intercept_p = -1
                        
        for p in range(0,len(prev_blobs)):
            pblob = prev_blobs[p]
            intersect = cv2.bitwise_and(nblob, pblob)
            
            if blob_area(intersect) > max_intercept:
                max_intercept = blob_area(intersect)
                max_intercept_p = p
          
        if max_intercept > 0 and blob_area(nblob) < blob_area(prev_blobs[max_intercept_p])*child_area_thresh and max_intercept > blob_area(nblob)*0.10:
            parent_to_children[max_intercept_p].append(c) #Huzzah! Child picked a parent
#             print("nparents {} -child {} picked {} ".format(len(prev_blobs), c, max_intercept_p))
        else: #Child has no parent
            orphans.append(c)
                    
    #Join children blobs (if they belong to the same parent) to deal with blob separation
    #and add (child,parent) tuple to a "family" list
#     joined_blobs = {} # len(new_blobs) == len(prev_blobs)+len(orphans)
    family = []
    
    for p in range (0, len(parent_to_children)):
#         joined_blobs[p] = []
        if len(parent_to_children[p])>1: #parent has more than one child
            joined_blob = np.zeros_like(prev_blobs[0])
            for child_blob_index in parent_to_children[p]:
                joined_blob = cv2.bitwise_or(new_blobs[child_blob_index], joined_blob)
#                 print("JOINED BLOB TYPE {}".format(joined_blob.dtype))
#                 print("joined_blobs shape: {}".format(joined_blob.shape))
#             joined_blobs[p].append(joined_blob)
            family.append((joined_blob,prev_blobs[p]))
            
        elif len(parent_to_children[p])==1:
            joined_blob = new_blobs[parent_to_children[p][0]]
#             joined_blobs[p].append(joined_blob)
            family.append((joined_blob,prev_blobs[p]))
        else: #do nothing -> i.e. a parent that didn't have a child will die
            family.append((np.zeros_like(prev_blobs[p]),prev_blobs[p]))
    
    #Add orphans to the future parents list, so they can have children
    orphan_imgs = []
    for o in orphans:
        if np.count_nonzero(new_blobs[o]) > orphan_min_area:
            orphan_imgs.append((new_blobs[o], None))
        
    
    return family, orphan_imgs #joined_blobs has prev_parents+orphans number of channels, each channel being an individual blob





def collapse_channels(blobs):
    """
    Collapses all image channels into a single frame

    Reverses get_blobs_single_class

    Args:
        blobs: multi-channel image
    Returns:
        all blobs in a single frame
    """
    blobs_new_frame = np.zeros_like(blobs[0])

    for b in blobs:
        blobs_new_frame = np.array(blobs_new_frame) | np.array(b)
    return blobs_new_frame

def get_centroid_x(o):
    if blob_area(o)>0:
        M = cv2.moments((o>0).astype(np.uint8))
        cX = int(M["m10"] / M["m00"])
        return cX
    
def get_non_None_section(last_values):
    # No blobs had children. Lost tracking
    if last_values.count(None) == len(last_values):
        raise Exception('Error tracking. Was not able to correlate any blobs. Must reset')
    
    start_none = 0
    end_none = len(last_values)-1
    for j, val in enumerate(last_values):
        if val is not None:
            start_none=j
            break
            
    for j, val in enumerate(reversed(last_values)):
        ctr=len(last_values)-1-j
        if val is not None and ctr >= start_none:
            end_none=ctr
            break
    return start_none, end_none

def plot_blobs_in_order(mid_blobs, blobs, last_values, plot=True):
    if len(mid_blobs)>len(blobs):
        print("Error: Mismatched midblobs len ({}) and blobs ({}).".format(len(mid_blobs), len(blobs)))
        return
    if len(blobs)==0:
        print("Error: No blobs found")
        return
    
    output_frame = np.zeros_like(blobs[0])
    for mb in mid_blobs:
        ID = last_values.index(get_centroid_x(mb))
        for i, b in enumerate(blobs):
            output_frame = (output_frame) | (b/np.max(b)*20).astype(np.uint8)
            if blobs_intersect(mb,b[112:122:].astype(np.uint8)):
                output_frame = (output_frame) | (b/np.max(b)*(ID+1)*25).astype(np.uint8)
                del blobs[i]
                break
    numbers_frame = np.zeros((224,224), np.uint8)
    
    start_none, end_none = get_non_None_section(last_values)
    for cX in last_values[start_none: end_none+1]:
        cv2.putText(numbers_frame , "{}".format(last_values.index(cX)), (cX, 120), cv2.FONT_HERSHEY_SIMPLEX,0.3, 255, 1)
    output_frame = (output_frame) | (numbers_frame).astype(np.uint8)
    output_frame = output_frame.astype(np.uint8)
    print(output_frame.dtype, np.unique(output_frame))
    if plot:
        cv2.imshow("blobs",np.array(output_frame/255,dtype=np.float32))
        cv2.waitKey(30)
    
    return output_frame



def overlay_transparent(new_img, transparent_img, x_offset, y_offset):
    alpha_s = transparent_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        new_img[y_offset:y_offset+transparent_img.shape[0], x_offset:x_offset+transparent_img.shape[1], c] = (alpha_s * transparent_img[:, :, c] + alpha_l * new_img[y_offset:y_offset+transparent_img.shape[0], x_offset:x_offset+transparent_img.shape[1], c])

    return new_img


# last_values=[None,None,None,4,4,4,41,3,2,6,None] #debug only

def draw_trachea_map(new_img, last_values, tracking_status):
    
    # new_img = np.zeros((new_img.shape[0],larynx_icon.shape[1]*2, 3), larynx_icon.dtype)
    # new_img = larynx_icon
    print("New img shape {}, larynx {}".format(new_img.shape, larynx_icon.shape))
    # cv2.imshow("lryx",new_img)
    # if not new_img.shape[1]>larynx_icon.shape[1]:
    #     print("image is not wide enough")
    
    y_offset=0
    x_offset=int(larynx_icon.shape[1]/2)

    #Add larynx
    new_img[y_offset:y_offset+larynx_icon.shape[0], x_offset:x_offset+larynx_icon.shape[1]]=larynx_icon
    y_offset+=larynx_icon.shape[0]-2
    x_offset += abs(round((larynx_icon.shape[1]-ring_on_icon.shape[1])/2)) #compensate for rings being less wide than larynx icon
    
    if not tracking_status:
#         cv2.putText(new_img,"TRACKING FAILED.",(0,y_offset+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        print("Returning new_img!")
        return new_img

    first_visible_ring=None
    for ring_num, val in enumerate(last_values):
        if val is None or not tracking_status:
            if y_offset+ring_off_icon.shape[0]<new_img.shape[0]:
                new_img[y_offset:y_offset+ring_off_icon.shape[0], x_offset:x_offset+ring_off_icon.shape[1]] = ring_off_icon
        else:
            if y_offset+ring_off_icon.shape[0]<new_img.shape[0]:
                new_img[y_offset:y_offset+ring_on_icon.shape[0], x_offset:x_offset+ring_on_icon.shape[1]] = ring_on_icon
                if first_visible_ring is None:
                    first_visible_ring=ring_num+1
    
        cv2.putText(new_img,"{}".format(ring_num+1),(ring_on_icon.shape[1]+10,y_offset+15),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1)
        y_offset+=ring_on_icon.shape[0]

        
    # Ellipsis
    x_offset=round(ring_on_icon.shape[1]/2)+int(larynx_icon.shape[1]/2)+3
    y_offset+=2
    if y_offset+dots_icon.shape[0]<new_img.shape[0]:
        new_img[y_offset:y_offset+dots_icon.shape[0], x_offset:x_offset+dots_icon.shape[1]] = dots_icon
    
    # Camera
    x_offset=round(ring_on_icon.shape[1]/2)+int(larynx_icon.shape[1]/2)+2
    y_offset=(first_visible_ring-1)*ring_on_icon.shape[0]+larynx_icon.shape[0]-15
    new_img = overlay_transparent(new_img, camera_icon, x_offset, y_offset)
        
    return new_img


def overlay_mask(img, mask, _alpha=0.6, show=False):
    alpha = _alpha
    beta = (1-alpha)
    # print("HELLO: {}".format(np.unique()))
    output=alpha*img+beta*np.array(mask/255,dtype=np.float32)
    # output = cv2.addWeighted(img, alpha, mask.astype('float'), beta, -10.0)
    if np.max(output)>1:
        print("!!!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!!!!!!!!")

    if show:
#         rgb_ = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        pltimg(output)
#         plt.show()
    
    return output
###ENDOF FUNCTIONS####




#################TRACKER#####################
class Tracker():
    
    def __init__(self, init_frame=None, _verbose=False):
        self.mid_blob_tracks = []
        self.mid_prev=[]
        self.verbose=_verbose
        mask_continuous=init_frame

        blobs_prev = get_blobs_single_class(mask_continuous==1, label_value=255)
        
        self.mid_blob_tracks = []
        for m in blobs_prev:
            if blob_area(m[112:122:])>0:
                M = cv2.moments((m[112:122:]>0).astype(np.uint8))
                cX = int(M["m10"] / M["m00"])
                self.mid_blob_tracks.append([cX])
        self.mid_blob_tracks.sort(reverse=True)
        mid_temp = get_middle_section(blobs_prev, collapse_channels=False)
        
        self.mid_prev=[]
        for mid in mid_temp:
            if blob_area(mid)>0:
                self.mid_prev.append(mid)
    
    def iterate(self, new_frame, ring_value=1):
        img = new_frame==ring_value
        
        #Find all blobs in current frame
        blobs_current = get_blobs_single_class(img, label_value=255)
        mid_temp = get_middle_section(blobs_current, collapse_channels=False)
        mid_new = []
        for mid in mid_temp:
            if blob_area(mid)>0:
                mid_new.append(mid)
                
        #Correlate children to parent blobs
        # blobs_new has the joined blobs with indices corresponding to their parents
        blobs_new, orphans = correlate_blobs(mid_new, self.mid_prev, orphan_min_area=50)
        if blobs_new is None:
            if DEBUG_MODE:
                print("EMPTY!!")
        
        #Add orphans to blobs_prev, so they can become parents in the next iteration
        #Note: this happens AFTER children have already found their parents
        self.mid_prev = []
        for blob_ in blobs_new:
            if blob_area(blob_[0])>0:
                self.mid_prev.append(blob_[0])
                
        if DEBUG_MODE:        
            #debug check
            for en, bl in enumerate(self.mid_prev):
                if blob_area(bl)<1:
                    print("Empty blob at mid_prev[{}]".format(en))

        orphan_imgs = [np.array(o[0]).astype(np.uint8) for o in orphans]
        for o in orphan_imgs:
            if blob_area(o)>0:
                 self.mid_prev.append(o)
        
        for ctr,m in enumerate(blobs_new):
            M_prev = cv2.moments((m[1]>0).astype(np.uint8))
            cX_prev = int(M_prev["m10"] / M_prev["m00"])
            last_values = [track[-1] for track in self.mid_blob_tracks]

            if blob_area(m[0])>0: #non-empty child
                M = cv2.moments((m[0]>0).astype(np.uint8))
                cX = int(M["m10"] / M["m00"])        

                if cX_prev not in last_values:
                    raise Exception('Bad correlation. Parent not in list')

                self.mid_blob_tracks[last_values.index(cX_prev)].append(cX)
            else:#child is empty
                if DEBUG_MODE: print("{} NO CHILD.".format(ctr))
                self.mid_blob_tracks[last_values.index(cX_prev)].append(None)
                
        #Insert at beginning (right-most blobs)
        last_values = [track[-1] for track in self.mid_blob_tracks]
        try:
            start_none, end_none = get_non_None_section(last_values)
        except Exception as e:
            if hasattr(e, 'message'):
                print(e.message)
            else:
                print(e)
            #Return out of iterate() indicating failed tracking
            return False
            
        right_orphans = sorted(get_centroid_x(o[0]) for o in orphans if get_centroid_x(o[0]) > last_values[start_none])

        for cX in right_orphans:
            last_values = [track[-1] for track in self.mid_blob_tracks]
            for j, val in enumerate(last_values):
                if val is not None:
                    if j==0 and cX > self.mid_blob_tracks[j][-1]:
                        self.mid_blob_tracks = [[cX]] + self.mid_blob_tracks
                        break
                    elif j!=0 and self.mid_blob_tracks[j-1][-1] is None and cX > self.mid_blob_tracks[j][-1]:
                        self.mid_blob_tracks[j-1].append(cX)
                    #break is intentionally outside of elif
                    break
        #Insert at end (left-most blobs)
        last_values = [track[-1] for track in self.mid_blob_tracks]
        start_none, end_none = get_non_None_section(last_values)
        left_orphans = sorted(get_centroid_x(o[0]) for o in orphans if get_centroid_x(o[0]) < last_values[end_none])
        left_orphans.reverse()

        for cX in left_orphans:
            last_values = [track[-1] for track in self.mid_blob_tracks]
            for j, val in enumerate(reversed(last_values)):
                ctr = len(last_values)-1-j
                if val is not None:
                    if ctr==len(last_values)-1 and cX < self.mid_blob_tracks[ctr][-1]: #i.e. last element in un-reversed last_values)
                        self.mid_blob_tracks = self.mid_blob_tracks+[[cX]]
                        break

                    elif ctr!=len(last_values)-1 and self.mid_blob_tracks[ctr+1][-1] is None and cX < self.mid_blob_tracks[ctr][-1]:
                        self.mid_blob_tracks[ctr+1].append(cX)
                    #break is intentionally outside of elif
                    break
                    
        #Must remove None islands before inserting a centre blob into tracks
        last_values = [track[-1] for track in self.mid_blob_tracks] #THIS ONE IS NEW - CHECK THAT IT WORKS IF NOT REMOVE LINE
        start_none, end_none = get_non_None_section(last_values)
        temp_tracks=[]
        idx_is_none = [val is None for val in [track[-1] for track in self.mid_blob_tracks][start_none: end_none+1]]
        for j,val in enumerate(idx_is_none):
            if val is False:
                temp_tracks=temp_tracks+[self.mid_blob_tracks[j+start_none]]
        self.mid_blob_tracks = self.mid_blob_tracks[:start_none]+temp_tracks+self.mid_blob_tracks[end_none+1:]

        #Insert at centre (in between other blobs in frame)
        last_values = [track[-1] for track in self.mid_blob_tracks]
        start_none, end_none = get_non_None_section(last_values)
        centre_orphans = sorted(get_centroid_x(o[0]) for o in orphans if get_centroid_x(o[0]) < last_values[start_none] and get_centroid_x(o[0]) > last_values[end_none])
        
        for cX in centre_orphans:
            last_values = [track[-1] for track in self.mid_blob_tracks][start_none: end_none+1]
            # Must reverse list for insort to work properly (i.e. insort requires low to hi sorted)
            last_values.reverse()
            bi.insort(last_values, cX)
            last_values.reverse()
            idx = last_values.index(cX)+start_none
            self.mid_blob_tracks=self.mid_blob_tracks[:idx]+[[cX]]+self.mid_blob_tracks[idx:]
            
        if self.verbose:
            last_values = [track[-1] for track in self.mid_blob_tracks]
            plot_blobs_in_order([b[0] for b in blobs_new if blob_area(b[0])>0],blobs_current,last_values,plot=True)
        
#             fig=plt.figure(figsize=(24,24))
#             columns = 3
#             rows = 1
            
#             imgs = [intersect,img, img-intersect]
#             for i in range(1, columns*rows +1):
#                 fig.add_subplot(rows, columns, i)
#                 plt.imshow(imgs[i-1])
#             plt.show()
            
        return True #successful tracking

###################ENDOF TRACKER#################################


path = Path('~/SemanticSegmentationAirways')
path_lbl = path/'data/labels'
path_img = path/'data/images'
path_trained_model = path/'data/models'
# path_classifier_model = path/'data/models'
path_lbl = path_lbl.resolve()
path_img = path_img.resolve()

num_classes = 4 #everything_else, vocal_cords, tracheal_rings, bifurcation

DEBUG_MODE = False
LIVE_VIDEO = True
torch.nn.Module.dump_patches = True

#SETUP LEARNER

classifier = load_learner('.', 'classifier_export.pkl')
learn = load_learner('.', 'stage-2-big-0614-rn101.pkl')
classifier_class_dict={0:'larynx',1:'subglottis',2:'trachea', 3:'bifurcation'}

#SETUP VIDEO

# if LIVE_VIDEO:
#     cap = cv2.VideoCapture(0)
# else:
    # cap = cv2.VideoCapture('20181010_12y_5031752 mild subglottic stenosis uneditted.mpg')
# out = cv2.VideoWriter(str(path/'data/videos/trachea_map_demo.avi'),cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (720,480))
# print ("Cap fps: {}".format(cap.get(cv2.CAP_PROP_FPS)))
ctr = 0

######PARAMS######
#fps of the input video
fps=30
#start time for annotation (in seconds)
start_time_s= 45
#end time for annotation (in seconds)
end_time_s = 54
mask_timeline = []
clean_timeline = []
clean_flattened_timeline = []
frame_timeline = []

OVERLAY_SEGMENTATION = True

# debug only
eroded_timeline = []
###ENDOF PARAMS###


# Icons
larynx_icon=cv2.imread('./icons/larynx.png')
ring_off_icon=cv2.imread('./icons/ring_unselected.png')
ring_on_icon=cv2.imread('./icons/ring_selected.png')
dots_icon=cv2.imread('./icons/dotdotdot.png')
camera_icon=cv2.imread('./icons/camera.png', cv2.IMREAD_UNCHANGED)


#Advance to starting frame
# if not LIVE_VIDEO:
#     for i in range(0,start_time_s*fps):
#         ret, frame = cap.read()
#         ctr+=1
#     for _ in range(0,72):
#         ret, frame = cap.read()
#     ret, frame = cap.read() #frame is uint8
#     cv2.imshow("frame", frame)
# ret, frame = cap.read() 

#Set up UI
from PIL import ImageTk
# import threading
import os
import  datetime

# class UserInterface:
# 	def __init__(self, cap, outputPath):
# 		# store the video stream object and output path, then initialize
# 		# the most recently read frame, thread for reading frames, and
# 		# the thread stop event
# 		self.cap = cap
# 		self.outputPath = outputPath
# 		self.frame = None
# 		self.thread = None
# 		self.stopEvent = None

# 		# initialize the root window and image panel
# 		self.root = tki.Tk()
# 		self.panel = None

# 		# create a button, that when pressed, will take the current
# 		# frame and save it to file
# 		btn = tki.Button(self.root, text="Snapshot", command=self.takeSnapshot)
# 		btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

# 		# start a thread that constantly pools the video sensor for
# 		# the most recently read frame
# 		self.stopEvent = threading.Event()
# 		self.thread` = threading.Thread(target=self.videoLoop, args=())
# 		self.thread.start()

# 		# set a callback to handle when the window is closed
# 		self.root.wm_title("Tracheal Localizer")
# 		self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

# 	def videoLoop(self):
# 		try:
# 			# keep looping over frames until we are instructed to stop
# 			while not self.stopEvent.is_set():
# 				# grab the frame from the video stream and resize it to
# 				# have a maximum width of 300 pixels
# 				self.frame = self.cap.read()
# 				self.frame = imutils.resize(self.frame, width=300)
		
# 				# OpenCV represents images in BGR order; however PIL
# 				# represents images in RGB order, so we need to swap
# 				# the channels, then convert to PIL and ImageTk format
# 				image = self.frame[...,::-1]
# 				image = Image.fromarray(image)
# 				image = ImageTk.PhotoImage(image)
		
# 				# if the panel is not None, we need to initialize it
# 				if self.panel is None:
# 					self.panel = tki.Label(image=image)
# 					self.panel.image = image
# 					self.panel.pack(side="left", padx=10, pady=10)
		
# 				# otherwise, simply update the panel
# 				else:
# 					self.panel.configure(image=image)
# 					self.panel.image = image

# 		except:
# 			print("[INFO] caught a RuntimeError {}")

# 	def takeSnapshot(self):
# 		# grab the current timestamp and use it to construct the
# 		# output path
# 		ts = datetime.datetime.now()
# 		filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
# 		p = os.path.sep.join((self.outputPath, filename))

# 		# save the file
# 		cv2.imwrite(p, self.frame.copy())
# 		print("[INFO] saved {}".format(filename))

# 	def onClose(self):
# 		# set the stop event, cleanup the camera, and allow the rest of
# 		# the quit process to continue
# 		print("[INFO] closing...")
# 		self.stopEvent.set()
# 		self.cap.stop()
# 		self.root.quit()

# class App(threading.Thread):
#     def __init__(self):
#         threading.Thread.__init__(self)
#         self.start()

#     def callback(self):
#         self.root.quit()

#     def run(self):
#         self.root = tki.Tk()
#         self.root.protocol("WM_DELETE_WINDOW", self.callback)

#         UIoverlay = tki.IntVar()
#         tki.Checkbutton(self.root, text="overlay", variable=UIoverlay).grid(row=0, sticky=tki.W)
#         UItracking = tki.IntVar()
#         tki.Checkbutton(self.root, text="tracking", variable=UItracking).grid(row=1, sticky=tki.W)
#         tki.Button(self.root, text='Reset Tracker', command=None).grid(row=3, sticky=tki.W, pady=4)

#         while True:
#             image = PIL.Image.fromarray(frame)
#             image = ImageTk.PhotoImage(image)
#             panel = tki.Label(self.root, image=image)
#             panel.image = image
#             panel.grid(padx=10, pady=10)

#             self.root.mainloop()

# app = App()

class App():
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        self.main_loop = MainLoop()

        #For debugging only
        self.out = cv2.VideoWriter('./../data/videos/trachea_map_demo_tracking.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (720,480))

        # Create a canvas that can fit the above video source size
        self.canvas = tki.Canvas(window, width = 720, height = 480)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot=tki.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tki.CENTER, expand=True)
        self.UIoverlay = tki.IntVar(value=1)
        tki.Checkbutton(window, text="Overlay", variable=self.UIoverlay).pack()
        self.UItracking = tki.IntVar(value=1)
        tki.Checkbutton(window, text="Tracking", variable=self.UItracking).pack()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 1
        self.update()

        self.window.mainloop()
 
    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
 
        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
 
    def update(self):
        # Get a frame from the video source
        
        ret, frame = self.vid.get_frame()

        output_frame = self.main_loop.iterate(frame,overlay_segmentation=self.UIoverlay.get(), track_position=self.UItracking.get())
        # output_frame = (resize(output_frame, (output_frame.shape[0]*2, output_frame.shape[1]*2), anti_aliasing=True)*255).astype(np.uint8)
        self.out.write(output_frame)
 
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(output_frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tki.NW)
 
        self.window.after(self.delay, self.update)

class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        print("before")
        self.vid = cv2.VideoCapture(video_source)
        print("after")
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        else:
            print("Cap is open!")

        for i in range(0,45*30+73):
            ret, frame = self.vid.read()

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                # cv2.imshow("",frame)
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


###MAIN LOOP###
# clear_output()
print ("Starting!")

#For finding posterior region
# posterior_region_ctr = 0
# prev_posterior_angles = [int(224/4),int(224/4),int(224/4),int(224/4)]
#endof For finding posterior region


#For tracking
# tracker=None
#endof For tracking
# ctr=0
# time_taken = 0

class MainLoop():
    def __init__(self, overlay_segmentation=True, debug_mode=False):
        self.ctr = 0
        self.tracker = None

        self.posterior_region_ctr = 0
        self.prev_posterior_angles = [int(224/4),int(224/4),int(224/4),int(224/4)]

        # self.OVERLAY_SEGMENTATION = overlay_segmentation
        # self.DEBUG_MODE = debug_mode

    def iterate(self, frame, overlay_segmentation=True, track_position=True, classify_section=True, debug_mode=False):
        self.ctr+=1
        
        frame = crop_img(frame[...,::-1])
        frame = 255 * frame # Now scale by 255
        frame = frame.astype(np.uint8)
        
        # Perform inference
        mask = inference(frame)
        classifier_main, classifier_second = inference_classification(frame)

        # Convert from linear to polar
        polar_image = unwrap_image(mask)

        # Clean up linear image
        clean_img = cv2.erode(polar_image,np.ones((11,1)))
    #     clean_flattened_timeline.append((clean_img,(cX,cY)))
        
        #Finding posterior region
        posterior, self.prev_posterior_angles, self.posterior_region_ctr = find_posterior_region(clean_img, self.prev_posterior_angles, self.posterior_region_ctr)
        mask_continuous = get_posterior_corrected_frame(clean_img, posterior)
        
        #Tracking
        # clear_output(wait=True)
        print("ctr {}".format(self.ctr))
        
        img = mask_continuous==1
        output_image = frame.copy()
        output_image = (output_image/255).astype(np.float32)
        if overlay_segmentation:
            cv2.imshow("raw frame",(crop_img(frame)))
            cv2.imshow("2",(mask_to_colour(mask)))
            overlay = overlay_mask(crop_img(frame), mask_to_colour(mask), _alpha=0.9, show=False)
    #         pltimg(overlay[...,::-1])
            # overlay = resize(overlay, (min(output_image.shape[0:2]),min(output_image.shape[0:2])),anti_aliasing=True)
            overlay = crop_img(overlay)
            # print('shape {}'.format(overlay.shape))
            # cv2.imshow("overlay", overlay)
            output_image = overlay
        
        tracheal_map = np.zeros((480,120,3), dtype=np.float32)
        if self.tracker is None:
            self.tracker=Tracker(init_frame=img, _verbose=True)
            if track_position:
                tracheal_map = np.array(draw_trachea_map(tracheal_map, [], False)/255, dtype=np.float32)
            # mapp = img_as_float(mapp)
            # mapp = np.array((mapp/255), dtype=np.float32)
            # print("Mapp dtype: {}, {}".format(output_image.dtype,np.unique(output_image)))
            # print(np.unique(output_image))
            # print(mapp)
            # pltimg(mapp)
            # cv2.imshow("map", mapp)
            # if track_position:
            #     output_image = np.concatenate((output_image, mapp), axis=1)
            # cv2.imshow('concat', overlayed_map)
            # pltimg(overlayed_map)
        else:
            success = self.tracker.iterate(img)
            if track_position:
                tracheal_map = np.array(draw_trachea_map(tracheal_map, [track[-1] for track in self.tracker.mid_blob_tracks], success)/255, dtype=np.float32)

    #         pltimg(overlayed_map[...,::-1])
            
            # cv2.imshow("overlayed map", overlayed_map)
            # out.write(overlayed_map)
            #Restart tracker from next frame if tracking is not successful
            if not success: self.tracker = None
        
        output_image = np.concatenate((tracheal_map, crop_img(output_image, size=480), np.zeros((480,120,3), dtype=np.float32)), axis=1)
        output_image = put_classifier_text(output_image, classifier_main, classifier_second)
        # cv2.imshow("3", )

        if debug_mode:
            #Tests for optimal erosion level
            e_list = []
            for num in range(5,12):
                e = cv2.erode(polar_image,np.ones((num,1)))
                e_list.append(e)
            eroded_timeline.append(e_list)
            
            # Cleaned up linear image
            linear_image = cv2.linearPolar(e,(cX,cY), value, cv2.WARP_INVERSE_MAP+cv2.WARP_FILL_OUTLIERS)
            clean_timeline.append(linear_image)
        return ((output_image*255).astype(np.uint8)[...,::-1])


App(tki.Tk(), "SmartEndoscope", video_source='20181010_12y_5031752 mild subglottic stenosis uneditted.mpg')
# cap = MyVideoCapture(0)
# ret, frame = cap.get_frame()
# cv2.imshow("", frame)


print ("End!")
print("Success!")