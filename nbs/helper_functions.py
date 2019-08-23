import imutils
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float, img_as_uint, img_as_int, img_as_ubyte
from skimage import measure
from skimage.transform import resize
import cv2
import random

#Constants
num_classes = 4 #everything_else, vocal_cords, tracheal_rings, bifurcation


"""
 Utilities 
"""

def crop_img(img,size=224):
    """
    Crops image into a square in the middle, and resizes to size.

    Args:
        img: array or 2D list that is to be resizes
        size (int): width and height of output square image

    Returns:
        Resized image (dtype=float64)

    """
    cropx=np.min(img.shape[:2])
    cropy=cropx
    y,x = img.shape[:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return resize(img[starty:starty+cropy,startx:startx+cropx], (size,size),anti_aliasing=True) #dtype = float64

def mask_to_colour(mask):
    """
    Takes grayscale image (shape=(size x, size y)) and returns
    colour image (shape=(size x, size y, 3)) where:
    1->red
    2->green
    3->blue

    Args:
        grayscale image

    Returns:
        colour image

    """
    output_mask = np.zeros((mask.shape[0],mask.shape[1],3)).astype(np.uint8)
    green = [0,255,0]
    red = [255,0,0]
    blue = [0,0,255]
    other = [150,200,0]
    output_mask[mask==1]=red
    output_mask[mask==2]=green
    output_mask[mask==3]=blue
    return output_mask

def red_blue_swap(img):
    """
    Swaps the r and b channels, for converting between PIL Image and Opencv BGR np_array
    """
    return img[:, :, ::-1]

def _deprecated_plot_many(images):
    """
    Plots 3 images in the same figure

    Args:
        list of 3 images
    Return:
        None
    """
    plt.figure(figsize=(9, 3.5))

    
    for i in range(1,len(images)):
        plt.subplot(130+i)
        plt.imshow(images[i], cmap='nipy_spectral')
        plt.title(str(i))

    plt.tight_layout()
    plt.show()

def plot_many(images, titles, cmap='nipy_spectral'):
    """
    Plot all images in 'images' in same horizontal figure
    Args:
        images: list of images

        titles (optional): list of strings

        cmap (optional): cmap to plot images with (default='nipy_spectral')
    Returns:
        None
    """
    fig, axes = plt.subplots(ncols=len(images), figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    for i, img in enumerate(images):
        ax[i].imshow(image, cmap=cmap, interpolation='nearest')
        if i < len(titles):
            ax[i].set_title(titles[i])
    
    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    plt.show()




def sort_contours(cnts, method="left-to-right"):
    """
    Sort contours by the centroids of their bounding boxes

    Args:
        cnts: list of contours

        method(string): one of
            "left-to-right", right-to-left",
            "bottom-to-top", "top-to-bottom"
        
    Returns:
        tuple containing a list of sorted contours and a list of their
        corresponding bounding boxes
    """
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b:b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def contiguous_regions(condition):
    """
    Finds contiguous True regions of the boolean array "condition".
    Args:
        1D boolean array
    Return:
        2D array where the first column is the start index of the region and the
        second column is the end index.
        array len is number of separate True regions in 'condition'
    """

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx

def longest_contiguous_region(condition, histogram):
    """
    Return longest contiguous region in 1D boolean array "condition"
    Args:
        1D boolean array
    Return:
        region, centrepoint

        region: tuple with (start, end) of longest contiguous region

        centrepoint: centrepoint of the longest contiguous region
            i.e. index of centre of longest True region in "condition"

    """
    region = (0,0) # (start,stop)
    longest = (0,0)
    for start, stop in contiguous_regions(condition):
        segment = histogram[start:stop]
        if len(segment) > longest[1]-longest[0]:
            region = (start,stop)
    centrepoint = int((region[0]+region[1])/2)
    return region, centrepoint

def pltimg(img):
    """
    Immediately plot an image. plt.imshow(img) + plt.show()
    """
    plt.imshow(img)
    plt.show()

def weight_fn(x,slope):
    """
    To be used with avg_frames only
    slope: intensity with which new frames will be more
           heavily weighted than previous frames
    """
    return slope*x+1

def avg_frames(frames, slope=1):
    """
    Weighted average of frames in "frames"

    Args:
        frames: list of frames to average
                Most current frame must be last

        slope (default 1): intensity gradient with which more recent
               frames will be weighted.
               slope=0 -> all frames weighted equally
               slope>0 -> latter frames more heavily weighted than first
                          frames
    """
    blank_frame = np.zeros_like(frames[0])
    for (i, f) in enumerate(frames):

        blank_frame += np.array(f==1).astype(np.uint8)*weight_fn(i,slope)
    
    blank_frame = blank_frame/np.max(blank_frame)
    blank_frame = blank_frame>0.75

    return blank_frame



# ################################################################################
# ################################################################################
# ################################################################################
# ################################################################################
# ################################################################################
# ################################################################################



"""
 Blob Helper 
"""

def get_blobs(img, imshow=False):
    """
    Separates multi-class image (in which each class has a unique pixel value)
    into different blobs per class.


    Args:
        img: Image in which each class has a different value
        imshow: Plot blobs_labels (d) 
    Returns:
        blobs_labels: image with 'num_classes' channels. Each channel represents
                      a class and has unique pixel values for each blob.

                      i.e. blobs_labels[i]==j will give blob #j for class i
    
    """
    img2 = img_as_ubyte(np.array(img))

    # each channel in sep_classes contains a given class of labels.
    # sep_blobs[i].shape is (img.width,img.height,num_classes)
    sep_classes = []
    for i in range (0,num_classes):
        sep_classes.append( np.array(img2 == i ).astype(np.uint) )
        

    blobs_labels = [measure.label(blobs, background=0)[:,:,0] for blobs in sep_classes]


    if imshow:
        for i in range(1,num_classes):
            plt.subplot(130+i)
            plt.imshow(blobs_labels[i]*100, cmap='nipy_spectral')
            plt.title(str(i))

        plt.tight_layout()
        plt.show()
        
    return blobs_labels

def get_blobs_single_class(img, label_value):

    """
    Like get_blobs() but for a single class
    Args:
        Image with blobs (they can all have same pixel value)
    Returns:
        list in which each channel is grayscale image with a single blob
        All blobs have pixel value 255
    """

    blobs = measure.label(np.array(img_as_ubyte(np.array(img)) == label_value).astype(np.uint) )
    independent_blobs = []
    for i in np.unique(blobs)[1:]:
        single_blob = np.zeros_like(blobs)
        single_blob[blobs==i]=255
        independent_blobs.append(single_blob)
    return independent_blobs

def collapse_family_channels(blobs_new):
    """
    Only to be used with blob_new.
    Collapses all children blobs into a single frame

    Reverses get_blobs_single_class

    Args:
        blobs_new: list of tuples of (child_blob, parent_blob)
    Returns:
        all blobs in a single frame
    """
    blobs_new_frame = np.zeros_like(blobs_new[0][0])

    for b in blobs_new:
        blobs_new_frame = np.array(blobs_new_frame) | np.array(b[0])
    return blobs_new_frame


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


def blob_area(bin_img):
    """
    Calculate number of non-zero pixels in grayscale image
    Args:
        Grayscale image

    Returns:
        Area of non-zero pixels in image

    """
    return len(np.where(bin_img>0)[0])

def plot_blob_generations(blobs_new, orphans, include_orphans=True, frame=None):
    """
    Plots children blobs in same colours as parent blobs
    if include_orphans: Draws orphans in white
    if frame is not None: draw raw frame

    Args:
        blobs_new: family returned by correlate_blobs(...).
        list of tuples of (child_blob, parent_blob)
                   
        orphans: orphans returned by correlate_blobs(...).
        list of tuples of (orphan_blob, None)

        include_orphans: if True, draw orphans (in white)

        frame: raw frame at current time. Used for debugging
    """

    colours = np.linspace(10,249,40).astype(np.uint8)


    overlayed_children = np.zeros_like(blobs_new[0][0])
    overlayed_parents = np.zeros_like(blobs_new[0][0])


    if frame is None:
        fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(ncols=4, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    for (i,b) in enumerate(blobs_new):
        colour = random.randint(10,255)
        if len(np.unique(b[0])) > 0 :
            overlayed_children = (overlayed_children) | (b[0]/np.max(b[0])*colour).astype(np.uint8)
        if len(np.unique(b[1])) > 0:
            overlayed_parents = (overlayed_parents) | (b[1]/np.max(b[1])*colour).astype(np.uint8)
    
    
    
    if (include_orphans):
#         print("this one has orphans")
        overlayed_orphans = np.zeros_like(blobs_new[0][0])
        for (i,b) in enumerate(orphans):
#             print("orphan {}: {}".format(i, np.unique(b[0])))
            overlayed_children = (overlayed_children) | (b[0]/np.max(b[0])*255).astype(np.uint8)
            overlayed_orphans = (overlayed_orphans) | (b[0]/np.max(b[0])*255).astype(np.uint8)
#             print('orphan img')
        ax[2].imshow(overlayed_orphans, cmap='nipy_spectral', vmin=0, vmax=255)
        ax[2].set_title('orphans')
#             overlayed_parents = (overlayed_parents) | (b[1]/np.max(b[1])*colours[i]).astype(np.uint8)

    ax[0].imshow(overlayed_children, cmap='nipy_spectral', vmin=0, vmax=255)
    ax[0].set_title('child')
    ax[1].imshow(overlayed_parents, cmap='nipy_spectral', vmin=0, vmax=255)
    ax[1].set_title('parent')
    
    if frame is not None:
        ax[3].imshow(frame)
        ax[3].set_title('raw frame')
    
    fig.tight_layout()
    plt.show()
    
    return overlayed_children


def plot_blobs_in_order(blobs_in_frame):
    """
    Plots blobs in a gradient, in order in which the appear
    in blobs_in_frame

    Args:
        List in which each entry is grayscale frame with a single blob

    Returns:
        Frame with all blobs having unique pixel values, in increasing order
        corresponding to their index in blobs_in_frame
        
    """

    colour = np.linspace(10,249,40).astype(np.uint8)

    output_frame = np.zeros_like(blobs_in_frame[0])
    for (i,b) in enumerate(blobs_in_frame):
        output_frame = (output_frame) | (b/np.max(b)*colour[i]).astype(np.uint8)
    
    pltimg(output_frame)
    return output_frame

def get_middle_section(blobs, collapse_channels=True):
    """
    Returns middle section of frame in which each blob_middle_section has
    a unique pixel-wise value

    Args:
        List of frames with single blobs each
    Returns:
        Middle section of the collapsed blob frame
    """
    blobs=np.array(blobs).astype(np.uint8)
    if collapse_channels:
        overlayed_children = np.zeros_like(blobs[0][112:122,:]).astype(np.uint8)
    else:
        mid_blobs = []
    
    cnt = 0
    for i,b in enumerate(blobs):
        blob = b[112:122:]
        if blob_area(blob) > 1:
            cnt+=1
            if collapse_channels:
                overlayed_children = (overlayed_children) | (blob/np.max(blob)*(cnt)).astype(np.uint8)
            else:
                mid_blobs.append(  (blob/np.max(blob)*(255)).astype(np.uint8)  )

    return overlayed_children if collapse_channels else mid_blobs

# +
# def get_middle_section_multichannel(blobs):
#     """
#     Returns middle section of frame in which each blob_middle_section has
#     a unique pixel-wise value

#     Args:
#         List of frames with single blobs each
#     Returns:
#         Middle section of all blobs
#     """
#     blobs=np.array(blobs)
#     overlayed_children = np.zeros_like(blobs[0][112:122,:]).astype(np.uint8)
    
#     cnt = 0
#     for i,b in enumerate(blobs):
#         blob = b[112:122:]
#         if len(np.unique(blob)) > 1:
#             cnt+=1
#             overlayed_children = (overlayed_children) | (blob/np.max(blob)*(cnt)).astype(np.uint8)

#     return overlayed_children
# -

def sort_right_to_left(mid_sec):
    """
    Orders blobs that are in the middle section from right to left, and returns a
    list of tuples of (blob's unique value, centroid x position)
    """

    mid_blob_vals = []
    
    for u in np.unique(mid_sec)[1:]:
        M = cv2.moments((mid_sec==u).astype(np.uint8))
        cX = int(M["m10"] / M["m00"])
        mid_blob_vals.append((u,cX))
        
    mid_blob_vals.sort(key=lambda tup: tup[1], reverse=True) 
    return mid_blob_vals

def blobs_intersect(b1, b2, intersect_thresh=1):
    """
    Checks for the intersection of blobs b1 and b2
    Args:
        b1, b2: grayscale frames potentially containin non-zero values

        intersect_thresh(default=1): number of pixels by which b1 and b2 must
                intersect to be considered intersecting
    Returns:
        bool: if blobs intersect by more than intersect_thresh amount
               returns True. False otherwise
    """

    if b1.shape != b2.shape: raise Exception('blob shapes {} and {} do not match'.format(b1.shape, b2.shape))
    return blob_area((b1)&(b2))>intersect_thresh
