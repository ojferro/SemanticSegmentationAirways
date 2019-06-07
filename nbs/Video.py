import sys
from fastai.vision import *
from fastai.vision import Image
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
import cv2
from matplotlib import pyplot as plt
import numpy as np
# import skvideo.io
# from skvideo.io import VideoWriter

def acc_trachea(input, target):
    target = target.squeeze(1)
    return (input.argmax(dim=1)==target).float().mean()

def bgr_to_tensor(frame):
    return 1 - torch.tensor(np.ascontiguousarray(np.flip(frame, 2)).transpose(2,0,1)).float()/255
def tensor_to_bgr():
    return 

class Segmenter:
    def segment_video(self):
        learn = None
        cap = None
        out = None

    #     def setup(self):
        metrics=acc_trachea


        path = Path('/storage')
        path_lbl = path/'vocal_chords_and_rings_data/data/labels'
        path_img = path/'vocal_chords_and_rings_data/data/images'
        path_lbl = path_lbl.resolve()
        path_img = path_img.resolve()

        get_y_fn = lambda x: path_lbl/f'{x.stem}{x.suffix}'

        codes = np.loadtxt(path_lbl/'../../codes.txt', dtype=str);
        bs = 4
        wd=1e-2


        src = (SegmentationItemList.from_folder(path_img)
               .split_by_fname_file('../../valid.txt')
               .label_from_func(get_y_fn, classes=codes))
        data = (src.transform(get_transforms(flip_vert=True), size=224, tfm_y=True)
                .databunch(bs=bs)
                .normalize(imagenet_stats))
        #Create empty data object
#         data = ImageDataBunch.single_from_classes(path_img, codes, tfms=get_transforms(), size=224).normalize(imagenet_stats)

        learn = unet_learner(data, models.resnet34)#, metrics=metrics, wd=wd)
        learn.load('stage-2-big-new-data')

        #Read in video and process frame by frame
        cap = cv2.VideoCapture('/storage/vocal_chords_and_rings_data/data/videos/20180825_9m_2771576_bilateralsu.avi')
        out = cv2.VideoWriter('/storage/vocal_chords_and_rings_data/data/videos/output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (224,224))
        
#         writer = VideoWriter('/storage/vocal_chords_and_rings_data/data/videos/output2.avi', frameSize=(224,224))
#         writer.open()

        print ("Starting!")
        while(cap.isOpened()):
            sys.stdout.write(" ... ")
            ret, frame = cap.read()

            #rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            #Convert from OpenCV image to pytorch tensor, to Image
            frame = 
cv2.resize(frame, (224,224))
            t = bgr_to_tensor(frame)
            t = Image(t)
            prediction = learn.predict(t)
            p = Image(prediction)
            #type(prediction) is tuple
            #print (type(prediction))
            
            #prediction = image2np(prediction)*255 # convert to numpy array in range 0-255
            #prediction = npim.astype(np.uint8) # convert to int
            
            
#             plt.imshow(prediction)
#             plt.title('my picture')
#             plt.show()
            #cv2.imshow('frame',frame)
            #cv2.waitKey(0)
            print (type(prediction))
            prediction = np.asarray(prediction, dtype="uint8", copy=True)
            print (type(prediction))
            #prediction = prediction
            out.write(prediction)#cv2.resize(prediction,(224,224)))
#             writer.write(prediction)

            
#             outputdata = .astype(np.uint8)

#             skvideo.io.vwrite("outputvideo.avi", outputdata)


        cap.release()
        writer.release()
        print ("End!")
        
        
# import time

# img = open_image('/storage/vocal_chords_and_rings_data/data/images/bc9cd224-0000000.png')
# start = time.time()
# prediction = learn.predict(img)
# end = time.time()
# print("Prediction time: {}".format(end - start))
# prediction[0].show(figsize=(5,5))

