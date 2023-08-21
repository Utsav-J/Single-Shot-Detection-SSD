import torch
from ssd import build_ssd
from data import BaseTransform, VOC_CLASSES as labelmap
import cv2 as cv
from torch.autograd import Variable
import imageio


def detect(frame,net,transform):
    height,width = frame.shape[0:2]
    frame_t = transform(frame)[0]
    frame_t = torch.from_numpy(frame_t).permute(2,0,1)
    x = Variable(frame_t.unsqueeze(0))
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width,height,width,height])
    for i in range(detections.size(1)):
        j = 0
        while (detections[0,i,j,0] >= 0.6):
            pt = (detections[0,i,j,1:] * scale).numpy()
            cv.rectangle(frame,(int(pt[0]),int(pt[1])),(int(pt[2]),int(pt[3])),(0,0,255),2,cv.LINE_AA)
            cv.putText(frame,labelmap[i-1],(int(pt[0]),int(pt[1])),cv.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv.LINE_AA)
            j += 1
    return frame

net = build_ssd('test')
net.load_state_dict(torch.load('Udemy\Code for Windows\ssd300_mAP_77.43_v2.pth',map_location = lambda storage, loc: storage))


transform = BaseTransform(net.size,(104/256.0, 117/256.0, 123/256.0))

reader = imageio.get_reader('Udemy\Code for Windows\epic_horses.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('Udemy\Code for Windows\output.mp4',fps = fps)

for i,frame in enumerate(reader):
    frame = detect(frame,net.eval(),transform)
    writer.append_data(frame)
    print(i)
writer.close()