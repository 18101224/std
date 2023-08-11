from models import Transformer
from models import pr_st,pr_ss
import torch
import torch.nn as nn
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import cv2 as cv
import models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st_model = Transformer()
st_model.load_state_dict(torch.load('./pretrained_model/Hayao_net_G_float.pth'))
st_model.to(device)
st_model.eval()
st_preprocessing = pr_st()
ss_preprocessing = pr_ss()
st_img = st_preprocessing.process('./udong.jpg')
ss_img = ss_preprocessing.process()

sam = sam_model_registry['vit_h'](checkpoint='../segment/sam_vit_h_4b8939.pth').to(device=device)
ss_model = SamAutomaticMaskGenerator(sam)

out = st_model(st_img)
#vutils.save_image(torch.tensor(out),'./result.jpg')
#img = cv.imread('result.jpg')
#cv.imshow('.',img)
#cv.waitKey(0)
#cv.destroyAllWindows()out = st_model(st_img)
out = models.denormalize(out[0])
cv.imshow('output',out)
cv.waitKey(0)
cv.destroyAllWindows()
mask = ss_model.generate(ss_img)[0]['segmentation']
ssed = models.segment(ss_img,mask)
cv.imshow('segmented',ssed)
cv.waitKey(0)
cv.destroyAllWindows()