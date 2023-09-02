from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torch import jit

from data import cfg_mnet, cfg_re50, cfg_resnet18
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time


parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.9, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=1000, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='display image')
parser.add_argument('-save', '--save_jit', action="store_true", default=True, help='save jit trace')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    elif args.network == "resnet18":
        cfg = cfg_resnet18
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    
    print('Finished loading model!')
    cudnn.benchmark = False
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    resize = 1
    cap = cv2.VideoCapture("./Ellen DeGeneres takes a selfie at the Oscars-GsSWj51uGnI.mp4", cv2.CAP_FFMPEG)
    priors, prior_data = None, None
    
    # testing begin
    while(cap.isOpened()):
        tic = time.time()
        
        ret, img_raw = cap.read()
        img_raw = cv2.imread("image.jpg")
        
        img = np.float32(img_raw)
        
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)
        
        loc, conf, landms = net(img)  # forward pass
        toc = time.time()
        
        if args.save_jit and priors is None:
            net_trace = jit.trace(net, img)
            jit.save(net_trace, 'model.zip')
            print("model saved: model.zip")
        
        # print('Net forward time: {:.4f}'.format(toc - tic))
        
        if priors is None:
            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(device)
            prior_data = priors.data
            
        # print(prior_data.shape)
        # print(prior_data[0])
        # print(conf.shape)
        # print(conf[:,0])
        # exit()
            
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])

        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()

        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()
        # print('Post process time: {:.4f}'.format((time.time() - tic) - (toc - tic) ))
        
        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        
        print(boxes)
        print(boxes.shape)
        exit()

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        
        # show image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                
                x = int(( b[0] + b[2] ) / 2)
                y = int(( b[1] + b[3] ) / 2)
                
                w = b[2] - b[0]
                h = b[3] - b[1]
            
                # landms
                _direction = np.degrees(np.arctan2((b[8] - b[6]),(b[7] - b[5])))
                eye_center = ((b[7] + b[5])//2, (b[8] + b[6])//2 )
                
                M = cv2.getRotationMatrix2D( eye_center , _direction, 1)
                img_rot = cv2.warpAffine(img_raw, M, (im_width, im_height))
                
                crop = img_rot[b[1]:b[1]+h,b[0]:b[0]+w,:]
                if 0 in crop.shape:
                    continue
                cv2.imshow("Crop", crop)
                
                cv2.ellipse(img_raw, (x, y), (int(w/2), int(h/2)), 0, 0, 360, (0, 255, 0), 3)
                
                # cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.circle(img_raw, eye_center, 1, (0, 0, 0), 5)
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 5)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 5)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 5)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 5)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 5)
                # print('FPS : {:.2f}'.format( (1000/(time.time() - tic))/1000))
                # print('Post process time: {:.4f}'.format((time.time() - tic) - (toc - tic) ))
            # save image

            name = "test.jpg"
            #cv2.imwrite(name, img_raw)
            cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
            cv2.imshow("Result", img_raw)
            if cv2.waitKey(1) == ord('q'):
                break 

