import os
import torch
from models import twinmixing as net
import torch.backends.cudnn as cudnn
import DataSet as myDataLoader
from argparse import ArgumentParser
from utils import train, val, netParams, save_checkpoint, poly_lr_scheduler
from copy import deepcopy
import math
import yaml

class ModelEMA:
    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

def val_net(args, hyp):
    use_ema = args.ema
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    model = net.TwinMixing(args)

    if num_gpus > 1:
        model = torch.nn.DataParallel(model)

    if not args.is320:
        valLoader = torch.utils.data.DataLoader(
            myDataLoader.Dataset(args.data_dir, hyp, valid=True),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    else:
        valLoader = torch.utils.data.DataLoader(
            myDataLoader.Dataset320(args.data_dir, hyp, valid=True),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    
    device = None
    if cuda_available:
        args.onGPU = True
        device = torch.device("cuda")
        model = model.to(device)
        cudnn.benchmark = True
    else:
        args.onGPU = False
        device = torch.device("cpu")
    print(device)
    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    if use_ema:
        ema = ModelEMA(model)
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            if args.pretrained.split(".")[-1] == "pth":
                print("=> loading checkpoint '{}'".format(args.pretrained))
                checkpoint = torch.load(args.pretrained)
                if use_ema:
                    ema.ema.load_state_dict(checkpoint)
                    if args.half:
                        ema.ema.half()
                else:
                    model.load_state_dict(checkpoint)
                    if args.half:
                        model.half()
                print("=> loaded checkpoint '{}'"
                    .format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
    da_segment_results,ll_segment_results = val(valLoader, ema.ema if use_ema else model,is320=args.is320, half=args.half, args=args, device=device) #da_mIoU_seg, ll_IoU_seg
    msg =  'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                    'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})'.format(
                        da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2],
                        ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1],ll_seg_miou=ll_segment_results[2])
    print(msg)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=16, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--hyp', type=str, default='./hyperparameters/twinmixing_hyper.yaml', help='hyperparameters path')
    parser.add_argument('--pretrained', type=str, default='', help='Pretrained weights path')
    parser.add_argument('--type', default="nano", help='')
    parser.add_argument('--is320', action='store_true')
    parser.add_argument('--seda', action='store_true', help='sigle encoder for Drivable Segmentation')
    parser.add_argument('--sell', action='store_true', help='sigle encoder for Lane Segmentation')
    parser.add_argument('--ema', action='store_true', help='')
    parser.add_argument('--half', action='store_true', help='')
    parser.add_argument('--verbose', action='store_true', help='')
    parser.add_argument('--data_dir', type=str, default='./dataset/', help='Dataset directory')
    args = parser.parse_args()
    with open(args.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    val_net(args, hyp.copy())
