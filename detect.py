# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync


class Yolov5:

    @torch.no_grad()
    def __init__(self,
                 weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
                 source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
                 imgsz=640,  # inference size (pixels)
                 conf_thres=0.25,  # confidence threshold
                 iou_thres=0.45,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 view_img=False,  # show results
                 save_txt=False,  # save results to *.txt
                 save_conf=False,  # save confidences in --save-txt labels
                 save_crop=False,  # save cropped prediction boxes
                 nosave=False,  # do not save images/videos
                 classes=None,  # filter by class: --class 0, or --class 0 2 3
                 agnostic_nms=False,  # class-agnostic NMS
                 augment=False,  # augmented inference
                 visualize=False,  # visualize features
                 update=False,  # update all models
                 project=ROOT / 'runs/detect',  # save results to project/name
                 name='exp',  # save results to project/name
                 exist_ok=False,  # existing project/name ok, do not increment
                 line_thickness=3,  # bounding box thickness (pixels)
                 hide_labels=False,  # hide labels
                 hide_conf=False,  # hide confidences
                 half=False,  # use FP16 half-precision inference
                 dnn=False,  # use OpenCV DNN for ONNX inference
                 ):
        # myinit
        self.frozen_func, self.input_details, self.int8, self.interpreter, self.modelc, self.net, self.session, = None, None, None, None, None, None, None
        self.augment = augment
        self.dnn = dnn
        self.half = half
        self.visualize = visualize
        self.output_details = None
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.save_crop = save_crop
        self.line_thickness = line_thickness
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.view_img = view_img
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf

        source = str(source)
        self.save_img = not nosave and not source.endswith('.txt')  # save inference images
        self.webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        self.save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        self.device = select_device(device)
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        w = str(weights[0] if isinstance(weights, list) else weights)
        self.classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
        check_suffix(w, suffixes)  # check weights have acceptable suffix
        self.pt, self.onnx, self.tflite, self.pb, self.saved_model = (suffix == x for x in suffixes)  # backend booleans
        self.stride, self.names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        if self.pt:
            self.model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=self.device)
            self.stride = int(self.model.stride.max())  # model stride
            self.names = self.model.module.names if hasattr(self.model,
                                                            'module') else self.model.names  # get class names
            if self.half:
                self.model.half()  # to FP16
            if self.classify:  # second-stage classifier
                self.modelc = load_classifier(name='resnet50', n=2)  # initialize
                self.modelc.load_state_dict(torch.load('resnet50.pt', map_location=self.device)['model']).to(
                    self.device).eval()
        elif self.onnx:
            if self.dnn:
                # check_requirements(('opencv-python>=4.5.4',))
                self.net = cv2.dnn.readNetFromONNX(w)
            else:
                check_requirements(('onnx', 'onnxruntime-gpu' if torch.has_cuda else 'onnxruntime'))
                import onnxruntime
                self.session = onnxruntime.InferenceSession(w, None)
        else:  # TensorFlow models
            check_requirements(('tensorflow>=2.4.1',))
            import tensorflow as tf
            if self.pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""),
                                                   [])  # wrapped import
                    return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                                   tf.nest.map_structure(x.graph.as_graph_element, outputs))

                graph_def = tf.Graph().as_graph_def()
                graph_def.ParseFromString(open(w, 'rb').read())
                self.frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
            elif self.saved_model:
                self.model = tf.keras.models.load_model(w)
            elif self.tflite:
                self.interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
                self.interpreter.allocate_tensors()  # allocate
                self.input_details = self.interpreter.get_input_details()  # inputs
                self.output_details = self.interpreter.get_output_details()  # outputs
                self.int8 = self.input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        # Dataloader
        if self.webcam:
            self.view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
            bs = 1  # batch_size
        self.vid_path, self.vid_writer = [None] * bs, [None] * bs

        # Run inference
        if self.pt and self.device.type != 'cpu':
            self.model(
                torch.zeros(1, 3, *self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        # self.getObjectAix(dataset)

        # # Print results
        # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        # print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        # if save_txt or save_img:
        #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #     print(f"Results saved to {colorstr('bold', save_dir)}{s}")
        # if update:
        #     strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    def getObjectAix(self, dataset):
        lines = []
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, img, im0s, vid_cap in dataset:
            t1 = time_sync()
            if self.onnx:
                img = img.astype('float32')
            else:
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            if self.pt:
                self.visualize = increment_path(self.save_dir / Path(path).stem,
                                                mkdir=True) if self.visualize else False
                pred = self.model(img, augment=self.augment, visualize=self.visualize)[0]
            elif self.onnx:
                if self.dnn:
                    self.net.setInput(img)
                    pred = torch.tensor(self.net.forward())
                else:
                    pred = torch.tensor(
                        self.session.run([self.session.get_outputs()[0].name],
                                         {self.session.get_inputs()[0].name: img}))
            else:  # tensorflow model (tflite, pb, saved_model)
                imn = img.permute(0, 2, 3, 1).cpu().numpy()  # image in numpy
                if self.pb:
                    import tensorflow as tf
                    pred = self.frozen_func(x=tf.constant(imn)).numpy()
                elif self.saved_model:
                    pred = self.model(imn, training=False).numpy()
                elif self.tflite:
                    if self.int8:
                        scale, zero_point = self.input_details[0]['quantization']
                        imn = (imn / scale + zero_point).astype(np.uint8)  # de-scale
                    self.interpreter.set_tensor(self.input_details[0]['index'], imn)
                    self.interpreter.invoke()
                    pred = self.interpreter.get_tensor(self.output_details[0]['index'])
                    if self.int8:
                        scale, zero_point = self.output_details[0]['quantization']
                        pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
                pred[..., 0] *= self.imgsz[1]  # x
                pred[..., 1] *= self.imgsz[0]  # y
                pred[..., 2] *= self.imgsz[1]  # w
                pred[..., 3] *= self.imgsz[0]  # h
                pred = torch.tensor(pred)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                       max_det=self.max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if self.webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # img.jpg
                txt_path = str(self.save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                        lines.append(line)
                        if self.save_txt:  # Write to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if self.hide_labels else (
                                self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            if self.save_crop:
                                save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg',
                                             BGR=True)

                # Print time (inference-only)
                print(f'{s}Done. ({t3 - t2:.3f}s)')

                # Stream results
                im0 = annotator.result()
                if self.view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if self.save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if self.vid_path[i] != save_path:  # new video
                            self.vid_path[i] = save_path
                            if isinstance(self.vid_writer[i], cv2.VideoWriter):
                                self.vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            self.vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                                                 (w, h))
                        self.vid_writer[i].write(im0)
        return lines


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    yolo = Yolov5(**vars(opt))
    ds = LoadImages('D:/pythonproject/yolov5/resource/test1/', img_size=yolo.imgsz, stride=yolo.stride,
                    auto=yolo.pt)
    lines = yolo.getObjectAix(ds)
    print(lines)
