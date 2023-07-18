import math
from pathlib import Path

from face_detector import YoloDetector
import numpy as np
from PIL import Image, ExifTags, ImageOps
from os import path
import os

model = YoloDetector(target_size=720, device="cuda:0", min_face=90)


# 7.13  增加脸部镜像,取消直接resize到512*512，增加corp模式2:3

# 脸部裁切的小脚本
# @param1   model             face corping model
# @param2   face_dir          face dir
# @param3   dilate = 0.4      face bbox dilate arg
# @parma4   corp_mode         0,原bb直接膨胀   1，取1:1的脸部进行膨胀 ，2   取2：3的脸部进行膨胀
def face_corping(model, face_dir, dilate=1.0, corp_mode=2):
    # 先建个保存图片的文件夹
    savedir = Path(face_dir).parent.joinpath("cropped_" + path.basename(Path(face_dir)))
    if not path.exists(savedir):
        os.mkdir(savedir)
    files = os.listdir(face_dir)
    images = list(filter(lambda x: True if ".jpg" in x or ".png" in x else False,files))
    for index,filename in enumerate(images):
        try:
            orgimg = Image.open(path.join(face_dir, filename))
            # 解决图像由于拍摄器材导致的旋转问题
            orgimg = ImageOps.exif_transpose(orgimg)
            # for orientation in ExifTags.TAGS.keys():
            #     if ExifTags.TAGS[orientation] == 'Orientation': break
            # exif = dict(orgimg._getexif().items())
            # if exif[orientation] == 3:
            #     orgimg = orgimg.rotate(180, expand=True)
            # elif exif[orientation] == 6:
            #     orgimg = orgimg.rotate(270, expand=True)
            # elif exif[orientation] == 8:
            #     orgimg = orgimg.rotate(90, expand=True)
            if index == 109:
                print("catch")
            orgimg = np.array(orgimg)
            # data preprocess
            if filename.split(".")[1] == 'png' and orgimg.shape[2]==4:
                orgimg = orgimg[:, :, :-1]

            # model output
            bboxes, points = model.predict(orgimg, conf_thres=0.5)
            # for item in points:
            #     orgimg[item[0],item[1],:] = []
            # nx1, nx2, ny1, ny2 = 0, 0, 0, 0
            # output corpface and save
            if corp_mode == 0:
                len_bb = bboxes[0][0][3] - bboxes[0][0][1]
                width_bb = bboxes[0][0][2] - bboxes[0][0][0]
                len_adjust = int(len_bb * dilate)
                width_adjust = int(width_bb * dilate)
                nx1 = max(bboxes[0][0][0] - len_adjust, 0)
                nx2 = min(bboxes[0][0][2] + len_adjust, orgimg.shape[1])
                ny1 = max(bboxes[0][0][1] - width_adjust, 0)
                ny2 = min(bboxes[0][0][3] + width_adjust, orgimg.shape[0])
            elif corp_mode == 1:
                len_bb = bboxes[0][0][3] - bboxes[0][0][1]
                width_bb = bboxes[0][0][2] - bboxes[0][0][0]
                centerY = int((bboxes[0][0][3] + bboxes[0][0][1]) / 2)
                centerX = int((bboxes[0][0][2] + bboxes[0][0][0]) / 2)
                disYmax = min(orgimg.shape[0] - centerY,centerY)
                disXmax = min(orgimg.shape[1] - centerX,centerX)
                adjust = int(min(min(disXmax,disYmax),len_bb* (dilate + 1)/2,width_bb*(dilate + 1)/2))
                # lenWidth_max = int(max(len_bb, width_bb) * (dilate + 1))
                # center_x = (((bboxes[0][0][2]-bboxes[0][0][0])>>1),((bboxes[0][0][3]-bboxes[0][0][1])>>1))
                # adjust_x = int((lenWidth_max - len_bb) / 2)
                # adjust_y = int((lenWidth_max - width_bb) / 2)

                nx1 = centerX - adjust
                nx2 = centerX + adjust
                ny1 = centerY - adjust
                ny2 = centerY + adjust
            else:
                len_bb = bboxes[0][0][3] - bboxes[0][0][1]
                width_bb = bboxes[0][0][2] - bboxes[0][0][0]
                centerY = int((bboxes[0][0][3] + bboxes[0][0][1]) / 2)
                centerX = int((bboxes[0][0][2] + bboxes[0][0][0]) / 2)
                disYmax = min(orgimg.shape[0] - centerY, centerY)
                disXmax = min(orgimg.shape[1] - centerX, centerX)
                dis = min(disXmax,disYmax)
                ad_y = min(disYmax, int(((1 + dilate) / 2) * len_bb))
                ad_x = int(ad_y*0.66)

                nx1 = centerX - ad_x
                nx2 = centerX + ad_x
                ny1 = centerY - ad_y
                ny2 = centerY + ad_y
            # 仅保存一个角度
            corpface = Image.fromarray(orgimg[ny1:ny2, nx1:nx2, :])

            # corpface = corpface.resize((512,512))
            face_flip = corpface.transpose(Image.FLIP_LEFT_RIGHT)
            if index%2==0:
                corpface.save(path.join(savedir,str(index)+".jpg"))
            else:
                face_flip.save(path.join(savedir,str(index)+".jpg"))
            # output face ratio  后续拓展
            # ratio = ((bboxes[0][0][2] - bboxes[0][0][0]) * (bboxes[0][0][3] - bboxes[0][0][1])) / (
            #         orgimg.shape[0] * orgimg.shape[1])
        except Exception as e:
            print(e)

def face_flip( face_dir):
    savedir = Path(face_dir).parent.joinpath("flipped_" + path.basename(Path(face_dir)))
    if not path.exists(savedir):
        os.mkdir(savedir)
    files = os.listdir(face_dir)
    images = list(filter(lambda x: True if ".jpg" in x or ".png" in x else False, files))
    for index, filename in enumerate(images):
        try:
            orgimg = Image.open(path.join(face_dir, filename))
            # 解决图像由于拍摄器材导致的旋转问题
            orgimg = ImageOps.exif_transpose(orgimg)
            orgimg = np.array(orgimg)
            # data preprocess
            if filename.split(".")[1] == 'png':
                orgimg = orgimg[:, :, :-1]
            orgimg = Image.fromarray(orgimg)
            face_flip = orgimg.transpose(Image.FLIP_LEFT_RIGHT)
            if index%2 == 0:
                orgimg.save(path.join(savedir, str(index) + ".jpg"))
            else:
                face_flip.save(path.join(savedir, str(index) + ".jpg"))
        except Exception as e:
            print(e)
            
face_corping(model, "D:/CodeFormer/results/superRe_0.9/final_results")
# face_flip("E:/50_baijingting")