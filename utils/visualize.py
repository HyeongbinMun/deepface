import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image, ImageFont, ImageDraw
from configs import pose_estimation as config
from configs.detection import SOCCER_ENG, SOCCER_KOR

# Constants
ALPHA = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
HANGEUL_FONT = os.path.join(os.getcwd(), 'utils/font/malgun.ttf')
TEXT_SCALE = 1.0
TEXT_THICKNESS = 1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)

def gen_colors(num_colors):
    """Generate different colors.
    # Arguments
      num_colors: total number of colors/classes.
    # Output
      bgrs: a list of (B, G, R) tuples which correspond to each of
            the colors/classes.
    """
    import random
    import colorsys

    hsvs = [[float(x) / num_colors, 1., 0.7] for x in range(num_colors)]
    random.seed(1234)
    random.shuffle(hsvs)
    rgbs = list(map(lambda x: list(colorsys.hsv_to_rgb(*x)), hsvs))
    bgrs = [(int(rgb[2] * 255), int(rgb[1] * 255),  int(rgb[0] * 255))
            for rgb in rgbs]
    return bgrs

def draw_boxed_text(img, text, topleft, color):
    """Draw a transluent boxed text in white, overlayed on top of a
    colored patch surrounded by a black border. FONT, TEXT_SCALE,
    TEXT_THICKNESS and ALPHA values are constants (fixed) as defined
    on top.
    # Arguments
      img: the input image as a numpy array.
      text: the text to be drawn.
      topleft: XY coordinate of the topleft corner of the boxed text.
      color: color of the patch, i.e. background of the text.
    # Output
      img: note the original image is modified inplace.
    """
    assert img.dtype == np.uint8
    img_h, img_w, _ = img.shape
    if topleft[0] >= img_w or topleft[1] >= img_h:
        return img
    margin = 3
    size = cv2.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICKNESS)
    w = size[0][0] + margin * 2
    h = size[0][1] + margin * 2
    # the patch is used to draw boxed text
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[...] = color
    cv2.putText(patch, text, (margin+1, h-margin-2), FONT, TEXT_SCALE,
                WHITE, thickness=TEXT_THICKNESS, lineType=cv2.LINE_8)
    cv2.rectangle(patch, (0, 0), (w-1, h-1), BLACK, thickness=1)
    w = min(w, img_w - topleft[0])  # clip overlay at image boundary
    h = min(h, img_h - topleft[1])
    # Overlay the boxed text onto region of interest (roi) in img
    roi = img[topleft[1]:topleft[1]+h, topleft[0]:topleft[0]+w, :]
    cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, roi, 1 - ALPHA, 0, roi)
    return img


def draw_boxed_scene_text(img, text, topleft, color):
    """Draw a transluent boxed text in white, overlayed on top of a
    colored patch surrounded by a black border. FONT, TEXT_SCALE,
    TEXT_THICKNESS and ALPHA values are constants (fixed) as defined
    on top.
    # Arguments
      img: the input image as a numpy array.
      text: the text to be drawn.
      topleft: XY coordinate of the topleft corner of the boxed text.
      color: color of the patch, i.e. background of the text.
    # Output
      img: note the original image is modified inplace.
    """
    assert img.dtype == np.uint8
    img_h, img_w, _ = img.shape
    if topleft[0] >= img_w or topleft[1] >= img_h:
        return img
    margin = 7
    size = cv2.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICKNESS)
    w = size[0][0] + margin * 1 - 45
    h = size[0][1] + margin * 2
    # the patch is used to draw boxed text
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[...] = color
    # cv2.putText(patch, text, (margin+1, h-margin-2), FONT, TEXT_SCALE,
    #             WHITE, thickness=TEXT_THICKNESS, lineType=cv2.LINE_8)

    # for writing hangeul
    color_converted = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(color_converted)

    font_size = 17
    draw = ImageDraw.Draw(img_pil)
    draw.text((0, 0), text, font=ImageFont.truetype(HANGEUL_FONT, font_size), fill=(255, 255, 255))
    numpy_img = np.array(img_pil)
    patch = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)

    cv2.rectangle(patch, (0, 0), (w-1, h-1), BLACK, thickness=1)
    w = min(w, img_w - topleft[0])  # clip overlay at image boundary
    h = min(h, img_h - topleft[1])
    # Overlay the boxed text onto region of interest (roi) in img
    roi = img[topleft[1]:topleft[1]+h, topleft[0]:topleft[0]+w, :]
    cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, roi, 1 - ALPHA, 0, roi)
    return img


class Visualization():
    def __init__(self, object_class, pose_dataset="coco_wholebody"):
        self.object_class = object_class
        self.object_colors = gen_colors(len(object_class) + 1)
        self.pose_dataset = pose_dataset
        self.event_colors = {
            "falldown_recognition": (196, 114, 68),
            "mission1":             (49, 125, 237),
            "mission2":             (167, 167, 167),
            "mission3":             (0, 192, 255),
        }
        self.joints = {
            "keypoints":  config.keypoints[self.pose_dataset],
            "skeleton": config.skeleton[self.pose_dataset]
        }

    def draw_bboxes(self, img, detection_results, score_threshold=0.5):
        for detection_result in detection_results:
            score = detection_result["label"][0]["score"]
            cl = detection_result["label"][0]["class_idx"]
            cls_name = detection_result["label"][0]["description"]
            bbox = detection_result["position"]
            x_min = int(bbox["x"])
            y_min = int(bbox["y"])
            x_max = int(bbox["x"] + bbox["w"])
            y_max = int(bbox["y"] + bbox["h"])
            color = self.object_colors[cl]
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            if cls_name in SOCCER_ENG:
                txt_loc = (max(x_min, 0), max(y_min - 25, 0))
                cls_name = SOCCER_KOR[cl]
                txt = '{}: {:.2f}'.format(cls_name, score)
                img = draw_boxed_scene_text(img, txt, txt_loc, color)
            else:
                txt_loc = (max(x_min + 2, 0), max(y_min + 2, 0))
                txt = '{}: {:.2f}'.format(cls_name, score)
                img = draw_boxed_text(img, txt, txt_loc, color)
        return img

    def draw_text_boxes(self, image, detection_results):
        color = self.object_colors[-1]
        for detection_result in detection_results:
            if detection_result['label']['score'] > 1:
                color = (0, 0, 255)
            else:
                color = self.object_colors[-1]

            position = detection_result['position']
            top_left = (int(position["top_left"][0]), int(position["top_left"][1])) # top left
            top_right = (int(position["top_right"][0]), int(position["top_right"][1])) # top right
            bottom_right = (int(position["bottom_right"][0]), int(position["bottom_right"][1])) # bottom right
            bottom_left = (int(position["bottom_left"][0]), int(position["bottom_left"][1])) # bottom left

            image = cv2.line(image, top_left, top_right, color, 2)
            image = cv2.line(image, top_right, bottom_right, color, 2)
            image = cv2.line(image, bottom_right, bottom_left, color, 2)
            image = cv2.line(image, bottom_left, top_left, color, 2)
            text = detection_result["label"][0]["text"]
            score = detection_result["label"][0]["score"]
            txt_loc = (max(top_left[0] + 2, 0), max(top_left[1] + 2, 0))
            txt = '{}: {:.2f}'.format(text, score)
            image = draw_boxed_scene_text(image, txt, txt_loc, color)

        return image


    def put_text(self, img, model_result, model_name):
        if type(model_name) != list:
            model_name = [model_name]
        img_h, img_w, _ = img.shape
        for i, event_name in enumerate(model_name):
            color = self.event_colors[model_name]
            text = "{}:{}".format(model_name, str(model_result[model_name]))
            if model_result[model_name]:
                font_color = RED
            else:
                font_color = BLACK
            margin = 3
            topleft = (40, 40 + 80 * (i))
            text_scale = 4.0
            text_size = 6
            text_thickness = 2
            size = cv2.getTextSize(text, FONT, text_size, text_thickness)
            w = size[0][0] + margin * 2
            h = size[0][1] + margin * 2
            patch = np.zeros((h, w, 3), dtype=np.uint8)
            patch[...] = color
            cv2.putText(patch, text, (margin + 1, h - margin - 2), FONT, text_scale,
                        font_color, thickness=text_thickness, lineType=cv2.LINE_8)
            cv2.rectangle(patch, (0, 0), (w - 1, h - 1), BLACK, thickness=1)
            w = min(w, img_w - topleft[0])  # clip overlay at image boundary
            h = min(h, img_h - topleft[1])
            # Overlay the boxed text onto region of interest (roi) in img
            roi = img[topleft[1]:topleft[1] + h, topleft[0]:topleft[0] + w, :]
            cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, roi, 1 - ALPHA, 0, roi)

        return img

    def draw_points_and_skeleton(self, image, pose_results):
        points = self.reformat_pose_results(pose_results)
        skeleton = self.joints["skeleton"]
        if len(points) :
            image = self.draw_skeleton(image, points, skeleton)
            image = self.draw_points(image, points)
        return image

    @staticmethod
    def reformat_pose_results(pose_results):
        points = []
        for i, pose_result in enumerate(pose_results):
            if "pose" in pose_result:
                dict_pose = pose_result["pose"]
                person = []
                for p in range(0, 133) :
                    x = dict_pose[p][0]
                    y = dict_pose[p][1]
                    score = pose_result["label"]["score"]
                    person.append([y,x])
                points.append(person)
        return points

    def draw_points(self, image, points):
        pose_kpt_color = config.pose_kpt_color[self.pose_dataset]
        circle_size = max(1, min(image.shape[:2]) // 160)  # ToDo Shape it taking into account the size of the det

        for p, person in enumerate(points):
            for i, point in enumerate(person[:-1]):
                pose = (int(point[1]), int(point[0]))
                color = tuple(int(c) for c in pose_kpt_color[i])
                image = cv2.circle(image, pose, circle_size, color, -1)

        return image

    def falldown_visual(self, image, points):
        font=cv2.FONT_HERSHEY_SIMPLEX
        for p, person in enumerate(points):
            x,y,w,h=person['position']['x'],person['position']['y'],person['position']['w'],person['position']['h']
            cv2.putText(image,str(person['label']['falldown']),(x,y),font,1,(0,0,255),2)
        return image

    def draw_skeleton(self, image, points, skeleton):
        pose_link_color = config.pose_link_color[self.pose_dataset]

        for p, person in enumerate(points):
            for i, joint in enumerate(skeleton):
                pt1 = person[joint[0]]
                pt2 = person[joint[1]]
                color = tuple(int(c) for c in pose_link_color[i])
                pose1 = (int(pt1[1]), int(pt1[0]))
                pose2 = (int(pt2[1]), int(pt2[0]))
                image = cv2.line(image, pose1, pose2, color, 2)

        return image