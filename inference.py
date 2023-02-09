import os
import cv2
import time
import argparse

from tqdm import tqdm
from deepface import DeepFace
from configs.detection import PERSON
from utils.visualize import Visualization
from utils.utils import createfolder
from configs.detection import SOCCER_ENG

class Face:
    def __init__(self, option):
        self.media_path = option.media_path
        self.db_path = option.db_path
        self.save_path = option.save_path
        createfolder(self.save_path)
        self.model = option.model
        self.backend = option.backend
        self.class_dict = SOCCER_ENG
        self.infer_type = option.infer_type
        self.data_type = option.data_type
        self.vis = Visualization(self.class_dict)
        self.infer_time = 0

    def recognition_matching(self, rec_result):
        # identity  source_x  source_y  source_w  source_h  ArcFace_cosine
        results = []
        for result in rec_result:
            tmp = []
            identity = [i for i in range(result.shape[0] - 1, -1, -1)]
            recall_top1_idx = identity[0]
            bset = result.loc[recall_top1_idx]
            best_dict = bset.to_dict()
            class_name_list = best_dict['identity'].split('/')
            class_name = class_name_list[-2]
            re = {'description': class_name,
                  'score': best_dict['ArcFace_cosine'],
                  'class_idx': SOCCER_ENG.index(class_name)}
            tmp.append(re)
            position = {'x': best_dict['source_x'], 'y': best_dict['source_y'],
                        'w': best_dict['source_w'], 'h': best_dict['source_h']}
            label = {'label': tmp, 'position': position}
            results.append(label)
        return results

    def detection_matching(self, det_result):
        result = []
        for el in det_result:
            tmp = []
            re = {'description': 'face',
                  'score': el['confidence'],
                  'class_idx': 0}
            tmp.append(re)
            label = {'label': tmp, 'position': el['facial_area']}
            result.append(label)
        return result

    def face_detection(self, path):
        start_time = time.time()
        face_objs = DeepFace.extract_faces(img_path=path,
                                           target_size=(224, 224),
                                           detector_backend=self.backend)
        end_time = time.time()
        total_infer_time = end_time - start_time
        self.infer_time += total_infer_time
        return face_objs

    def face_recognition(self, path):
        start_time = time.time()
        dfs = DeepFace.find(img_path=path,
                            db_path=self.db_path,
                            detector_backend=self.backend,
                            model_name=self.model)
        end_time = time.time()
        total_infer_time = end_time - start_time
        self.infer_time += total_infer_time
        return dfs

    def data_folder_list(self):
        img_list = os.listdir(self.media_path)
        if '.' in img_list[0]:
            real_img_list = img_list
        else:
            real_img_list = []
            for img_dir in img_list:
                img_dir_path = os.path.join(self.media_path, img_dir)
                img_tmp_list = os.listdir(img_dir_path)
                for img in img_tmp_list:
                    img_path = os.path.join(img_dir, img)
                    real_img_list.append(img_path)

        return real_img_list

    def data_folder_check(self, img):
        if '/' in img:
            img_dir, img_name = img.split('/')
            path = os.path.join(self.save_path, img_dir)
            createfolder(path)

    def db_create(self):
        img_list = self.data_folder_list()

        for img in tqdm(img_list, ncols=100, leave=True):
            img_path = os.path.join(self.media_path, img)
            try:
                infer_result = self.face_detection(img_path)

                result = self.detection_matching(infer_result)
                image = cv2.imread(img_path)
                bbox_image = self.vis.draw_bboxes(image, result)
                self.data_folder_check(img)
                save_img_path = os.path.join(self.save_path, img)
                cv2.imwrite(save_img_path, bbox_image)
            except ValueError:
                print('\n{0} is not face'.format(img_path))

    def inference_by_image(self):
        img_list = self.data_folder_list()

        for img in tqdm(img_list, ncols=100, leave=True):
            img_path = os.path.join(self.media_path, img)
            try:
                if self.infer_type == 'detection':
                    infer_result = self.face_detection(img_path)
                    result = self.detection_matching(infer_result)
                elif self.infer_type == 'recognition':
                    recognition_result = self.face_recognition(img_path)
                    result = self.recognition_matching(recognition_result)
                else:
                    print('no infer type')
                    break
                image = cv2.imread(img_path)
                bbox_image = self.vis.draw_bboxes(image, result)
                self.data_folder_check(img)
                save_img_path = os.path.join(self.save_path, img)
                cv2.imwrite(save_img_path, bbox_image)
            except:
                    image = cv2.imread(img_path)
                    self.data_folder_check(img)
                    save_img_path = os.path.join(self.save_path, img)
                    cv2.imwrite(save_img_path, image)
                    print('\n{0} is not face'.format(img_path))

        print(("total infer time: {0:03f}".format(self.infer_time)))
        print(("1 image infer time: {0:03f}".format(self.infer_time / len(img_list))))

    def inference_by_video(self):
        capture = cv2.VideoCapture(self.media_path)
        length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        frame_number = 1

        media_list = self.media_path.split('/')
        mdeia_name = media_list[-1]
        save_media_name = mdeia_name.replace(".mp4", "_result.mp4")
        save_mdeia_path = os.path.join(self.save_path, save_media_name)
        video_writer = cv2.VideoWriter(save_mdeia_path, fourcc, int(fps), (width, height))

        print("video info:")
        print(f"\tvideo path:  {self.media_path}")
        print(f"\tvideo fps:   {fps}")
        print(f"\tlength : {length}")
        print(f"\tresolution : {width} * {height}")

        while True:
            ret, frame = capture.read()
            if not ret:
                break
            try:
                if self.infer_type == 'detection':
                    infer_result = self.face_detection(frame)
                    result = self.detection_matching(infer_result)
                elif self.infer_type == 'recognition':
                    recognition_result = self.face_recognition(frame)
                    result = self.recognition_matching(recognition_result)
                else:
                    print('no infer type')
                    break
                bbox_image = self.vis.draw_bboxes(frame, result)
                video_writer.write(bbox_image)
            except:
                video_writer.write(frame)
                print('\n{0} is not face'.format(frame_number))

            print("\r video inference : {0}/{1}\t".format(frame_number, length), end="")
            frame_number += 1
            if self.infer_type == 'recognition':print('\n')

        capture.release()
        video_writer.release()
        print(("\ntotal infer time: {0:03f}".format(self.infer_time)))
        print(("1 frame infer time: {0:03f}".format(self.infer_time / length)))

    def main(self):
        if self.data_type == 'image':
            self.inference_by_image()
        elif self.data_type == 'db':
            self.db_create()
        else:
            self.inference_by_video()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", default='ArcFace', type=str, help="face recognition model")
    parser.add_argument("--backend", default='retinaface', type=str, help="face detection backends")
    parser.add_argument("--infer_type", default="detection", type=str, help="recognition or detection")
    parser.add_argument("--data_type", default="video", type=str, help="image or video or db")
    parser.add_argument("--media_path", default="/workspace/data/video/test_video_2.mp4", type=str, help="media path")
    parser.add_argument("--db_path", default="/workspace/data/origin_face_ver2", type=str, help="db path")
    parser.add_argument("--save_path", default="/workspace/results/test_video/", type=str, help="save result path")

    option = parser.parse_known_args()[0]

    face = Face(option)
    start_total_time = time.time()
    face.main()
    end_total_time = time.time()
    total_process_time = end_total_time - start_total_time
    print(("total process time: {0:03f}".format(total_process_time)))