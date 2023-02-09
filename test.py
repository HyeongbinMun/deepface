import os
from deepface import DeepFace

if __name__ == '__main__':
    models = [
        "VGG-Face",
        "Facenet",
        "Facenet512",
        "OpenFace",
        "DeepFace",
        "DeepID",
        "ArcFace",
        "Dlib",
        "SFace",
    ]

    backends = [
      'opencv',
      'ssd',
      'dlib',
      'mtcnn',
      'retinaface',
      'mediapipe'
    ]
    img_root = '/hdd/face/'
    img_path = os.path.join(img_root, 'sample/sample_4.jpg')

    # #face verification
    # obj = DeepFace.verify(img1_path=img_path + "sample/sample_1.jpg",
    #         img2_path=img_path + "sample/sample_2.jpg",
    #         detector_backend = backends[0]
    # )

    # face recognition
    # dfs = DeepFace.find(img_path=img_path + 'train/ChoGuesung/0001.png',
    #                     db_path=img_path + 'origin/',
    #                     detector_backend=backends[5],
    #                     model_name=models[6]
    #                     )

    # #embeddings
    # embedding_objs = DeepFace.represent(img_path = "img.jpg",
    #         detector_backend = backends[2]
    # )
    #
    # #facial analysis
    # demographies = DeepFace.analyze(img_path = "img4.jpg",
    #         detector_backend = backends[3]
    # )
    #
    #face detection and alignment
    print(img_path)
    face_objs = DeepFace.extract_faces(img_path=img_path,
            target_size=(224, 224),
            detector_backend=backends[3]
    )
    for el in face_objs:
        print(el['facial_area'])