import numpy as np

keypoints = {
    "mpii": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
    "coco_wholebody": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    }
}

palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255],
                    [255, 0, 0], [255, 255, 255]])


body = {
    "coco_wholebody": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
}
foot = {
    "coco_wholebody": [[17, 20], [18, 21], [19, 22]]
}

face = {
    "coco_wholebody": [[23, 39], [24, 38], [25, 37], [26, 36], [27, 35], [28, 34],
            [29, 33], [30, 32], [40, 49], [41, 48], [42, 47], [43, 46],
            [44, 45], [54, 58], [55, 57], [59, 68], [60, 67], [61, 66],
            [62, 65], [63, 70], [64, 69], [71, 77], [72, 76], [73, 75],
            [78, 82], [79, 81], [83, 87], [84, 86], [88, 90]]
}

hand = {
    "coco_wholebody": [[91, 112], [92, 113], [93, 114], [94, 115], [95, 116],
            [96, 117], [97, 118], [98, 119], [99, 120], [100, 121],
            [101, 122], [102, 123], [103, 124], [104, 125], [105, 126],
            [106, 127], [107, 128], [108, 129], [109, 130], [110, 131],
            [111, 132]]
}
flip_pairs = {"coco_wholebody": body["coco_wholebody"] + foot["coco_wholebody"] + face["coco_wholebody"] + hand["coco_wholebody"]}

skeleton = {
    "mpii": [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
        [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]
    ],
    "coco_wholebody": [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
        [8, 10], [1, 2], [0, 1], [0, 2],
        [1, 3], [2, 4], [3, 5], [4, 6], [15, 17], [15, 18],
        [15, 19], [16, 20], [16, 21], [16, 22], [91, 92],
        [92, 93], [93, 94], [94, 95], [91, 96], [96, 97],
        [97, 98], [98, 99], [91, 100], [100, 101], [101, 102],
        [102, 103], [91, 104], [104, 105], [105, 106],
        [106, 107], [91, 108], [108, 109], [109, 110],
        [110, 111], [112, 113], [113, 114], [114, 115],
        [115, 116], [112, 117], [117, 118], [118, 119],
        [119, 120], [112, 121], [121, 122], [122, 123],
        [123, 124], [112, 125], [125, 126], [126, 127],
        [127, 128], [112, 129], [129, 130], [130, 131],
        [131, 132]
    ]
}

pose_link_color = {"coco_wholebody": palette[[0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]
                        + [16, 16, 16, 16, 16, 16]
                        + [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16, 16]
                        + [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16, 16]]
                   }
pose_kpt_color = {"coco_wholebody": palette[[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0] + [19] * (68 + 42)]}