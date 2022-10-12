import os
import shutil
import cv2
import json
import numpy as np

def compress_video(input_path, output_path):
    cmd_str = 'ffmpeg -i {} {}'.format(input_path, output_path)
    os.system(cmd_str)

def json2dict(json_dict):
    result = {}
    for person_key in json_dict.keys():
        person_dict = json_dict[person_key]
        # print(person_dict.keys())
        for frame_key in person_dict.keys():
            if frame_key not in result.keys():
                result[frame_key] = []
            frame_res = person_dict[frame_key]['keypoints'] + [int(person_key)]
            result[frame_key].append(frame_res)
    frame_idxs = [int(i) for i in result.keys()]
    print('min_frame:{}  max_frame:{}'.format(min(frame_idxs), max(frame_idxs)))
    return result



SKELETON = [
    [1,3],[1,0],[2,4],[2,0],[0,5],[0,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 17

def draw_vis(img, results):
    for result in results:
        result = np.array(result)
        keypoints = result[:-1].reshape((-1, 3))
        keypoints = keypoints[:, :2]
        person_id = int(result[-1])
        # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0, 255, 0),thickness=1)
        xmin, ymin = int(np.min(keypoints[:, 0])), int(np.min(keypoints[:, 1]))
        cv2.putText(img, str(person_id), (xmin, ymin), 0, 0.5, (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        assert keypoints.shape == (NUM_KPTS,2)
        for i in range(len(SKELETON)):
            kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
            x_a, y_a = keypoints[kpt_a][0],keypoints[kpt_a][1]
            x_b, y_b = keypoints[kpt_b][0],keypoints[kpt_b][1] 
            cv2.circle(img, (int(x_a), int(y_a)), 2, CocoColors[i], -1)
            cv2.circle(img, (int(x_b), int(y_b)), 2, CocoColors[i], -1)
            cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 1)
    return img

def main():
    clip_ls = os.listdir(ann_dir)
    for clip in clip_ls:
        clip_name = clip.split('_alphapose')[0]
        print('>>>>> precessing:{}'.format(clip_name))
        ann_path = os.path.join(ann_dir, clip)
        with open(ann_path, 'r') as f:
            ann_json = json.load(f)
        ann_dict = json2dict(ann_json)
        # img_dir = os.path.join(data_dir, clip_name)
        write_fps = 24
        video_save_path = os.path.join(vis_dir, clip_name + '.avi')
        compress_video_save_path = os.path.join(vis_dir, clip_name + '.mp4')
        one_img_name = os.listdir(os.path.join(data_dir, clip_name))[0]
        img_postfix = '.' + one_img_name.split('.')[-1]
        one_img= cv2.imread(os.path.join(data_dir, clip_name, one_img_name))
        height, width = one_img.shape[:2]
        writer = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), write_fps, (width, height))
        for frame_name in sorted(ann_dict.keys()):
            frame_path = os.path.join(data_dir, clip_name, frame_name + img_postfix)
            frame_img = cv2.imread(frame_path)
            ann = ann_dict[frame_name]
            frame_img = draw_vis(frame_img, ann)
            writer.write(frame_img)
        writer.release()
        compress_video(video_save_path, compress_video_save_path)
        os.remove(video_save_path)


# if __name__ == '__main__':
#     data_dir = '/home/yaboliu/data/cvae/shanghaitech/testing/frames'
#     ann_dir = '/home/yaboliu/work/research/gepc/gepc/data/pose/testing/tracked_person'
#     # ann_dir = '/home/yaboliu/data/cvae/shanghaitech/testing/pose/json_results'
#     vis_dir = '/home/yaboliu/data/cvae/shanghaitech/testing/vis_release'
#     if os.path.exists(vis_dir):
#         shutil.rmtree(vis_dir)
#     os.mkdir(vis_dir)
#     main()

# if __name__ == '__main__':
#     data_dir = '/home/yaboliu/data/cvae/avenue/testing/frames'
#     ann_dir = '/home/yaboliu/data/cvae/avenue/testing/pose_refine1_5/json_results'
#     # ann_dir = '/home/yaboliu/data/cvae/shanghaitech/testing/pose/json_results'
#     vis_dir = '/home/yaboliu/data/cvae/avenue/testing/vis_release_refine1_5'
#     if os.path.exists(vis_dir):
#         shutil.rmtree(vis_dir)
#     os.mkdir(vis_dir)
#     main()

if __name__ == '__main__':
    data_dir = '/home/yaboliu/data/cvae/ucsd/UCSDped2/Test/frames'
    ann_dir = '/home/yaboliu/data/cvae/ucsd/UCSDped2/Test/pose_refined/json_results'
    vis_dir = '/home/yaboliu/data/cvae/ucsd/UCSDped2/Test/vis_release_refined'
    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir)
    os.mkdir(vis_dir)
    main()