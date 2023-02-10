from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import cv2
import matplotlib.pyplot as plt

config_file = 'G:/github/OpenMMLabCamp/balloon/mask_rcnn_r50_fpn_2x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'G:/github/OpenMMLabCamp/balloon/epoch_14.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')
img_file = 'H:/aiCodeCamp/balloon/val/5603212091_2dfe16ea72_b.jpg'
result = inference_detector(model, img_file)
img = cv2.imread(img_file)
print(len(result[1][0][0]))
print(result[0][0])
result_temp = result[1][0][0][:]
for x in range(len(result_temp)):
    for y in range(len(result_temp[x])):
        if not result_temp[x][y]:
            result_temp[x][y]=0;
        else:
            result_temp[x][y]=255
#cv2.imwrite('test.jpg',result_temp)
plt.imshow(result_temp)
plt.show()
show_result_pyplot(model, img_file, result)

