import cv2
import json
import numpy as np

img_dir='H:/aiCodeCamp/balloon/val'
json_file = 'H:/aiCodeCamp/balloon/val/via_region_data.json'
with open(json_file,'r') as r:
    json_temp = r.read()

category={
    "id": 1,                                 # int 类别id编号
    "name": "balloon",                     # str 类别名字
    "supercategory": "None"
    }
output_json={}
output_json['images']=[]
output_json['annotations']=[]
output_json['categories']=[category]
labels_dict = json.loads(json_temp)
n=0
m=0
for k in labels_dict.keys():
    image_temp={}
    annotations_temp={}
    
    image_temp['id']=n
    image_temp['file_name']=labels_dict[k]['filename']
    print(img_dir+'/'+labels_dict[k]['filename'])
    im = cv2.imread(img_dir+'/'+labels_dict[k]['filename'])
    image_temp['height']=im.shape[0]
    image_temp['width']=im.shape[1]
    output_json['images'].append(image_temp)
    for region_id in labels_dict[k]['regions']:
        seg_temp=[]
        vertices=[]
        annotations_temp['id']=m
        annotations_temp['image_id']=n
        annotations_temp['category_id']=category['id']
        annotations_temp["iscrowd"]=0 
        annotations_temp["segmentation"] = []
        for l in range(len(labels_dict[k]['regions'][region_id]["shape_attributes"]["all_points_x"])):
            seg_temp.append(labels_dict[k]['regions'][region_id]["shape_attributes"]["all_points_x"][l])
            seg_temp.append(labels_dict[k]['regions'][region_id]["shape_attributes"]["all_points_y"][l])
            vertices.append([labels_dict[k]['regions'][region_id]["shape_attributes"]["all_points_x"][l],labels_dict[k]['regions'][region_id]["shape_attributes"]["all_points_y"][l]])
        annotations_temp["segmentation"].append(seg_temp)
        annotations_temp['bbox'] = [min(labels_dict[k]['regions'][region_id]["shape_attributes"]["all_points_x"]),
                                    min(labels_dict[k]['regions'][region_id]["shape_attributes"]["all_points_y"]),
                                    max(labels_dict[k]['regions'][region_id]["shape_attributes"]["all_points_x"])-min(labels_dict[k]['regions'][region_id]["shape_attributes"]["all_points_x"]),
                                    max(labels_dict[k]['regions'][region_id]["shape_attributes"]["all_points_y"])-min(labels_dict[k]['regions'][region_id]["shape_attributes"]["all_points_y"]),
                                    ]
        vertices = np.array(vertices)
        annotations_temp["area"] = area = cv2.contourArea(vertices)
        m+=1
    output_json["annotations"].append(annotations_temp)

    n+=1
print(output_json)
with open('H:/aiCodeCamp/balloon/val/via_region_data_coco.json','w') as w:
    w.write(json.dumps(output_json))
