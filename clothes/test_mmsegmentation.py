from matplotlib import pyplot as plt
from mmseg.apis import init_segmentor,inference_segmentor,show_result_pyplot
config_file='deeplabv3plus_clothes_ade20k.py'
checkpoint_file='iter_40000.pth'
input_file = 'H:/aiCodeCamp/clothes/data/img/val/img_0007.jpg'
model = init_segmentor(config_file,checkpoint_file,device="cpu")
result = inference_segmentor(model,input_file)
#plt.imshow(result.pred_seg_data.data[0].detach().cpu().numpy())
#plt.show()
show_result_pyplot(model, input_file, result, opacity=0.8,title="mmseg")
