from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

register_all_modules()

config_file = 'C:/face_recognition_app/mmpose/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
checkpoint_file = 'C:/face_recognition_app/mmpose/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
model = init_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

# please prepare an image with person
results = inference_topdown(model, 'C:/face_recognition_app/demo.jpg')
import os
print(os.path.exists(config_file))  # Should print True
print(os.path.exists(checkpoint_file))  # Should print True
print(os.path.exists('C:/face_recognition_app/demo.jpg'))  # Should print True
print(results)
