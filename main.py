import os
import sys
sys.path.insert(1, './first-order-model/')
from demo import load_checkpoints, make_animation
from skimage import img_as_ubyte
from skimage.transform import resize
import imageio

simage = input("Source Image: ")
simage = './media/'+simage
source_image = imageio.imread(simage)
dvideo = input("Driving Video: ")
dvideo = './media/'+dvideo
driving_video = imageio.mimread(dvideo, memtest=False)

#Resize image and video to 256x256
source_image = resize(source_image, (256, 256))[..., :3]
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

generator, kp_detector = load_checkpoints(config_path='./first-order-model/config/vox-256.yaml', 
    checkpoint_path='./media/vox-cpk.pth.tar', cpu=True)

predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True, cpu=True)

#save resulting video
imageio.mimsave('../generated.mp4', [img_as_ubyte(frame) for frame in predictions])
