'''
Write inference video based on target dataset
'''
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # where to save video
    save_root = f"{opt.video_folder}/source_{opt.name}_target_{opt.target_name}.mp4"

    # codec
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = None
    
    if opt.eval:
        model.eval() # did not notice a difference with or without eval

    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results

        # get the full image
        fake_full_image = visuals['fake_full_image'][0]

        # initiate the video writer if it is not initiated yet
        if video is None:
            height, width = int(fake_full_image.shape[1]), int(fake_full_image.shape[2]) 
            video = cv2.VideoWriter(save_root, fourcc, 25.0, (width, height))

        # process the frame
        fake_full_image = torch.clamp(fake_full_image, 0., 1.0)
        fake_full_image = cv2.cvtColor(np.array(transforms.ToPILImage()(fake_full_image)), cv2.COLOR_RGB2BGR)

        video.write(fake_full_image)
    video.release()