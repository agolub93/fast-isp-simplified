import os, torch, cv2, numpy as np

from torch.utils.data import DataLoader
from fast_isp import fast_isp
from load_data import LoadVisualData
from kornia.utils import tensor_to_image

# PARAMETERS #
restore_epoch = 20
test_size = 5
dataset_dir = "./images/"
# # #

def test_model():
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    # creating dataset loaders
    visual_dataset = LoadVisualData(dataset_dir, test_size)
    visual_loader = DataLoader(dataset = visual_dataset, batch_size = 1, shuffle = False, num_workers = 0, pin_memory = True, drop_last = False)

    # creating and loading pretrained model
    fastisp = fast_isp().to(device)
    fastisp.load_state_dict(torch.load("models/fast_isp_epoch_" + str(restore_epoch - 1) + ".pth"), strict=True)
    fastisp.eval() # set model to test mode

    # processing full resolution Bayer images
    with torch.no_grad():
        visual_iter = iter(visual_loader)
        for j in range(len(visual_loader)):
            print("Processing image " + str(j) + "...")
            torch.cuda.empty_cache()
            raw_image, idx = next(visual_iter)
            raw_image = raw_image.to(device)

            # test model
            enhanced = fastisp(raw_image)
            enhanced= tensor_to_image(enhanced) * 255.0 # convert to RGB image
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR) # convert to cv2 representation: RGB -> BGR
            cv2.imwrite("results/" + str(idx.item()) + ".png", enhanced) # save image


if __name__ == '__main__':
    test_model()
