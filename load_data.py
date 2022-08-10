import cv2, imageio, torch, os, numpy as np
from torch.utils.data import Dataset

def extract_bayer_channels(raw):
    # Reshape the input bayer image
    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255) # convert to float32 and normalize

    return RAW_norm


class LoadData(Dataset):
    def __init__(self, dataset_dir, dataset_size, test = False):
        if test:
            self.raw_dir = os.path.join(dataset_dir, 'test', 'raw')
            self.dslr_dir = os.path.join(dataset_dir, 'test', 'dslr')
            self.dataset_size = dataset_size
        else:
            self.raw_dir = os.path.join(dataset_dir, 'train', 'raw')
            self.dslr_dir = os.path.join(dataset_dir, 'train', 'dslr')

        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        raw_image = imageio.v2.imread(os.path.join(self.raw_dir, str(idx) + '.png')) # loading the image. needs imageio since cv2 doesn't read Bayer images
        raw_image = np.asarray(raw_image, dtype = np.float32) # convert into numpy array
        raw_image = extract_bayer_channels(raw_image) # reshape Bayer image
        raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1))) # convert to torch tensor

        dslr_image = cv2.imread(os.path.join(self.dslr_dir, str(idx) + ".jpg")) # loading the image. here we can use cv2 since this is an RGB image
        dslr_image = np.asarray(cv2.cvtColor(dslr_image, cv2.COLOR_BGR2RGB), dtype = np.float32) / 255.0 # convert into numpy array
        dslr_image = torch.from_numpy(dslr_image.transpose((2, 0, 1))) # convert to tensor

        return raw_image, dslr_image


class LoadVisualData(Dataset):
    def __init__(self, data_dir, size, full_resolution = True):
        self.raw_dir = os.path.join(data_dir, 'full resolution')
        self.dataset_size = size
        self.full_resolution = full_resolution
        self.test_images = os.listdir(self.raw_dir)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        raw_image = imageio.v2.imread(os.path.join(self.raw_dir, self.test_images[idx]))
        raw_image = np.asarray(raw_image, dtype = np.float32)
        raw_image = extract_bayer_channels(raw_image)
        raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))

        return raw_image, idx