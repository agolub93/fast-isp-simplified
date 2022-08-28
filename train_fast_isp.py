import torch, math, numpy as np

from load_data import LoadData
from fast_isp import fast_isp
from helper import normalize_batch, vgg_19
from kornia.color import rgb_to_lab
from torch.utils.data import DataLoader
from torch.optim import Adam

# set seeds manually for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# CONSTANTS #
TRAIN_SIZE = 46839
TEST_SIZE = 1204
dataset_dir = './images/'
# # #

# PARAMETERS #
batch_size = 10
learning_rate = 1e-4
num_train_epochs = 20
start = 0 # set to 0 if training from scratch
vgg_onset = 5 # when to start adding VGG-19 loss to total loss function
use_lab = True
# # #

def train_model():

    # device parameters
    torch.backends.cudnn.deterministic = True # for reproducibility
    device = torch.device("cuda")
    memory = torch.cuda.get_device_properties(device).total_memory

    # creating network and optimizer
    fastisp = fast_isp().to(device)
    optimizer = Adam(params = fastisp.parameters(), lr = learning_rate)

    print(f"CUDA visible devices: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(device)}")
    print(f"Device memory: {(memory / 1e9):.2f} GB", end = "\n\n")

    print(f"Batch size: {batch_size}")
    print("Adam learning rate: {}".format(optimizer.param_groups[0]["lr"]))
    print(f"VGG-19 loss onset: {vgg_onset}")
    print(f"No. of training epochs: {num_train_epochs}")
    print(f"Starting from epoch: {start}")
    print(f"Using LAB color space: {use_lab}", end = "\n\n")

    # creating dataset loaders
    train_dataset = LoadData(dataset_dir, TRAIN_SIZE, test = False)
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0, pin_memory = True, drop_last = True) # pin_memory speeds up data transfer

    test_dataset = LoadData(dataset_dir, TEST_SIZE, test = True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False, num_workers = 0, pin_memory = True, drop_last = False)

    # restoring state
    if start > 0:
        fastisp.load_state_dict(torch.load("models/fast_isp_epoch_" + str(start) + ".pth"), strict = True)

    # losses
    VGG_19 = vgg_19(device)
    L1_loss = torch.nn.L1Loss()
    MSE_loss = torch.nn.MSELoss()

    # training
    for epoch in range(start, start + num_train_epochs):
        torch.cuda.empty_cache()
        train_iter = iter(train_loader)
        for i in range(len(train_loader)):
            optimizer.zero_grad()

            raw, dslr = next(train_iter) # get next pair of images
            raw, dslr = raw.to(device, non_blocking = True), dslr.to(device, non_blocking = True) # non_blocking allows gpu computation and data transfer at the same time

            enhanced = fastisp(raw) # run images through network

            # LAB + L1 Loss
            if use_lab:
                enhanced_lab = rgb_to_lab(enhanced) # convert images to LAB space. empirically works better than standard RGB
                dslr_lab = rgb_to_lab(dslr)
                loss_l1 = L1_loss(enhanced_lab, dslr_lab)
            else:
                loss_l1 = L1_loss(enhanced, dslr)

            # VGG-19 loss
            if epoch >= vgg_onset:
                enhanced_vgg = VGG_19(normalize_batch(enhanced))
                dslr_vgg = VGG_19(normalize_batch(dslr))
                loss_content = MSE_loss(enhanced_vgg, dslr_vgg)

            # total Loss
            if epoch >= vgg_onset:
                total_loss = loss_l1 + loss_content
            else:
                total_loss = loss_l1

            # optimization step
            total_loss.backward()
            optimizer.step()

            if i == 0:
                # save the model that corresponds to the current epoch
                fastisp.eval().cpu()
                torch.save(fastisp.state_dict(), "models/fast_isp_epoch_" + str(epoch) + ".pth")
                fastisp.to(device).train()

                # evaluate the model
                loss_mse_eval = 0
                loss_l1_eval = 0
                loss_psnr_eval = 0
                loss_vgg_eval = 0

                fastisp.eval() # set model to test mode

                with torch.no_grad():
                    test_iter = iter(test_loader)
                    for j in range(len(test_loader)):
                        raw, dslr = next(test_iter)
                        raw, dslr = raw.to(device, non_blocking = True), dslr.to(device, non_blocking = True)

                        enhanced = fastisp(raw)

                        if use_lab:
                            enhanced_lab = rgb_to_lab(enhanced)
                            dslr_lab = rgb_to_lab(dslr)
                            loss_l1_temp = L1_loss(enhanced_lab, dslr_lab).item()
                        else:
                            loss_l1_temp = L1_loss(enhanced, dslr).item()

                        loss_mse_temp = MSE_loss(enhanced, dslr).item()

                        loss_mse_eval += loss_mse_temp
                        loss_l1_eval += loss_l1_temp
                        loss_psnr_eval += 20 * math.log10(1.0 / math.sqrt(loss_mse_temp)) # compute psnr

                        enhanced_vgg_eval = VGG_19(normalize_batch(enhanced)).detach()
                        target_vgg_eval = VGG_19(normalize_batch(dslr)).detach()
                        loss_vgg_eval += MSE_loss(enhanced_vgg_eval, target_vgg_eval).item()

                loss_mse_eval = loss_mse_eval / TEST_SIZE
                loss_l1_eval = loss_l1_eval / TEST_SIZE
                loss_psnr_eval = loss_psnr_eval / TEST_SIZE
                loss_vgg_eval = loss_vgg_eval / TEST_SIZE

                print(f"Epoch: {epoch}, LAB L1 loss: {loss_l1_eval:.2f}, VGG-19 loss: {loss_vgg_eval:.2f}, PSNR: {loss_psnr_eval:.2f}") # print current epoch's results
                fastisp.train() # set model back to training mode


if __name__ == '__main__':
    train_model()

