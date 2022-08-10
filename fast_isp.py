import torch
import torch.nn as nn

# simple CNN for converting Bayer images to RGB. replaces the full classical ISP process.
# uses LAB space loss and pixelshuffle for improved results.
class fast_isp(nn.Module):
    def __init__(self):
        super(fast_isp, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = 4, out_channels = 16, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Hardtanh(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Hardtanh(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels = 16, out_channels = 12, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Hardtanh(),
            nn.PixelShuffle(upscale_factor = 2)
        )

    def forward(self, input):
        return self.layers(input)