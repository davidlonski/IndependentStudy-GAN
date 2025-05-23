import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc, image_size=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.nc = nc
        self.image_size = image_size
        
        # Initialize the appropriate discriminator based on image size
        if image_size == 64:
            self.main = self._64x64_discriminator()
        elif image_size == 128:
            self.main = self._128x128_discriminator()
        elif image_size == 256:
            self.main = self._256x256_discriminator()
        else:
            raise ValueError(f"Unsupported image size: {image_size}. Supported sizes are 64, 128, and 256.")

    def _64x64_discriminator(self):
        return nn.Sequential(
            # input is (nc*2) x 64 x 64 (concatenated input and target)
            nn.Conv2d(self.nc * 2, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf) x 32 x 32
            
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*2) x 16 x 16
            
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*4) x 8 x 8
            
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 4 x 4
            
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def _128x128_discriminator(self):
        return nn.Sequential(
            # input is (nc*2) x 128 x 128 (concatenated input and target)
            nn.Conv2d(self.nc * 2, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf) x 64 x 64
            
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*2) x 32 x 32
            
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*4) x 16 x 16
            
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 8 x 8
            
            nn.Conv2d(self.ndf * 8, self.ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*16) x 4 x 4
            
            nn.Conv2d(self.ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def _256x256_discriminator(self):
        return nn.Sequential(
            # input is (nc*2) x 256 x 256 (concatenated input and target)
            nn.Conv2d(self.nc * 2, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf) x 128 x 128
            
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*2) x 64 x 64
            
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*4) x 32 x 32
            
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 16 x 16
            
            nn.Conv2d(self.ndf * 8, self.ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*16) x 8 x 8
            
            nn.Conv2d(self.ndf * 16, self.ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*32) x 4 x 4
            
            nn.Conv2d(self.ndf * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, target):
        # Concatenate input and target images
        x = torch.cat([input, target], 1)
        return self.main(x) 