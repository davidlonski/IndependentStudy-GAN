import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, image_size=64):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.image_size = image_size
        
        # Initialize the appropriate generator based on image size
        if image_size == 64:
            self.main = self._64x64_generator()
        elif image_size == 128:
            self.main = self._128x128_generator()
        elif image_size == 256:
            self.main = self._256x256_generator()
        else:
            raise ValueError(f"Unsupported image size: {image_size}. Supported sizes are 64, 128, and 256.")

    def _64x64_generator(self):
        return nn.Sequential(
            # Encoder
            nn.Conv2d(self.nc + self.nz, self.ngf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ngf) x 32 x 32
            
            nn.Conv2d(self.ngf, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ngf*2) x 16 x 16
            
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ngf*4) x 8 x 8
            
            nn.Conv2d(self.ngf * 4, self.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ngf*8) x 4 x 4
            
            # Decoder
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size: (ngf*4) x 8 x 8
            
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size: (ngf*2) x 16 x 16
            
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size: (ngf) x 32 x 32
            
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: (nc) x 64 x 64
        )

    def _128x128_generator(self):
        return nn.Sequential(
            # Encoder
            nn.Conv2d(self.nc + self.nz, self.ngf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ngf) x 64 x 64
            
            nn.Conv2d(self.ngf, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ngf*2) x 32 x 32
            
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ngf*4) x 16 x 16
            
            nn.Conv2d(self.ngf * 4, self.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ngf*8) x 8 x 8
            
            nn.Conv2d(self.ngf * 8, self.ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ngf*16) x 4 x 4
            
            # Decoder
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size: (ngf*8) x 8 x 8
            
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size: (ngf*4) x 16 x 16
            
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size: (ngf*2) x 32 x 32
            
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size: (ngf) x 64 x 64
            
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: (nc) x 128 x 128
        )

    def _256x256_generator(self):
        return nn.Sequential(
            # Encoder
            nn.Conv2d(self.nc + self.nz, self.ngf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ngf) x 128 x 128
            
            nn.Conv2d(self.ngf, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ngf*2) x 64 x 64
            
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ngf*4) x 32 x 32
            
            nn.Conv2d(self.ngf * 4, self.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ngf*8) x 16 x 16
            
            nn.Conv2d(self.ngf * 8, self.ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ngf*16) x 8 x 8
            
            nn.Conv2d(self.ngf * 16, self.ngf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ngf*32) x 4 x 4
            
            # Decoder
            nn.ConvTranspose2d(self.ngf * 32, self.ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 16),
            nn.ReLU(True),
            # state size: (ngf*16) x 8 x 8
            
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size: (ngf*8) x 16 x 16
            
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size: (ngf*4) x 32 x 32
            
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size: (ngf*2) x 64 x 64
            
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size: (ngf) x 128 x 128
            
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: (nc) x 256 x 256
        )

    def forward(self, input, noise):
        # Concatenate input image with noise
        x = torch.cat([input, noise], 1)
        return self.main(x) 