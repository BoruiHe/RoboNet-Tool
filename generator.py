import torch.nn as nn
import numpy as np
import torch


class Generator_120_160(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator_120_160, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution. Latent vector Z is of size(batch_size, 5, 1, 1)
            nn.ConvTranspose2d(nz, 64*ngf, kernel_size=(3, 4), stride=1, padding=0, bias=True),
            nn.BatchNorm2d(64*ngf),
            nn.LeakyReLU(True),
            # state size. (64*ngf) (batch_size, 512, 3, 4)
            nn.ConvTranspose2d(64*ngf, 32*ngf, kernel_size=(3, 4), stride=1, padding=0, bias=True),
            nn.BatchNorm2d(32*ngf),
            nn.LeakyReLU(True),
            # state size. (32*ngf) (batch_size, 256, 5, 7)
            nn.ConvTranspose2d(32*ngf, 16*ngf, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(16*ngf),
            nn.LeakyReLU(True),
            # state size. (16*ngf) (batch_size, 128, 12, 16)
            nn.ConvTranspose2d(16*ngf, 8*ngf, kernel_size=(3, 4), stride=1, padding=0, bias=True),
            nn.BatchNorm2d(8*ngf),
            nn.LeakyReLU(True),
            # state size. (8*ngf) (batch_size, 64, 14, 19)
            nn.ConvTranspose2d(8*ngf, 4*ngf, kernel_size=(3, 3), stride=2, padding=0, bias=True),
            nn.BatchNorm2d(4*ngf),
            nn.LeakyReLU(True),
            # state size. (4*ngf) (batch_size, 32, 29, 39)
            nn.ConvTranspose2d(4*ngf, 2*ngf, kernel_size=2, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(2*ngf),
            nn.LeakyReLU(True),
            # state size. (2*ngf) (batch_size, 16, 30, 40)
            nn.ConvTranspose2d(2*ngf, ngf, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(True), 
            # state size. (ngf) (batch_size, 8, 60, 80)
            nn.ConvTranspose2d(ngf, nc, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(nc),
            nn.LeakyReLU(True),
            # state size. (nc) (batch_size, 3, 120, 160)
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Generator_60_80(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator_60_80, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution. Latent vector Z is of size(batch_size, 5, 1, 1)
            nn.ConvTranspose2d(nz, 32*ngf, kernel_size=(3, 4), stride=1, padding=0, bias=True),
            nn.BatchNorm2d(32*ngf),
            nn.LeakyReLU(True),
            # state size. (32*ngf) (batch_size, 256, 3, 4)
            nn.ConvTranspose2d(32*ngf, 16*ngf, kernel_size=(3, 4), stride=1, padding=0, bias=True),
            nn.BatchNorm2d(16*ngf),
            nn.LeakyReLU(True),
            # state size. (16*ngf) (batch_size, 128, 5, 7)
            nn.ConvTranspose2d(16*ngf, 8*ngf, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(8*ngf),
            nn.LeakyReLU(True),
            # state size. (8*ngf) (batch_size, 64, 12, 16)
            nn.ConvTranspose2d(8*ngf, 4*ngf, kernel_size=(3, 4), stride=1, padding=0, bias=True),
            nn.BatchNorm2d(4*ngf),
            nn.LeakyReLU(True),
            # state size. (4*ngf) (batch_size, 32, 14, 19)
            nn.ConvTranspose2d(4*ngf, 2*ngf, kernel_size=(3, 3), stride=2, padding=0, bias=True),
            nn.BatchNorm2d(2*ngf),
            nn.LeakyReLU(True),
            # state size. (2ngf) (batch_size, 16, 29, 39)
            nn.ConvTranspose2d(2*ngf, ngf, kernel_size=2, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(True),
            # state size. (ngf) (batch_size, 8, 30, 40)
            nn.ConvTranspose2d(ngf, nc, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(nc),
            nn.LeakyReLU(True),
            # state size. (nc) (batch_size, 3, 60, 80)
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Generator_30_40(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator_30_40, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution. Latent vector Z is of size(batch_size, 5, 1, 1)
            nn.ConvTranspose2d(nz, ngf * 16, kernel_size=(3, 4), stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(True),
            # state size. (ngf*8) x 3 x 4 (batch_size, 512, 3, 4)
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=(3, 4), stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(True),
            # state size. (ngf*4) x 16 x 12 (batch_size, 256, 5, 7)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(True),
            # state size. (ngf*2) x 80 x 60 (batch_size, 128, 12, 16)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=(3, 4), stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(True),
            # state size. (ngf) x 80 x 60 (batch_size, 64, 14, 19)
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=(3, 3), stride=2, padding=0, bias=True),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(True),
            # state size. (ngf/2) x 80 x 60 (batch_size, 32, 29, 39)
            nn.ConvTranspose2d(ngf, nc, kernel_size=2, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(nc),
            nn.LeakyReLU(True),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution. Latent vector Z is of size(batch_size, 5, 1, 1)
            nn.ConvTranspose2d(nz, ngf * 64, kernel_size=(3, 4), stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ngf * 64),
            nn.LeakyReLU(True),
            # state size. (ngf*8) x 3 x 4 (batch_size, 512, 3, 4)
            nn.ConvTranspose2d(ngf * 64, ngf * 32, kernel_size=(3, 4), stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ngf * 32),
            nn.LeakyReLU(True),
            # state size. (ngf*4) x 16 x 12 (batch_size, 256, 5, 7)
            nn.ConvTranspose2d(ngf * 32, ngf * 16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(True),
            # state size. (ngf*2) x 80 x 60 (batch_size, 128, 12, 16)
            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=(3, 4), stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(True),
            # state size. (ngf) x 80 x 60 (batch_size, 64, 14, 19)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=(3, 3), stride=2, padding=0, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(True),
            # state size. (ngf/2) x 80 x 60 (batch_size, 32, 29, 39)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=(3, 3), stride=2, padding=0, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(True),
            # state size. (ngf/4) x 80 x 60 (batch_size, 16, 59, 79)
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=(3, 3), stride=2, padding=0, bias=True),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(True),
            # state size. (ngf/8) x 80 x 60 (batch_size, 8, 119, 159)
            nn.ConvTranspose2d(ngf, nc, kernel_size=(4, 4), stride=2, padding=0, bias=True),
            nn.BatchNorm2d(nc),
            nn.LeakyReLU(True),
            # state size. (nc) x 240 x 320
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Generator_toy(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator_toy, self).__init__()

        self.CT2d1 = nn.ConvTranspose2d(nz, 64*ngf, kernel_size=(3, 4), stride=1, padding=0, bias=True)
        self.batch2d1 = nn.BatchNorm2d(64*ngf)
        self.Lrelu1 = nn.LeakyReLU(True)

        self.CT2d2 = nn.ConvTranspose2d(64*ngf, 32*ngf, kernel_size=(3, 4), stride=1, padding=0, bias=True)
        self.batch2d2 = nn.BatchNorm2d(32*ngf)
        self.Lrelu2 = nn.LeakyReLU(True)

        self.CT2d3 = nn.ConvTranspose2d(32*ngf, 16*ngf, kernel_size=4, stride=2, padding=0, bias=True)
        self.batch2d3 = nn.BatchNorm2d(16*ngf)
        self.Lrelu3 = nn.LeakyReLU(True)

        self.CT2d4 = nn.ConvTranspose2d(16*ngf, 8*ngf, kernel_size=(3, 4), stride=1, padding=0, bias=True)
        self.batch2d4 = nn.BatchNorm2d(8*ngf)
        self.Lrelu4 = nn.LeakyReLU(True)

        self.CT2d5 = nn.ConvTranspose2d(8*ngf, 4*ngf, kernel_size=(3, 3), stride=2, padding=0, bias=True)
        self.batch2d5 = nn.BatchNorm2d(4*ngf)
        self.Lrelu5 = nn.LeakyReLU(True)

        self.CT2d6 = nn.ConvTranspose2d(4*ngf, 2*ngf, kernel_size=2, stride=1, padding=0, bias=True)
        self.batch2d6 = nn.BatchNorm2d(2*ngf)
        self.Lrelu6 = nn.LeakyReLU(True)

        self.CT2d7 = nn.ConvTranspose2d(2*ngf, ngf, kernel_size=2, stride=2, padding=0, bias=True)
        self.batch2d7 = nn.BatchNorm2d(ngf)
        self.Lrelu7 = nn.LeakyReLU(True)

        self.CT2d8 = nn.ConvTranspose2d(ngf, nc, kernel_size=2, stride=2, padding=0, bias=True)
        self.batch2d8 = nn.BatchNorm2d(nc)
        self.Lrelu8 = nn.LeakyReLU(True)

        # self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.CT2d1(input)
        output = self.batch2d1(output)
        output = self.Lrelu1(output)

        output = self.CT2d2(output)
        output = self.batch2d2(output)
        output = self.Lrelu2(output)

        output = self.CT2d3(output)
        output = self.batch2d3(output)
        output = self.Lrelu3(output)

        output = self.CT2d4(output)
        output = self.batch2d4(output)
        output = self.Lrelu4(output)

        output = self.CT2d5(output)
        output = self.batch2d5(output)
        output = self.Lrelu5(output)

        output = self.CT2d6(output)
        output = self.batch2d6(output)
        output = self.Lrelu6(output)

        output = self.CT2d7(output)
        output = self.batch2d7(output)
        output = self.Lrelu7(output)

        output = self.CT2d8(output)
        output = self.batch2d8(output)
        output = self.Lrelu8(output)

        # output = self.tanh(output) 

        return output
        