import torch
from torchsummary import summary

import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_chan, out_chan, stride=1):
    return nn.Sequential(
        nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1, stride=stride),
        nn.BatchNorm3d(out_chan),
        nn.ReLU(inplace=True)
    )


def conv_stage(in_chan, out_chan):
    return nn.Sequential(
        conv_block(in_chan, out_chan),
        conv_block(out_chan, out_chan),
    )


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc1 = conv_stage(1, 16)
        self.enc2 = conv_stage(16, 32)
        self.enc3 = conv_stage(32, 64)
        self.enc4 = conv_stage(64, 128)
        self.enc5 = conv_stage(128, 128)
        self.pool = nn.MaxPool3d(2, 2)

        self.dec4 = conv_stage(256, 64)
        self.dec3 = conv_stage(128, 32)
        self.dec2 = conv_stage(192, 16)
        self.dec1 = conv_stage(224, 16)
        self.conv_out = nn.Conv3d(336, 1, 1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))

        dec4 = self.dec4(torch.cat((enc4, F.interpolate(enc5, enc4.size()[2:], mode='trilinear', align_corners=False)), 1))


        dec3_1 = self.dec3(torch.cat((enc3, F.interpolate(dec4, enc3.size()[2:], mode='trilinear', align_corners=False)), 1))
        enc5dec3 = F.interpolate(enc5, enc3.size()[2:], mode='trilinear', align_corners=False)
        dec3 = torch.cat((enc5dec3, dec3_1), 1)



        dec2_1 = self.dec2(torch.cat((enc2, F.interpolate(dec3, enc2.size()[2:], mode='trilinear', align_corners=False)), 1))
        enc5dec2 = F.interpolate(enc5, enc2.size()[2:], mode='trilinear', align_corners=False)
        dec4dec2 = F.interpolate(dec4, enc2.size()[2:], mode='trilinear', align_corners=False)
        dec2 = torch.cat((enc5dec2,dec4dec2, dec2_1), 1)


        dec1_1 = self.dec1(torch.cat((enc1, F.interpolate(dec2, enc1.size()[2:], mode='trilinear', align_corners=False)), 1))
        enc5dec1 = F.interpolate(enc5, enc1.size()[2:], mode='trilinear', align_corners=False)
        enc4dec1 = F.interpolate(enc4, enc1.size()[2:], mode='trilinear', align_corners=False)
        enc3dec1 = F.interpolate(enc3, enc1.size()[2:], mode='trilinear', align_corners=False)
        dec1 = torch.cat((enc5dec1, enc4dec1, enc3dec1, dec1_1), 1)

        out = self.conv_out(dec1)
        out = torch.sigmoid(out)
        return out


if __name__ == '__main__':
    import time
    import torch
    from torch.autograd import Variable

    torch.cuda.set_device(0)
    net = UNet().cuda().eval()
    # net = UNet().eval()

    # data = Variable(torch.randn(1, 64, 64, 155))
    data = Variable(torch.randn(1, 1, 64, 64, 155)).cuda()

    start_time = time.time()
    print("go")
    for _ in range(5):
        _ = net(data)
    print((time.time() - start_time)/5)
    summary(net, ( 1, 64, 64, 64) )

    print(f'torch version: {torch.__version__}')

