import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(CoordAttention, self).__init__()
        self.pool_w_avg = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h_avg = nn.AdaptiveAvgPool2d((None, 1))

        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // reduction)
        self.act1 = h_swish()

        self.conv_h = nn.Conv2d(in_channels // reduction, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(in_channels // reduction, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        s, c, H, W = x.shape
        x_h = self.pool_h_avg(x)
        x_w = self.pool_w_avg(x).permute(0, 1, 3, 2)

        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))

        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        x_h = self.conv_h(x_h)
        x_h = x_h.reshape(x_h.shape[0], 4, 4, x_h.shape[2], x_h.shape[3])
        h_new = torch.mean(x_h, dim=2).unsqueeze(2)
        h_new = h_new.expand(x_h.shape[0], 4, 4, x_h.shape[3], x_h.shape[4]).reshape(x_h.shape[0], 4 * 4, x_h.shape[3],
                                                                                                 x_h.shape[4])
        out_h = torch.sigmoid(h_new)

        x_w = self.conv_w(x_w)
        x_w = x_w.reshape(x_w.shape[0], 4, 4, x_w.shape[2], x_w.shape[3])
        w_new = torch.mean(x_w, dim=2).unsqueeze(2)
        w_new = w_new.expand(x_w.shape[0], 4, 4, x_w.shape[3], x_w.shape[4]).reshape(x_w.shape[0], 4 * 4, x_w.shape[3],
                                                                                           x_w.shape[4])
        out_w = torch.sigmoid(w_new)
        out = identity * out_w * out_h
        return out

class MSFAA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSFAA, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.CA = CoordAttention(in_channels=in_channels, out_channels=in_channels)

    def forward(self, x):
        identity = x
        out = self.CA(x)
        out += identity
        out = self.relu(out)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class s_ex(nn.Module):
    def __init__(self, c_div=8):
        super(s_ex, self).__init__()
        self.c_div = c_div

    def forward(self, x):
        nt, c, h, w = x.size()
        cnum = c // self.c_div
        out = torch.zeros_like(x)
        out[:-1, :cnum] = x[1:, :cnum]
        out[1:, cnum: 2 * cnum] = x[:-1, cnum: 2 * cnum]
        out[:, 2 * cnum:] = x[:, 2 * cnum:]
        return out

class SFS(nn.Module):
    def __init__(self, s_top=16, c_div=8):
        super(SFS, self).__init__()
        self.k_size = 3
        self.s_top = s_top
        self.c_div = c_div
        self.linear = nn.Linear(384, 1, bias=False)
        self.sl = s_ex(c_div=c_div)
        self.sg = nn.Sequential(
            nn.Linear(self.s_top, self.s_top * 2, bias=False),
            nn.BatchNorm1d(self.s_top * 2), nn.ReLU(inplace=True),
            nn.Linear(self.s_top * 2, self.k_size, bias=False), nn.Softmax(-1))

    def s_sort(self, xg, s):
        xgs = self.linear(xg).squeeze()
        index_sort = torch.sort(xgs, 0, descending=True)[1]
        xgex = xg.permute(1, 0)
        if self.s_top > s:
            slss = xgex[:, :s]
            inc_sls = torch.zeros(slss.shape[0], self.s_top-s, requires_grad=True).cuda(0)
            sls = torch.cat([slss, inc_sls], dim=1)
        else:
            sls = xgex[:, :self.s_top]
            index_top = torch.sort(index_sort[:self.s_top], 0)[0]
            for s in range(index_top.shape[0]):
                tops = index_top[s].item()
                sls[:, s] = xgex[:, tops]
        return sls

    def forward(self, x):
        s, c, h, w = x.size()
        xg = F.adaptive_avg_pool2d(x, (1, 1)).view(s, -1)
        sls = self.s_sort(xg, s)
        convk = self.sg(sls).view(c, 1, -1, 1)
        xl = self.sl(x)
        out = F.conv2d(xl.permute(1, 0, 2, 3).contiguous().view(1, c, s, h * w),
                       convk,
                       bias=None,
                       stride=(1, 1),
                       padding=(1, 0),
                       groups=c)
        out = out.view(c, s, h, w)
        out = out.permute(1, 0, 2, 3)
        return out


class SSAMNet(nn.Module):
    def __init__(self, cla_num=1, s_top=16, c_div=8):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.features0 = nn.Sequential(*list(self.model.features.children())[:3])
        self.features1 = nn.Sequential(*list(self.model.features.children())[3:6])
        self.features2 = nn.Sequential(*list(self.model.features.children())[6:8])
        self.features3 = nn.Sequential(*list(self.model.features.children())[8:10])
        self.features4 = nn.Sequential(*list(self.model.features.children())[10:])

        self.sfs = SFS(s_top=s_top, c_div=c_div)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.ms0 = nn.Conv2d(64, 4, kernel_size=1, stride=1, padding=0)
        self.ms1 = nn.Conv2d(192, 4, kernel_size=1, stride=1, padding=0)
        self.ms2 = nn.Conv2d(384, 4, kernel_size=1, stride=1, padding=0)
        self.ms4 = nn.Conv2d(256, 4, kernel_size=1, stride=1, padding=0)

        self.msca = MSFAA(in_channels=4 * 4, out_channels=4)
        self.classifier = nn.Linear(4 * 6 * 6, cla_num)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        xf0 = self.features0(x)
        xf1 = self.features1(xf0)
        xf2 = self.features2(xf1)
        xfs2 = self.sfs(xf2)
        xr2 = xf2 + xfs2

        xf3 = self.features3(xr2)
        xf4 = self.features4(xf3)

        ms0 = F.interpolate(self.ms0(xf0), size=7, mode='bilinear', align_corners=False)
        ms1 = F.interpolate(self.ms1(xf1), size=7, mode='bilinear', align_corners=False)
        ms2 = F.interpolate(self.ms2(xr2), size=7, mode='bilinear', align_corners=False)
        ms4 = self.ms4(xf4)
        ms_cat = torch.cat([ms0, ms1, ms2, ms4], dim=1)
        ms_out = self.msca(ms_cat)
        out = self.avgpool(ms_out)
        out = torch.flatten(out, 1)
        out = torch.max(out, 0, keepdim=True)[0]
        out = self.classifier(out)
        return out

