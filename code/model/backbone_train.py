import torch
import torch.nn as nn
import torch.nn.functional as F


from .modules.deablock_train import CSPCA, HorizontalCellTrain



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class ProcessingBranch(nn.Module):


    def __init__(self, channels, base_dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(channels, base_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.seq(x)

class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()
        self.height = height
        d = max(int(dim / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, *in_feats):

        in_feats = list(in_feats)


        if len(in_feats) > 0:

            target_h, target_w = in_feats[0].shape[2], in_feats[0].shape[3]


            for i in range(1, len(in_feats)):
                current_h, current_w = in_feats[i].shape[2], in_feats[i].shape[3]
                if current_h != target_h or current_w != target_w:

                    in_feats[i] = F.interpolate(
                        in_feats[i],
                        size=(target_h, target_w),
                        mode='bilinear',
                        align_corners=False
                    )

        B, C, H, W = in_feats[0].shape
        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class OnOffFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        )

        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim * 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, on, off):

        x_cat = torch.cat([on, off], dim=1)


        weights = self.attn(on + off)
        w_on, w_off = torch.split(weights, on.size(1), dim=1)


        out_on = on * w_on
        out_off = off * w_off


        fused = self.fusion_conv(torch.cat([out_on, out_off], dim=1))


        return fused + on











class RADNet(nn.Module):
    def __init__(self, base_dim=32, num_blocks=[4, 4, 8, 4]):
        super(RADNet, self).__init__()
        self.processing_branch = ProcessingBranch(3, base_dim=base_dim)


        self.down1 = nn.Sequential(nn.Conv2d(base_dim, base_dim*2, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(base_dim*2, base_dim*4, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(base_dim * 4, base_dim * 8, kernel_size=3, stride=2, padding=1),
                                   nn.ReLU(True))
        

        self.down_level1_blocks = nn.ModuleList([
            HorizontalCellTrain(base_dim, 3) for _ in range(num_blocks[0])
        ])
        
        # level3-wu
        self.fe_level_3 = nn.Conv2d(in_channels=base_dim * 2, out_channels=base_dim * 2, kernel_size=3, stride=1, padding=1)
        self.level3_blocks = nn.ModuleList([
            HorizontalCellTrain(base_dim * 2, 3) for _ in range(num_blocks[1])
        ])
        
        # level2-shuang
        self.fe_level_2 = nn.Conv2d(in_channels=base_dim * 4, out_channels=base_dim * 4, kernel_size=3, stride=1, padding=1)
        self.down_level2_blocks = nn.ModuleList([
            CSPCA(default_conv, base_dim * 4, 3) for _ in range(num_blocks[2])
        ])

        self.fe_level_4 = nn.Conv2d(in_channels=base_dim * 8, out_channels=base_dim * 8, kernel_size=3, stride=1,
                                    padding=1)
        self.down_level4_blocks = nn.ModuleList([
            CSPCA(default_conv, base_dim * 8, 3) for _ in range(num_blocks[3])
        ])

        # up-sample
        self.up0 = nn.Sequential(nn.ConvTranspose2d(base_dim * 8, base_dim * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base_dim*4, base_dim*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base_dim*2, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(True))
        self.up3 = nn.Sequential(nn.Conv2d(base_dim, 3, kernel_size=3, stride=1, padding=1))


        self.mix1 = SKFusion(base_dim * 4, reduction=8)
        self.mix2= SKFusion(base_dim * 2, reduction=4)


        self.onoffusion1 = OnOffFusion(base_dim * 2)
        self.onoffusion2 = OnOffFusion(base_dim * 4)
        self.onoffusion3 = OnOffFusion(base_dim * 8)




    def forward(self, x):
        de_x=x


        x_inp = x
        x_inv = 1.0 - x


        processed_x = self.processing_branch(x_inp)
        processed_x_inv = self.processing_branch(x_inv)


        for block in self.down_level1_blocks:
            processed_x = block(processed_x)
            
        for block in self.down_level1_blocks:
            processed_x_inv = block(processed_x_inv)

        x_down1 = self.down1(processed_x)

        x_down2_init = self.fe_level_3(x_down1)
        x_curr = x_down2_init
        for block in self.level3_blocks:
            x_curr = block(x_curr)
        x4 = x_curr

        x_down11 = self.down1(processed_x_inv)

        x_down22_init = self.fe_level_3(x_down11)
        x_curr = x_down22_init
        for block in self.level3_blocks:
            x_curr = block(x_curr)
        x24 = x_curr

        onoff1 = self.onoffusion1(x4, x24)

        x_down2 = self.down2(x4)

        x_down1_init = self.fe_level_2(x_down2)
        x_curr = x_down1_init
        for block in self.down_level2_blocks:
            x_curr = block(x_curr)
        x_down1_init = x_down2 + x_curr

        x_down21 = self.down2(x24)
        x_down21_init = self.fe_level_2(x_down21)
        x_curr = x_down21_init
        for block in self.down_level2_blocks:
            x_curr = block(x_curr)
        x_down21_init = x_down21 + x_curr

        onoff2 = self.onoffusion2(x_down1_init, x_down21_init)

        x_down3 = self.down3(x_down1_init)

        x_down3_ = self.fe_level_4(x_down3)
        x_curr = x_down3_
        for block in self.down_level4_blocks:
            x_curr = block(x_curr)
        x_down3_8 = x_curr + x_down3

        x_down31 = self.down3(x_down21_init)

        x_down23_ = self.fe_level_4(x_down31)
        x_curr = x_down23_
        for block in self.down_level4_blocks:
            x_curr = block(x_curr)
        x_down3_28 = x_curr + x_down31

        onoff3=self.onoffusion3(x_down3_8,x_down3_28)

        x_up1 = self.up0(onoff3)

        x_level4_mix = self.mix1(onoff2, x_up1)

        x_up2 = self.up1( x_level4_mix)

        x_level2_mix = self.mix2(onoff1, x_up2)

        x_up2 = self.up2(x_level2_mix)
        out = self.up3(x_up2)
        out = de_x+out
        out = torch.sigmoid(out)

        return out