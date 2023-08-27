import torch
from torch import nn

class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv3d(in_dim, hidden_dim, 1, bias=False)
        self.conv_2 = nn.Conv3d(hidden_dim, out_dim, 1, bias=False)
        self.conv_shortcut = nn.Conv3d(in_dim, out_dim, 1, bias=False)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, D, H, W)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out


class SkipTransformer(nn.Module):
    def __init__(self, in_channels, dim=512):
        super().__init__()
        self.query = nn.Conv3d(in_channels, dim, 1, bias=False)
        self.key = nn.Conv3d(in_channels, dim, 1, bias=False)
        self.value = nn.Conv3d(in_channels, dim, 1, bias=False)

        self.value_mlp = MLP_Res(in_dim=dim*2, hidden_dim=in_channels, out_dim=in_channels)

        self.att_mlp = nn.Sequential(
            nn.Conv3d(dim, dim * 2, 1, bias=False),
            nn.BatchNorm3d(dim * 2),
            nn.ReLU(),
            nn.Conv3d(dim * 2, dim, 1, bias=False)
        )

        self.end_conv = nn.Conv3d(dim, in_channels, 1, bias=False)

    def forward(self, x):
        '''
        Args:
            x: [B, C, D, H, W]
            other args like key, query and so on, produced by input x as CoT
        Returns:
            Shape Context Feature: [B, C, D, H, W]
        '''
        query = self.query(x) #[B, dim, D, H, W]
        key = self.query(x) #[B, dim, D, H, W]
        value = self.value_mlp(torch.cat([query, key], dim=1)) #[B, 2*dim, D, H, W] -> [B, inchannel, D, H, W]
        final_adding_value = value #[B, inchannel, D, H, W]
        value = self.value(value) #[B, dim, D, H, W]

        B,C,D,H,W = value.shape

        attention = self.att_mlp(query-key) #[B, dim, D, H, W]

        # attention = self.att_mlp(query-key).reshape(B,C,-1)  #[B, dim, D*H*W]
        # att_weight = torch.softmax(attention, -1).reshape(B,C,D,H,W)

        y = self.end_conv(torch.mul(attention, value)) #[B, dim, D, H, W]
        return y + final_adding_value
    
if __name__ == "__main__":
    x = torch.randn(2,128,12,12,12)
    net = SkipTransformer(128)
    print(net(x).shape)