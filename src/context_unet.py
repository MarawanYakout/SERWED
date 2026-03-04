"""
ContextUnet: UNet with time and context conditioning for diffusion models.
Extracted from training/Sampling_storm notebook.
"""
import torch
import torch.nn as nn
from src_refactored.model import ResidualConvBlock, UnetDown, UnetUp, EmbedFC

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=28):
        super(ContextUnet, self).__init__()
        """
        Context-conditional UNet for diffusion models.
        
        Args:
            in_channels: number of input channels (3 for RGB)
            n_feat: base number of feature channels
            n_cfeat: number of context features (one-hot encoding size)
            height: image height (assumes square images)
        """
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height

        # Initialize the initial convolutional layer
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # Initialize the down-sampling path of the U-Net with two levels
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        
        # Average pooling to vector
        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())

        # Embed the timestep and context labels with a one-layer fully connected neural network
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2*n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1*n_feat)

        # Initialize the up-sampling path of the U-Net with three levels
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h//4, self.h//4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        # Initialize the final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t, c=None):
        """
        Forward pass with time and context conditioning.
        
        Args:
            x: (batch, n_feat, h, w) input image
            t: (batch, 1) or (batch,) timestep
            c: (batch, n_cfeat) context label (optional, zeros if None)
        
        Returns:
            (batch, in_channels, h, w) predicted noise
        """
        # Ensure t has correct shape
        if t.dim() == 1:
            t = t.unsqueeze(1).float()
        
        # Pass the input image through the initial convolutional layer
        x = self.init_conv(x)
        
        # Pass the result through the down-sampling path
        down1 = self.down1(x)
        down2 = self.down2(down1)
        
        # Convert the feature maps to a vector and apply an activation
        hiddenvec = self.to_vec(down2)
        
        # Mask out context if context is None
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)
        
        # Embed context and timestep
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # Up-sampling with conditioning
        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1 + temb1, down2)
        up3 = self.up2(cemb2*up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out
