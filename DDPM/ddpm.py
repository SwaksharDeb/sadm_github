import torch
import torch.nn as nn
import numpy as np
import os 
import matplotlib.pyplot as plt


def ddpm_schedules_cosine(beta1, beta2, T, s=0.008):
    """
    Returns pre-computed schedules for DDPM sampling, training process using cosine scheduler.
    
    Args:
        beta1: minimum beta value (not used in cosine, kept for compatibility)
        beta2: maximum beta value (not used in cosine, kept for compatibility) 
        T: number of timesteps
        s: small offset to prevent beta from being too small near t=0
    """
    def alpha_bar_fn(t):
        return np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    
    # Create timesteps
    timesteps = torch.arange(0, T + 1, dtype=torch.float32)
    
    # Calculate alpha_bar for each timestep
    alpha_bar_t = torch.tensor([alpha_bar_fn(t / T) for t in timesteps], dtype=torch.float32)
    
    # Calculate beta_t from alpha_bar relationship
    # alpha_t = alpha_bar_t / alpha_bar_{t-1}
    alpha_t = torch.cat([torch.tensor([1.0]), alpha_bar_t[1:] / alpha_bar_t[:-1]])
    beta_t = 1 - alpha_t
    
    # Clip beta values to reasonable range
    beta_t = torch.clamp(beta_t, min=0.0001, max=0.9999)
    
    sqrt_beta_t = torch.sqrt(beta_t)
    
    # Recalculate alpha_bar_t to ensure consistency after clipping
    alphabar_t = torch.cumprod(alpha_t, dim=0)
    
    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


def ddpm_schedules_linear(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, vivit_model, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.vivit_model = vivit_model
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        
        for k, v in ddpm_schedules_cosine(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        # for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
        #     self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def plot_3d_volume_axes(self, x_0_predicted, x_t, save_folder="reconstructed_images", filename="volume_axes.png"):
        """
        Plot different axis views of a 3D volume in a single figure
        
        Args:
            x_0_predicted: torch tensor of shape [batch, channels, depth, height, width] - reconstructed volume
            x_t: torch tensor of shape [batch, channels, depth, height, width] - noisy volume
            save_folder: folder to save the image
            filename: name of the output file
        """
        
        # Take first sample from batch and remove batch and channel dimensions
        volume_clean = x_0_predicted[0, 0].detach().cpu().numpy()  # Shape: [128, 128, 128]
        volume_noisy = x_t[0, 0].detach().cpu().numpy()  # Shape: [128, 128, 128]
        
        # Create save directory if it doesn't exist
        os.makedirs(save_folder, exist_ok=True)
        
        # Get middle slices for each axis
        depth, height, width = volume_clean.shape
        mid_depth = depth // 2
        mid_height = height // 2
        mid_width = width // 2
        
        # Create figure with subplots (2 rows, 3 columns)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('3D Volume Comparison: Noisy (Top) vs Reconstructed (Bottom)', fontsize=16)
        
        # TOP ROW: Noisy x_t volumes
        # Axial view (top-down, looking along depth axis)
        im1 = axes[0, 0].imshow(volume_noisy[mid_depth, :, :], cmap='gray')
        axes[0, 0].set_title(f'Noisy Axial View (Depth slice: {mid_depth})')
        axes[0, 0].set_xlabel('Width')
        axes[0, 0].set_ylabel('Height')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Coronal view (front-to-back, looking along height axis)
        im2 = axes[0, 1].imshow(volume_noisy[:, mid_height, :], cmap='gray')
        axes[0, 1].set_title(f'Noisy Coronal View (Height slice: {mid_height})')
        axes[0, 1].set_xlabel('Width')
        axes[0, 1].set_ylabel('Depth')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Sagittal view (side-to-side, looking along width axis)
        im3 = axes[0, 2].imshow(volume_noisy[:, :, mid_width], cmap='gray')
        axes[0, 2].set_title(f'Noisy Sagittal View (Width slice: {mid_width})')
        axes[0, 2].set_xlabel('Height')
        axes[0, 2].set_ylabel('Depth')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # BOTTOM ROW: Reconstructed x_0_predicted volumes
        # Axial view (top-down, looking along depth axis)
        im4 = axes[1, 0].imshow(volume_clean[mid_depth, :, :], cmap='gray')
        axes[1, 0].set_title(f'Reconstructed Axial View (Depth slice: {mid_depth})')
        axes[1, 0].set_xlabel('Width')
        axes[1, 0].set_ylabel('Height')
        plt.colorbar(im4, ax=axes[1, 0])
        
        # Coronal view (front-to-back, looking along height axis)
        im5 = axes[1, 1].imshow(volume_clean[:, mid_height, :], cmap='gray')
        axes[1, 1].set_title(f'Reconstructed Coronal View (Height slice: {mid_height})')
        axes[1, 1].set_xlabel('Width')
        axes[1, 1].set_ylabel('Depth')
        plt.colorbar(im5, ax=axes[1, 1])
        
        # Sagittal view (side-to-side, looking along width axis)
        im6 = axes[1, 2].imshow(volume_clean[:, :, mid_width], cmap='gray')
        axes[1, 2].set_title(f'Reconstructed Sagittal View (Width slice: {mid_width})')
        axes[1, 2].set_xlabel('Height')
        axes[1, 2].set_ylabel('Depth')
        plt.colorbar(im6, ax=axes[1, 2])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(save_folder, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Volume visualization saved to: {save_path}")
        
    def forward(self, x, x_prev, step=3, div_factor=100):
        """
        this method is used in training, so samples t and noise randomly
        """

        c = self.vivit_model(x_prev)

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
                self.sqrtab[_ts, None, None, None, None] * x
                + self.sqrtmab[_ts, None, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + self.drop_prob).to(self.device)

        ##Predicted noise from the model
        predicted_noise = self.nn_model(x_t, c, _ts / self.n_T, context_mask)

        # Reconstruct the original image using the reverse diffusion formula
        x_0_predicted = (x_t - self.sqrtmab[_ts, None, None, None, None] * predicted_noise) / self.sqrtab[_ts, None, None, None, None]

        if step % div_factor == 0:
            self.plot_3d_volume_axes(x_0_predicted, x_t, save_folder="reconstructed_images", filename=f'volume_axes_{step}.png')

        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, x_prev, device, guide_w=0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        c_i = self.vivit_model(x_prev)

        n_sample = c_i.shape[0]
        size = c_i.shape[1:]

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise

        # don't drop context at test time
        context_mask = torch.zeros(c_i.shape[0]).to(device)

        # double the batch
        c_i = c_i.repeat(2, 1, 1, 1, 1)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.  # makes second half of batch context free

        x_i_store = []  # keep track of generated steps in case want to plot something
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}', end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1, 1)

            # double batch
            x_i = x_i.repeat(2, 1, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:n_sample]
            x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
            )
            #x_i = x_i.clip(-1,1)
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store