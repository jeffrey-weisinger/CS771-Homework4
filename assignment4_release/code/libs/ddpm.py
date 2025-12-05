# Implementation of DDPM described in https://arxiv.org/abs/2006.11239
# Reference: https://huggingface.co/blog/annotated-diffusion

import torch
from torch import nn
import torch.nn.functional as F

from .unet import UNet
from .tiny_autoencoder import TAESD


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model"""

    def __init__(
        self,
        img_shape=(3, 32, 32),
        timesteps=100,
        dim=64,
        context_dim=64,
        num_classes=10,
        dim_mults=(1, 2, 4),
        attn_levels=(0, 1),
        use_vae=False,
        vae_encoder_weights=None,
        vae_decoder_weights=None,
    ):
        """
        Args:
            img_shape (tuple/list of int): shape of input image or diffusion
                latent space (C x H x W)
            timesteps (int): number of timesteps in the diffusion process
            dim (int): base feature dimension in UNet
            context_dim (int): condition dimension (embedding of the label) in UNet
            num_classes (int): number of classes used for conditioning
            dim_mults (tuple/list of int): multiplier of feature dimensions in UNet
                length of this list specifies #blockes in UNet encoder/decoder
                e.g., (1, 2, 4) -> 3 blocks with output dims of 1x, 2x, 4x
                w.r.t. the base feature dim
            attn_levels (tuple/list of int): specify if attention layer is included
                in a block in UNet encoder/decoder
                e.g., (0, 1) -> the first two blocks in the encoder and the last two
            use_vae (bool): if a VAE is used before DDPM (thus latent diffusion)
            vae_encoder_weights (str): path to pre-trained VAE encoder weights
            vae_decoder_weights (str): path to pre-trained VAE encoder weights
        """

        super().__init__()

        assert len(img_shape) == 3
        self.timesteps = timesteps
        # use different variance schedule for LDM / DDPM
        if use_vae:
            betas = self.quadratic_beta_schedule(timesteps)
        else:
            betas = self.linear_beta_schedule(timesteps)

        alpha_vars = self.compute_alpha_vars(betas)
        self.betas = betas
        self.sqrt_recip_alphas = alpha_vars[0]
        self.sqrt_alphas_cumprod = alpha_vars[1]
        self.sqrt_one_minus_alphas_cumprod = alpha_vars[2]
        self.posterior_variance = alpha_vars[3]
        self.use_vae = use_vae

        # if VAE is considered, input / output dim will be the dim of latent
        if use_vae:
            in_channels = out_channels = 4
            self.img_shape = [4, img_shape[1]//8, img_shape[2]//8]
        else:
            in_channels = out_channels = img_shape[0]
            self.img_shape = img_shape

        # the denoising model using UNet (conditioned on input label)
        self.model = UNet(
            dim,
            context_dim,
            num_classes,
            in_channels=in_channels,
            out_channels=out_channels,
            dim_mults=dim_mults,
            attn_levels=attn_levels
        )

        # if we should consider latent DDPM
        if use_vae:
            assert vae_encoder_weights is not None
            assert vae_decoder_weights is not None
            self.vae = TAESD(
                encoder_path=vae_encoder_weights,
                decoder_path=vae_decoder_weights
            )
            # freeze the encoder / decoder
            for param in self.vae.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def linear_beta_schedule(self, timesteps):
        """
        linear schedule as in DDPM paper (Sec 4)
        """
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)

    @torch.no_grad()
    def quadratic_beta_schedule(self, timesteps):
        """
        quadratic schedule as in Stable Diffusion
        """
        beta_start = 0.00085
        beta_end = 0.012
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

    @torch.no_grad()
    def compute_alpha_vars(self, betas):
        """
        compute vars related to alphas from betas
        """
        # define alphas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        return (
            sqrt_recip_alphas,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            posterior_variance
        )

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Helper function to match the dimensions.
        It sets a[t[i]] for every element t[i] in t and expands the results
        into x_shape
        """

        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    @torch.no_grad()
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process (using the nice property from Gaussian)
        It adds noise to a starting image and outputs its noisy version at
        an arbitary time step t
        """

        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        """
        Fill in the missing code here. See Equation 4 in DDPM paper.
        """

        #in the paper, they give the equation of N(x_t; mean =sqrt(cumprod(a))*x_start, covariance=(1-cumprod(a))*I)
        #if covariance=(1-cumprod(a))*I, then var for a single dimension=(1-cumprod(a)), and std = sqrt(1-cumprod(a)) (since variances live on diagonal of covariance matrix)
        #so we can use torch.normal which takes mean=sqrt(cumprod(a))*x_start, and std=sqrt(1-cumprod(a))

        x_t = torch.normal(sqrt_alphas_cumprod_t*x_start, std = sqrt_one_minus_alphas_cumprod_t, size = x_start.shape)
        return x_t

    # compute the simplified loss
    def compute_loss(self, x_start, label, noise=None):
        """
        Compute the simplified loss for training the mode - e_predictionel.
        Fill in the missing code here. Algorithm 1 line 3-5 in the paper.
        For latent DDPMs, an additional encoding step will be needed.
        """

        #we want one new timestep for every iteration
        timestep = torch.randint(0, self.timesteps, size=x_start.shape[0])

        #getting values to calculate the e_prediction
        sqrt_alphas_cumprod = self._extract(
            self.sqrt_alphas_cumprod, timestep, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timestep, x_start.shape
        )

        #defining the two different e's (true and prediction) used in loss
        e = torch.normal(mean=0, std=1, size=x_start.shape)
        e_prediction = self.model(sqrt_alphas_cumprod*x_start + sqrt_one_minus_alphas_cumprod*e, timestep)
        #putting everything all together into the  loss
        loss = torch.mean(torch.square(e - e_prediction))
        return loss

    @torch.no_grad()
    def p_sample(self, x, label, t, t_index):
        """
        Denoise a noisy image at time step t (single step)
        """

        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)

        """
        Fill in the missing code here. See Equation 11 (also Algorithm 2 line 3-4)
        in DDPM paper.
        """
        #as mentioned in the paper, e is predicted using UNet. therefore, we will call the forward pass
        e_prediction = self.model(x, label, t)
        #this just uses the paper equation to write mu
        mu = sqrt_recip_alphas_t * (x - (betas_t/sqrt_one_minus_alphas_cumprod_t)*e_prediction)
        #this again just uses the paper equation to sample x_t-1
        #note that the paper writes two choices for var, we choose the first version (var_t^2 = beta_t) since they say it works better with N(0, I) which we are using.
        if t > 1:
            z = torch.normal(mean=0, std=1, size=x.shape)
        else:
            z = 0
        x_t_minus_1_sample = mu + torch.sqrt(betas_t)*z
        return x_t_minus_1_sample
    @torch.no_grad()
    def generate(self, labels):
        """
        Sampling from DDPM (algorithm 2)
        """

        device = next(self.model.parameters()).device
        # shape of the results
        shape = [len(labels)] + list(self.img_shape)
        # start from pure noise (for each example in the batch)
        imgs = torch.randn(shape, device=device)

        # draw samples
        """
        Fill in the missing code here. See Equation 11 / Algorithm 2 in DDPM paper.
        For latent DDPMs, an additional decoding step will be needed.
        """
        # for ...

        # postprocessing the images
        imgs = self.postprocess(imgs)
        return imgs

    @torch.no_grad()
    def postprocess(self, imgs):
        """
        Postprocess the sampled images (e.g., normalization / clipping)
        """
        if self.use_vae:
            # already in range, clip the pixel values
            imgs.clamp_(min=0.0, max=1.0)
        else:
            # clip the pixel values within range
            imgs.clamp_(min=-1.0, max=1.0)
            # mapping to range [0, 1]
            imgs = (imgs + 1.0) * 0.5
        return imgs
