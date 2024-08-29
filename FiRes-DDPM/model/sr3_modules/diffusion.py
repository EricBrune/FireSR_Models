import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import logging 

logger = logging.getLogger("base")


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64
    )
    return betas


def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "quad":
        betas = (
            np.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64
            )
            ** 2
        )
    elif schedule == "linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "warmup10":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == "warmup50":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == "const":
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class


def exists(x):
    return x is not None

class FeatureTransform(nn.Module):
    def __init__(self, feature_channels, transform_channels):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(feature_channels, transform_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(transform_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.transform(x)

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type="l1",
        conditional=True,
        schedule_opt=None,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)
        self.feature_transform_sssr = FeatureTransform(1, 1)

    def set_loss(self, device):
        if self.loss_type == "l1":
            self.loss_func = nn.L1Loss(reduction="sum").to(device)
        elif self.loss_type == "l2":
            self.loss_func = nn.MSELoss(reduction="sum").to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt["schedule"],
            n_timestep=schedule_opt["n_timestep"],
            linear_start=schedule_opt["linear_start"],
            linear_end=schedule_opt["linear_end"],
        )
        betas = (
            betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1.0, alphas_cumprod))

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self.sqrt_recip_alphas_cumprod[t] * x_t[:, :3]
            - self.sqrt_recipm1_alphas_cumprod[t] * noise[0],
            self.sqrt_recip_alphas_cumprod[t] * x_t[:, 3:]
            - self.sqrt_recipm1_alphas_cumprod[t] * noise[1],
        )

    def feature_similarity(self, features1, features2):
        # Perform down-sampling by a factor of 8 to save memory
        features1_down = F.interpolate(features1, scale_factor=1/8, mode='bilinear', align_corners=False)
        features2_down = F.interpolate(features2, scale_factor=1/8, mode='bilinear', align_corners=False)
        
        # Calculate normalized feature similarity
        f1 = features1_down.view(features1_down.size(0), features1_down.size(1), -1)
        f2 = features2_down.view(features2_down.size(0), features2_down.size(1), -1)
        f1_norm = F.normalize(f1, p=2, dim=2)
        f2_norm = F.normalize(f2, p=2, dim=2)
        
        # Compute similarity matrix S_ij
        similarity_matrix = torch.bmm(f1_norm.transpose(1, 2), f2_norm)
        
        return similarity_matrix


    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        )
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = (
            torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]])
            .repeat(batch_size, 1)
            .to(x.device)
        )
        x_recon1, x_recon2 = self.predict_start_from_noise(
            x,
            t=t,
            noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level),
        )
        x_recon = torch.cat((x_recon1, x_recon2), dim=1)

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x
        )
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = 1 | (self.num_timesteps // 10)
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(
                reversed(range(0, self.num_timesteps)),
                desc="sampling loop time step",
                total=self.num_timesteps,
            ):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = (x.shape[0], 4, x.shape[2], x.shape[3])
            img = torch.randn(shape, device=device)
            ret_img = x[:, :4]
            for i in tqdm(
                reversed(range(0, self.num_timesteps)),
                desc="sampling loop time step",
                total=self.num_timesteps,
            ):
                img = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop(
            (batch_size, channels, image_size, image_size), continous
        )

    def feature_affinity_loss(self, sr_features, seg_features):
        # Calculate feature similarity matrices
        sr_similarity = self.feature_similarity(sr_features, sr_features)
        seg_similarity = self.feature_similarity(seg_features, seg_features)
        
        # Compute Feature Affinity Loss
        fa_loss = self.loss_func(sr_similarity, seg_similarity)
        
        return fa_loss

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start
            + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = torch.cat((x_in["HR"], x_in["Post_Fire_Mask"]), dim=1)
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b,
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start,
            continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(
                -1, 1, 1, 1
            ),
            noise=noise,
        )

        sr_recon, seg_recon = self.denoise_fn(
            torch.cat([x_in["SR"], x_in["Pre_Fire"], x_in["Daymet"], x_in["LULC"], x_noisy], dim=1),
            continuous_sqrt_alpha_cumprod,
        )


        sr_loss = self.loss_func(noise[:, :3], sr_recon)
        seg_loss = self.loss_func(noise[:, 3:], seg_recon)
        
        sr_features, seg_features = self.predict_start_from_noise(x_noisy, min(t, 1999), (sr_recon, seg_recon))
        
        seg_features_transformed = self.feature_transform_sssr(seg_features)

        # Calculate feature similarity matrices using the transformed features
        sr_similarity = self.feature_similarity(sr_features, sr_features)
        seg_similarity = self.feature_similarity(seg_features_transformed, seg_features_transformed)
        
        # Compute Feature Affinity Loss using the similarity matrices
        fa_loss = self.loss_func(sr_similarity, seg_similarity)
        
        return sr_loss, seg_loss, fa_loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
