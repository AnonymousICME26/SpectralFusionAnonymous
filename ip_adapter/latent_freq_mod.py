"""
Latent-Space Frequency Modulation for SpectralFusion
Novel contribution: Operate in latent space (128x128x4) instead of pixel space (1024x1024x3)
64x faster + integrated into diffusion process
"""
import torch
import torch.nn.functional as F

class LatentFrequencyModulator:
    """
    Novel: Frequency modulation in latent space during diffusion
    - Operates on 128x128x4 latents (64x smaller than pixel space)
    - Timestep-adaptive frequency mixing
    - No post-processing needed
    """
    def __init__(self, device='cuda', dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        
    def create_frequency_mask(self, fft_shape, t, total_steps, cutoff_low=0.25, cutoff_high=0.75):
        """
        Timestep-adaptive frequency mask for FFT output
        Args:
            fft_shape: Shape of fftshift output (H, W_fft) where W_fft = W//2+1 for rfft2
        """
        H, W_fft = fft_shape
        
        # Normalized timestep (0 = end, 1 = start)
        t_norm = t / total_steps
        
        # Create radial frequency mask
        crow, ccol = H // 2, W_fft // 2
        
        # Create coordinate grids matching FFT output shape
        y = torch.arange(H, device=self.device, dtype=torch.float32).view(-1, 1).expand(H, W_fft)
        x = torch.arange(W_fft, device=self.device, dtype=torch.float32).view(1, -1).expand(H, W_fft)
        
        dist_from_center = torch.sqrt((x - ccol)**2 + (y - crow)**2)
        max_dist = torch.sqrt(torch.tensor(crow**2 + ccol**2, dtype=torch.float32, device=self.device))
        normalized_dist = dist_from_center / (max_dist + 1e-8)
        
        # Adaptive cutoffs based on timestep
        cutoff_low_adaptive = cutoff_low * (0.7 + 0.3 * (1 - t_norm))
        cutoff_high_adaptive = cutoff_high * (1.3 - 0.3 * (1 - t_norm))
        
        # Create masks
        low_freq_mask = (normalized_dist < cutoff_low_adaptive).float()
        high_freq_mask = (normalized_dist > cutoff_high_adaptive).float()
        mid_freq_mask = 1.0 - low_freq_mask - high_freq_mask
        
        return low_freq_mask, mid_freq_mask, high_freq_mask, t_norm
    
    @torch.no_grad()
    def modulate(self, content_latent, style_latent, t, total_steps=50):
        """
        Apply frequency-domain modulation in latent space
        
        Args:
            content_latent: Latent from content (B, 4, H, W)
            style_latent: Style reference latent (B, 4, H_s, W_s) - will be resized
            t: Current timestep
            total_steps: Total diffusion steps
            
        Returns:
            Modulated latent
        """
        if style_latent is None:
            return content_latent
            
        B, C, H, W = content_latent.shape
        original_dtype = content_latent.dtype
        
        # Convert to float32 for FFT (cuFFT requires power-of-2 for float16)
        content_latent = content_latent.float()
        style_latent = style_latent.float()
        
        # Resize style_latent to match content_latent if needed
        if style_latent.shape[-2:] != (H, W):
            style_latent = torch.nn.functional.interpolate(
                style_latent, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Process each channel independently
        modulated = torch.zeros_like(content_latent)
        
        for c in range(C):
            # FFT (rfft2 output: B x H x (W//2+1))
            content_fft = torch.fft.rfft2(content_latent[:, c])
            style_fft = torch.fft.rfft2(style_latent[:, c])
            
            # Shift to center (preserves shape)
            content_fft_shift = torch.fft.fftshift(content_fft, dim=(-2, -1))
            style_fft_shift = torch.fft.fftshift(style_fft, dim=(-2, -1))
            
            # Get FFT output shape: (H, W_fft) where W_fft = W//2+1
            fft_h, fft_w = content_fft_shift.shape[-2:]
            
            # Create masks matching FFT dimensions
            low_mask, mid_mask, high_mask, t_norm = self.create_frequency_mask(
                (fft_h, fft_w), t, total_steps
            )
            
            # Expand masks to batch dimension
            low_mask = low_mask.unsqueeze(0).expand(B, -1, -1)
            mid_mask = mid_mask.unsqueeze(0).expand(B, -1, -1)
            high_mask = high_mask.unsqueeze(0).expand(B, -1, -1)
            
            # Adaptive blending based on timestep
            structure_weight = 0.9 + 0.1 * t_norm
            texture_weight = 0.3 + 0.4 * (1 - t_norm)
            
            # Frequency-domain mixing
            result_fft = (
                content_fft_shift * low_mask * structure_weight +
                (content_fft_shift * 0.5 + style_fft_shift * 0.5) * mid_mask +
                style_fft_shift * high_mask * texture_weight
            )
            
            # Inverse FFT
            result_fft = torch.fft.ifftshift(result_fft, dim=(-2, -1))
            result = torch.fft.irfft2(result_fft, s=(H, W))
            
            modulated[:, c] = result
        
        # Convert back to original dtype
        return modulated.to(original_dtype)
    
    def encode_style(self, style_image, vae):
        """Encode style image to latent space"""
        with torch.no_grad():
            style_latent = vae.encode(style_image).latent_dist.sample()
            style_latent = style_latent * vae.config.scaling_factor
        return style_latent

