import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


BLOCKS = {
    "content": ["down_blocks"],
    "style": ["up_blocks"],
}

controlnet_BLOCKS = {
    "content": [],
    "style": ["down_blocks"],
}


def resize_width_height(width, height, min_short_side=512, max_long_side=1024):
    if width < height:
        if width < min_short_side:
            scale = min_short_side / width
            new_width = min_short_side
            new_height = int(height * scale)
        else:
            new_width, new_height = width, height
    else:
        if height < min_short_side:
            scale = min_short_side / height
            new_width = int(width * scale)
            new_height = min_short_side
        else:
            new_width, new_height = width, height

    if max(new_width, new_height) > max_long_side:
        scale = max_long_side / max(new_width, new_height)
        new_width = int(new_width * scale)
        new_height = int(new_height * scale)

    return new_width, new_height


def resize_content(content_image):
    new_w, new_h = resize_width_height(
        content_image.size[0],
        content_image.size[1],
        min_short_side=1024,
        max_long_side=1024,
    )

    new_w = new_w // 16 * 16
    new_h = new_h // 16 * 16

    content_image = content_image.resize((new_w, new_h))
    return new_w, new_h, content_image


attn_maps = {}


def hook_fn(name):
    def forward_hook(module, _, __):
        if hasattr(module.processor, "attn_map"):
            attn_maps[name] = module.processor.attn_map
            del module.processor.attn_map

    return forward_hook


def register_cross_attention_hook(unet):
    for name, module in unet.named_modules():
        if name.split(".")[-1].startswith("attn2"):
            module.register_forward_hook(hook_fn(name))
    return unet


def upscale(attn_map, target_size):
    attn_map = torch.mean(attn_map, dim=0)
    attn_map = attn_map.permute(1, 0)

    temp_size = None
    for i in range(5):
        scale = 2**i
        if (target_size[0] // scale) * (target_size[1] // scale) == attn_map.shape[1] * 64:
            temp_size = (target_size[0] // (scale * 8), target_size[1] // (scale * 8))
            break

    if temp_size is None:
        raise RuntimeError

    attn_map = attn_map.view(attn_map.shape[0], *temp_size)

    attn_map = F.interpolate(
        attn_map.unsqueeze(0).float(),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )[0]

    return torch.softmax(attn_map, dim=0)


def get_net_attn_map(image_size, batch_size=2, instance_or_negative=False, detach=True):
    idx = 0 if instance_or_negative else 1
    maps = []

    for _, attn_map in attn_maps.items():
        attn_map = attn_map.cpu() if detach else attn_map
        attn_map = torch.chunk(attn_map, batch_size)[idx].squeeze()
        attn_map = upscale(attn_map, image_size)
        maps.append(attn_map)

    return torch.mean(torch.stack(maps, dim=0), dim=0)


def attnmaps2images(net_attn_maps):
    images = []
    for attn_map in net_attn_maps:
        attn_map = attn_map.cpu().numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        attn_map = (attn_map * 255).astype(np.uint8)
        images.append(Image.fromarray(attn_map))
    return images


def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")


def get_generator(seed, device):
    if seed is None:
        return None
    if isinstance(seed, list):
        return [torch.Generator(device).manual_seed(s) for s in seed]
    return torch.Generator(device).manual_seed(seed)
