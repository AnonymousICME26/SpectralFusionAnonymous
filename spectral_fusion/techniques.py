import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import List, Optional

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

try:
    import pywt
except ImportError:
    pywt = None

from .config import (
    FDSM_STRUCTURE_WEIGHT,
    FDSM_TEXTURE_WEIGHT,
    FDSM_TRANSITION_SMOOTH,
    SAF_WAVELET_TYPE,
    SAF_DECOMP_LEVELS,
    SAF_COARSE_WEIGHT,
    SAF_DETAIL_WEIGHTS,
    PLR_NUM_STAGES,
    PLR_RESOLUTION_RATIOS,
    PLR_REFINEMENT_WEIGHTS,
    HCP_PROTECTION_RADIUS,
    HCP_ATTENUATION_FACTOR,
    EPSILON,
)


def transfer_color_lab(
    generated_image: Image.Image,
    style_image: Image.Image,
    strength: float = 1.0,
) -> Image.Image:
    gen_array = np.array(generated_image).astype(np.float32)
    style_array = np.array(style_image).astype(np.float32)

    if style_array.shape[:2] != gen_array.shape[:2]:
        style_array = cv2.resize(
            style_array, (gen_array.shape[1], gen_array.shape[0])
        )

    gen_lab = cv2.cvtColor(
        gen_array.astype(np.uint8), cv2.COLOR_RGB2LAB
    ).astype(np.float32)
    style_lab = cv2.cvtColor(
        style_array.astype(np.uint8), cv2.COLOR_RGB2LAB
    ).astype(np.float32)

    gen_l, gen_a, gen_b = cv2.split(gen_lab)
    _, style_a, style_b = cv2.split(style_lab)

    gen_a_mean, gen_a_std = gen_a.mean(), gen_a.std() + EPSILON
    gen_b_mean, gen_b_std = gen_b.mean(), gen_b.std() + EPSILON

    style_a_mean, style_a_std = style_a.mean(), style_a.std() + EPSILON
    style_b_mean, style_b_std = style_b.mean(), style_b.std() + EPSILON

    gen_a_norm = (gen_a - gen_a_mean) / gen_a_std
    gen_b_norm = (gen_b - gen_b_mean) / gen_b_std

    result_a = gen_a_norm * style_a_std + style_a_mean
    result_b = gen_b_norm * style_b_std + style_b_mean

    result_a = gen_a * (1 - strength) + result_a * strength
    result_b = gen_b * (1 - strength) + result_b * strength

    result_lab = cv2.merge([gen_l, result_a, result_b])
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    result_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(result_rgb)


def compute_adaptive_frequency_attention(gen_fft_shift, style_fft_shift):
    content_magnitude = np.abs(gen_fft_shift)
    style_magnitude = np.abs(style_fft_shift)

    rows, cols = content_magnitude.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    max_dist = np.sqrt(crow ** 2 + ccol ** 2) + EPSILON
    norm_dist = dist / max_dist

    perceptual_weight = (
        np.exp(-((norm_dist - 0.4) ** 2) / 0.15) * 0.5 + 0.75
    )

    base_attention = style_magnitude / (
        content_magnitude + style_magnitude + EPSILON
    )
    attention = base_attention * perceptual_weight

    fine = cv2.GaussianBlur(attention.astype(np.float32), (0, 0), 1.5)
    coarse = cv2.GaussianBlur(attention.astype(np.float32), (0, 0), 4.0)
    attention = fine * 0.6 + coarse * 0.4

    attention = np.power(attention, 0.9)
    return np.clip(attention, 0.0, 1.0)


def compute_saliency_protection_mask(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    saliency = np.sqrt(sobelx ** 2 + sobely ** 2)
    saliency = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)
    saliency = cv2.GaussianBlur(saliency, (15, 15), 0)
    return saliency


def compute_clarity_modulation(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    clarity = cv2.GaussianBlur(np.abs(laplacian), (15, 15), 0)
    clarity = cv2.normalize(clarity, None, 0, 1, cv2.NORM_MINMAX)
    return clarity


def apply_fdsm(
    generated_image: Image.Image,
    style_image: Image.Image,
    structure_weight: float = FDSM_STRUCTURE_WEIGHT,
    texture_weight: float = FDSM_TEXTURE_WEIGHT,
    content_aware: bool = True,
) -> Image.Image:
    gen_array = np.array(generated_image).astype(np.float32)
    style_array = np.array(style_image).astype(np.float32)

    if style_array.shape[:2] != gen_array.shape[:2]:
        style_array = cv2.resize(
            style_array, (gen_array.shape[1], gen_array.shape[0])
        )

    saliency = compute_saliency_protection_mask(generated_image)
    clarity = compute_clarity_modulation(generated_image)

    gen_lab = cv2.cvtColor(
        gen_array.astype(np.uint8), cv2.COLOR_RGB2LAB
    ).astype(np.float32)
    style_lab = cv2.cvtColor(
        style_array.astype(np.uint8), cv2.COLOR_RGB2LAB
    ).astype(np.float32)

    result_channels = []

    for i in range(3):
        gen_channel = gen_lab[:, :, i]
        style_channel = style_lab[:, :, i]

        gen_fft = np.fft.fft2(gen_channel)
        style_fft = np.fft.fft2(style_channel)

        gen_shift = np.fft.fftshift(gen_fft)
        style_shift = np.fft.fftshift(style_fft)

        rows, cols = gen_channel.shape
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        dist = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
        max_dist = np.sqrt(crow ** 2 + ccol ** 2)
        norm_dist = dist / max_dist

        adaptive_attention = compute_adaptive_frequency_attention(
            gen_shift, style_shift
        )

        sigma = FDSM_TRANSITION_SMOOTH * max_dist
        sigma_sq = sigma * sigma + EPSILON

        low = np.exp(-(norm_dist ** 2) / (2.0 * sigma_sq))
        high = 1.0 - np.exp(-((norm_dist - 1.0) ** 2) / (2.0 * sigma_sq))
        mid = np.clip(1.0 - low - high, 0.0, 1.0)

        saliency_fft = np.fft.fftshift(np.fft.fft2(saliency))
        edge_protection = 1.0 + np.abs(saliency_fft) * 0.3

        if i == 0:
            struct_w = structure_weight * 1.2
            text_w = texture_weight * 0.6
        else:
            struct_w = structure_weight * 0.7
            text_w = texture_weight * 1.3

        if content_aware:
            content_imp = (low * struct_w + mid * 0.7) * edge_protection
            style_imp = (high * text_w + mid * 0.3) * adaptive_attention
            total = content_imp + style_imp + EPSILON
            content_imp /= total
            style_imp /= total

            result_fft = (
                gen_shift * content_imp + style_shift * style_imp
            )
        else:
            result_fft = (
                gen_shift * low * struct_w * edge_protection
                + (gen_shift * 0.5 + style_shift * 0.5) * mid
                + style_shift * high * text_w * adaptive_attention
            )

        result_channel = np.fft.ifft2(
            np.fft.ifftshift(result_fft)
        ).real
        result_channels.append(result_channel)

    result_lab = np.stack(result_channels, axis=-1)
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    result_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)

    result_img = Image.fromarray(result_rgb)

    avg_clarity = np.mean(clarity)
    style_lab_stats = cv2.cvtColor(
        np.array(style_image).astype(np.uint8),
        cv2.COLOR_RGB2LAB,
    ).astype(np.float32)
    style_var = np.std(style_lab_stats[:, :, 1:])

    strength_factor = np.clip(style_var / 30.0, 0.6, 1.2)
    base_strength = 0.85 * strength_factor
    modulated = base_strength * (0.7 + 0.3 * (1.0 - avg_clarity))

    result_img = transfer_color_lab(
        result_img, style_image, strength=modulated
    )

    result_array = np.array(result_img)
    lab_check = cv2.cvtColor(
        result_array, cv2.COLOR_RGB2LAB
    ).astype(np.float32)
    current_sat = np.std(lab_check[:, :, 1:])

    target_sat = style_var * 0.9
    if current_sat < target_sat:
        boost = 1.0 + min(
            (target_sat - current_sat) / 20.0, 0.25
        )
    else:
        boost = 1.05

    enhancer = ImageEnhance.Color(result_img)
    return enhancer.enhance(boost)


def apply_saf(
    images: List[Image.Image],
    weights: Optional[List[float]] = None,
) -> Image.Image:
    if len(images) == 1:
        return images[0]

    if pywt is None:
        return images[0]

    if weights is None:
        weights = [1.0 / len(images)] * len(images)

    arrays = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img).astype(np.float32)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arrays.append(arr)

    if not arrays:
        return images[0]

    target_shape = arrays[0].shape[:2]
    for i in range(1, len(arrays)):
        if arrays[i].shape[:2] != target_shape:
            arrays[i] = cv2.resize(
                arrays[i],
                (target_shape[1], target_shape[0]),
            )

    saliency_maps = []
    for arr in arrays:
        gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sal = cv2.GaussianBlur(np.abs(lap), (5, 5), 0)
        saliency_maps.append(sal)

    result_channels = []

    for ch in range(3):
        decomps = [
            pywt.wavedec2(
                arr[:, :, ch],
                SAF_WAVELET_TYPE,
                level=SAF_DECOMP_LEVELS,
            )
            for arr in arrays
        ]

        fused = []
        for level in range(len(decomps[0])):
            if level == 0:
                acc = np.zeros_like(decomps[0][0])
                for i, coeffs in enumerate(decomps):
                    sal = cv2.resize(
                        saliency_maps[i],
                        (acc.shape[1], acc.shape[0]),
                    )
                    sal /= np.sum(sal) + EPSILON
                    acc += (
                        coeffs[0]
                        * weights[i]
                        * SAF_COARSE_WEIGHT
                        * (1.0 + sal * 0.5)
                    )
                fused.append(acc)
            else:
                detail_w = SAF_DETAIL_WEIGHTS[
                    min(level - 1, len(SAF_DETAIL_WEIGHTS) - 1)
                ]
                fused_details = []

                for d in range(len(decomps[0][level])):
                    energies = [
                        np.abs(coeffs[level][d])
                        for coeffs in decomps
                    ]
                    stack = np.stack(energies, axis=-1)
                    max_idx = np.argmax(stack, axis=-1)

                    out = np.zeros_like(decomps[0][level][d])
                    for i in range(len(decomps)):
                        mask = (max_idx == i).astype(np.float32)
                        mask = cv2.GaussianBlur(mask, (5, 5), 0)
                        out += (
                            decomps[i][level][d] * mask * detail_w
                        )

                    for i, coeffs in enumerate(decomps):
                        out += (
                            coeffs[level][d]
                            * weights[i]
                            * detail_w
                            * 0.1
                        )

                    fused_details.append(out)

                fused.append(tuple(fused_details))

        rec = pywt.waverec2(fused, SAF_WAVELET_TYPE)
        rec = rec[: target_shape[0], : target_shape[1]]
        result_channels.append(rec)

    result = np.stack(result_channels, axis=-1)
    result = np.clip(result, 0, 255).astype(np.uint8)
    result = cv2.bilateralFilter(result, 5, 10, 10)
    return Image.fromarray(result)


def apply_plr(
    image: Image.Image,
    num_stages: int = PLR_NUM_STAGES,
) -> Image.Image:
    img = np.array(image).astype(np.float32)
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    detail_map = cv2.normalize(
        cv2.GaussianBlur(np.abs(lap), (5, 5), 0),
        None,
        0,
        1,
        cv2.NORM_MINMAX,
    )

    edges = cv2.Canny(gray, 50, 150)
    edge_map = cv2.normalize(
        cv2.GaussianBlur(
            cv2.dilate(edges, np.ones((3, 3), np.uint8)).astype(
                np.float32
            ),
            (3, 3),
            0,
        ),
        None,
        0,
        1,
        cv2.NORM_MINMAX,
    )

    current = img

    for i in range(num_stages):
        ratio = PLR_RESOLUTION_RATIOS[
            min(i, len(PLR_RESOLUTION_RATIOS) - 1)
        ]
        weight = PLR_REFINEMENT_WEIGHTS[
            min(i, len(PLR_REFINEMENT_WEIGHTS) - 1)
        ]

        th, tw = int(h * ratio), int(w * ratio)

        if ratio < 1.0:
            down = cv2.resize(
                current, (tw, th), interpolation=cv2.INTER_AREA
            )
            up = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)
            residual = current - up
        else:
            down = current
            residual = np.zeros_like(current)

        refined = cv2.bilateralFilter(
            down.astype(np.uint8), 9, 75, 75
        ).astype(np.float32)

        dm = (
            cv2.resize(detail_map, (tw, th))
            if ratio < 1.0
            else detail_map
        )
        aw = weight * (1.0 - dm * 0.6)
        aw = np.expand_dims(aw, -1)

        blended = down * (1.0 - aw) + refined * aw

        if ratio < 1.0:
            up = cv2.resize(
                blended, (w, h), interpolation=cv2.INTER_CUBIC
            )
            ew = np.expand_dims(edge_map * 0.4, -1)
            current = up + residual * ew
        else:
            current = blended

        if i == num_stages - 1:
            g = cv2.cvtColor(
                current.astype(np.uint8), cv2.COLOR_RGB2GRAY
            )
            if np.var(cv2.Laplacian(g, cv2.CV_64F)) < 180:
                blur = cv2.GaussianBlur(current, (0, 0), 2.0)
                current = current + 0.3 * (current - blur)

    return Image.fromarray(
        np.clip(current, 0, 255).astype(np.uint8)
    )


def apply_hcp(
    generated_image: Image.Image,
    content_reference: Image.Image,
    protection_strength: float = 0.8,
) -> Image.Image:
    gen = np.array(generated_image).astype(np.float32)
    ref = np.array(content_reference).astype(np.float32)

    if ref.shape[:2] != gen.shape[:2]:
        ref = cv2.resize(ref, (gen.shape[1], gen.shape[0]))

    gray_ref = cv2.cvtColor(ref.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    sal = cv2.normalize(
        cv2.GaussianBlur(
            np.sqrt(
                cv2.Sobel(gray_ref, cv2.CV_64F, 1, 0, 3) ** 2
                + cv2.Sobel(gray_ref, cv2.CV_64F, 0, 1, 3) ** 2
            ),
            (15, 15),
            0,
        ),
        None,
        0,
        1,
        cv2.NORM_MINMAX,
    )

    grad_var = np.var(sal)
    radius = HCP_PROTECTION_RADIUS * (
        0.7
        + 0.6 * np.clip(1.0 - grad_var / 0.1, 0, 1)
    )

    diff = np.mean(np.abs(gen - ref)) / 255.0
    adapt_strength = protection_strength * (
        1.0 - np.clip(diff * 0.3, 0, 0.4)
    )

    result_channels = []

    for c in range(3):
        gen_fft = np.fft.fftshift(np.fft.fft2(gen[:, :, c]))
        ref_fft = np.fft.fftshift(np.fft.fft2(ref[:, :, c]))

        rows, cols = gen.shape[:2]
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        dist = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
        norm_dist = dist / np.sqrt(crow ** 2 + ccol ** 2)

        base = np.exp(-norm_dist / radius)

        sal_fft = np.abs(
            np.fft.fftshift(np.fft.fft2(sal))
        )
        sal_w = cv2.normalize(sal_fft, None, 0, 0.3, cv2.NORM_MINMAX)

        mask = np.clip((base + sal_w) * adapt_strength, 0, 1)

        edge_fft = np.abs(
            np.fft.fftshift(
                np.fft.fft2(
                    cv2.Canny(gray_ref, 50, 150).astype(np.float32)
                )
            )
        )
        edge_boost = cv2.normalize(
            edge_fft, None, 0, 0.15, cv2.NORM_MINMAX
        )

        atten = HCP_ATTENUATION_FACTOR * (1.0 - edge_boost)

        result_fft = (
            gen_fft * mask
            + ref_fft * (1.0 - mask) * (1.0 - atten * norm_dist)
        )

        result = np.fft.ifft2(
            np.fft.ifftshift(result_fft)
        ).real
        result_channels.append(result)

    out = np.stack(result_channels, axis=-1)
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = cv2.bilateralFilter(out, 5, 15, 15)
    return Image.fromarray(out)
