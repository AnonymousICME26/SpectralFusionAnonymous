import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import Dict, List, Optional

from .config import (
    COLOR_STD_NORMALIZER,
    TEXTURE_VAR_NORMALIZER,
    EDGE_THRESHOLD_LOW,
    EDGE_THRESHOLD_HIGH,
    WEAK_STYLE_THRESHOLD,
    STRONG_STYLE_THRESHOLD,
    DEFAULT_CONTRAST_FACTOR,
    DEFAULT_SATURATION_BOOST,
    EPSILON,
    MULTISCALE_WEIGHTS,
)
from .utils import resize_image_hq, load_image_safe


def analyze_style_intensity(style_image: Image.Image) -> Dict:
    img_array = np.array(style_image)

    color_std = np.std(img_array, axis=(0, 1)).mean()
    color_intensity = min(color_std / COLOR_STD_NORMALIZER, 1.0)

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, EDGE_THRESHOLD_LOW, EDGE_THRESHOLD_HIGH)
    edge_density = np.sum(edges > 0) / edges.size

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_complexity = min(
        np.var(laplacian) / TEXTURE_VAR_NORMALIZER, 1.0
    )

    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)

    h, w = magnitude_spectrum.shape
    high_freq_energy = np.sum(
        magnitude_spectrum[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    )
    total_energy = np.sum(magnitude_spectrum)
    freq_ratio = high_freq_energy / (total_energy + EPSILON)

    intensity = (
        color_intensity * 0.3
        + edge_density * 0.25
        + texture_complexity * 0.25
        + freq_ratio * 0.2
    )

    if intensity < WEAK_STYLE_THRESHOLD:
        recommended_scale = 1.3
        recommended_steps = 60
        recommended_cfg = 6.0
        guidance = ""
    elif intensity > STRONG_STYLE_THRESHOLD:
        recommended_scale = 0.8
        recommended_steps = 100
        recommended_cfg = 4.0
        guidance = ""
    else:
        recommended_scale = 1.0
        recommended_steps = 75
        recommended_cfg = 5.0
        guidance = ""

    return {
        "intensity": intensity,
        "color_strength": color_intensity,
        "detail_level": edge_density,
        "texture_complexity": texture_complexity,
        "frequency_ratio": freq_ratio,
        "recommended_scale": recommended_scale,
        "recommended_steps": recommended_steps,
        "recommended_cfg": recommended_cfg,
        "guidance": guidance,
    }


def enhance_style_image(
    style_image: Image.Image,
    enhancement_factor: float = DEFAULT_CONTRAST_FACTOR,
) -> Image.Image:
    img_array = np.array(style_image)

    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    l_global = cv2.convertScaleAbs(
        l, alpha=enhancement_factor * 1.15, beta=0
    )
    l_final = cv2.addWeighted(l_enhanced, 0.7, l_global, 0.3, 0)

    a_boosted = cv2.convertScaleAbs(a, alpha=1.1, beta=0)
    b_boosted = cv2.convertScaleAbs(b, alpha=1.1, beta=0)

    lab = cv2.merge([l_final, a_boosted, b_boosted])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.5)
    unsharp = cv2.addWeighted(enhanced, 1.6, gaussian, -0.6, 0)
    sharpened = np.clip(unsharp, 0, 255).astype(np.uint8)

    pil_image = Image.fromarray(sharpened)
    enhancer = ImageEnhance.Color(pil_image)
    saturated = enhancer.enhance(DEFAULT_SATURATION_BOOST * 1.1)

    enhancer = ImageEnhance.Sharpness(saturated)
    return enhancer.enhance(1.3)


def load_multiscale_style(
    image_path: str,
    sizes: Optional[List[int]] = None,
) -> Dict[int, Image.Image]:
    if sizes is None:
        sizes = [512, 768, 1024]

    scales = {}

    try:
        img_array = load_image_safe(image_path)
        if img_array is None:
            raise ValueError()

        for size in sizes:
            try:
                img_resized = resize_image_hq(img_array, size)
                scales[size] = Image.fromarray(img_resized)
            except Exception:
                pass

        if not scales:
            raise ValueError()

    except Exception:
        return {}

    return scales


def blend_multiscale_features(
    scales: Dict[int, Image.Image],
    weights: Optional[Dict[int, float]] = None,
) -> Image.Image:
    if weights is None:
        weights = MULTISCALE_WEIGHTS

    if not scales:
        raise ValueError()

    total_weight = sum(
        weights.get(size, 1.0 / len(scales)) for size in scales
    )
    target_size = max(scales.keys())

    blended = None

    for size, img in scales.items():
        weight = weights.get(size, 1.0 / len(scales)) / total_weight
        img_array = np.array(img).astype(np.float32)

        if size != target_size:
            img_array = resize_image_hq(
                img_array.astype(np.uint8), target_size
            ).astype(np.float32)

        if blended is None:
            blended = img_array * weight
        else:
            blended += img_array * weight

    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def get_adaptive_style_schedule(
    base_scale: float,
    num_steps: int,
    schedule_type: str = "progressive",
) -> List[float]:
    from .config import (
        SCHEDULE_EARLY_PHASE,
        SCHEDULE_MIDDLE_PHASE,
        EARLY_STYLE_BOOST,
        LATE_STYLE_REDUCTION,
    )

    schedule = []

    for step in range(num_steps):
        progress = step / max(num_steps - 1, 1)

        if schedule_type == "progressive":
            if progress < SCHEDULE_EARLY_PHASE:
                scale = base_scale * EARLY_STYLE_BOOST
            elif progress < SCHEDULE_MIDDLE_PHASE:
                scale = base_scale
            else:
                scale = base_scale * LATE_STYLE_REDUCTION

        elif schedule_type == "cosine":
            scale = base_scale * (
                0.5 + 0.5 * np.cos(np.pi * progress)
            )
            scale = max(scale, base_scale * 0.5)

        elif schedule_type == "inverse":
            if progress < SCHEDULE_EARLY_PHASE:
                scale = base_scale * 0.7
            elif progress < SCHEDULE_MIDDLE_PHASE:
                scale = base_scale
            else:
                scale = base_scale * 1.2
        else:
            scale = base_scale

        schedule.append(scale)

    return schedule


class AdaptiveStyleController:
    def __init__(
        self,
        base_scale: float,
        num_steps: int,
        schedule_type: str = "progressive",
    ):
        self.base_scale = base_scale
        self.num_steps = num_steps
        self.schedule = get_adaptive_style_schedule(
            base_scale, num_steps, schedule_type
        )
        self.current_step = 0

    def get_scale(self, step: int) -> float:
        if step < len(self.schedule):
            return self.schedule[step]
        return self.base_scale

    def reset(self):
        self.current_step = 0
