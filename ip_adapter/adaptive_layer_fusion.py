import torch
import torch.nn as nn
from typing import Dict


class AdaptiveLayerFusion:
    def __init__(self, total_layers=16):
        self.total_layers = total_layers
        self.structure_layers = list(range(0, 5))
        self.style_layers = list(range(5, 12))
        self.detail_layers = list(range(12, 17))

    def analyze_prompt_complexity(self, prompt: str) -> Dict[str, float]:
        tokens = prompt.split()
        token_length = len(tokens)

        prompt_lower = prompt.lower()

        comma_count = prompt.count(",")
        conjunction_count = sum(
            1
            for conj in [" and ", " or ", " with ", " plus "]
            if conj in prompt_lower
        )

        attribute_words = [
            "beautiful", "detailed", "intricate", "complex", "simple",
            "elegant", "modern", "ancient", "colorful", "vibrant", "dark",
            "bright", "vivid", "subtle", "bold", "soft", "sharp", "smooth",
            "rough", "glossy", "matte", "textured"
        ]

        attr_word_count = sum(
            1 for word in attribute_words if word in prompt_lower
        )

        n_attr = comma_count + conjunction_count + 0.5 * attr_word_count
        eta = 2.0

        raw_score = token_length + eta * n_attr
        complexity = raw_score / 70.0
        complexity = max(0.0, min(complexity, 1.0))

        return {
            "complexity": complexity,
            "token_length": token_length,
            "attr_count": n_attr,
            "raw_score": raw_score,
        }

    def get_adaptive_layers(
        self, prompt: str, style_strength: float = 1.0
    ) -> Dict[int, float]:
        analysis = self.analyze_prompt_complexity(prompt)
        complexity = analysis["complexity"]

        layer_weights = {}

        if complexity < 0.3:
            for l in self.structure_layers:
                layer_weights[l] = 0.0
            for l in self.style_layers:
                layer_weights[l] = style_strength * 1.2
            for l in self.detail_layers:
                layer_weights[l] = style_strength * 0.9

        elif complexity < 0.6:
            for l in self.structure_layers:
                layer_weights[l] = 0.0
            for l in self.style_layers:
                layer_weights[l] = style_strength * 1.0
            for l in self.detail_layers:
                layer_weights[l] = style_strength * 0.7

        else:
            for l in self.structure_layers:
                layer_weights[l] = 0.0
            for l in self.style_layers[:4]:
                layer_weights[l] = style_strength * 0.8
            for l in self.style_layers[4:]:
                layer_weights[l] = 0.0
            for l in self.detail_layers:
                layer_weights[l] = style_strength * 0.5

        return layer_weights

    def get_layer_info(
        self, prompt: str, style_strength: float = 1.0
    ) -> Dict[str, float]:
        analysis = self.analyze_prompt_complexity(prompt)
        weights = self.get_adaptive_layers(prompt, style_strength)

        active = [w for w in weights.values() if w > 0]
        avg_strength = (
            sum(active) / len(active) if active else 0.0
        )

        return {
            "complexity": analysis["complexity"],
            "token_length": analysis["token_length"],
            "attr_count": analysis["attr_count"],
            "active_layers": len(active),
            "avg_strength": avg_strength,
        }


def create_adaptive_layer_fusion():
    return AdaptiveLayerFusion()
