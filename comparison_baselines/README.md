# Comparison Baselines

This folder contains 4 baseline methods for comparison:

1. **IP-Adapter** - Original IP-Adapter for SDXL
2. **InstantStyle** - Style-preserving generation with attention masking
3. **StyleStudio** - Text-driven style transfer with selective control
4. **CSGO** - Content-Style composition in text-to-image generation

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download model weights (see download_models.sh)
bash download_models.sh
```

## Run Comparison

```bash
# Run unified comparison
python ../run_comparison.py
```

## Folder Structure

```
comparison_baselines/
├── IP-Adapter-/       # Original IP-Adapter
├── InstantStyle/      # InstantStyle method  
├── StyleStudio/       # StyleStudio method
├── CSGO/              # CSGO method
└── README.md          # This file
```

