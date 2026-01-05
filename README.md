### Requirements

- Python 3.8 or newer  
- PyTorch 2.1 or newer  
- CUDA-capable GPU recommended  

### Environment Setup

```bash
conda create -n freq_style python=3.10
conda activate freq_style
pip install -r requirements.txt


### Environment Setup
python infer_SemanticFusion.py \
    --prompt "A red car" \
    --style_path ./style.jpg \
    --output_path ./output.jpg


