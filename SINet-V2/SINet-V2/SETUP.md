# Running SINet-V2 single-image inference on another laptop

The shared folder already contains everything: the model code (`lib/`), the
pretrained weights (`snapshot/SINet_V2/Net_epoch_best.pth`, ~103 MB), the
inference script (`infer_single.py`), and a sample image (`samples/gecko.jpg`).

Works on Mac (Apple Silicon MPS), NVIDIA GPU (CUDA), or plain CPU — the script
auto-picks the device.

## 1. Check the weights came across

```bash
cd /path/to/SINet-V2
ls -lh snapshot/SINet_V2/Net_epoch_best.pth   # should be ~103 MB
```

If it's missing or 0 bytes (some sync tools skip large files):

```bash
pip install gdown
mkdir -p snapshot/SINet_V2
python -m gdown "1D3RKQ8Nzd0ArV_c47StVKEuaoYTwnclR" -O snapshot/SINet_V2/Net_epoch_best.pth
```

## 2. Set up Python (3.9–3.12 recommended)

Use any one of these.

**conda:**
```bash
conda create -n sinet python=3.11 -y
conda activate sinet
pip install -r requirements.txt
```

**venv:**
```bash
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run it

```bash
python infer_single.py --image samples/gecko.jpg
```

Outputs land in `res/single/`:
- `gecko_mask.png`    — grayscale probability map (white = camouflaged object)
- `gecko_overlay.png` — red highlight over the original image

Run on your own image:
```bash
python infer_single.py --image /path/to/your_photo.jpg
```

## Notes / troubleshooting

- **Device** is printed at startup (`[device] mps|cuda|cpu`). CPU works too, just slower.
- **`weights_only` error on old torch:** the script uses `torch.load(..., weights_only=True)`. If you're on torch < 1.13 this arg doesn't exist — either upgrade torch, or remove that argument from `infer_single.py`.
- **No `timm` needed** — the model is self-contained; the backbone weights are inside the .pth file.
- The original `MyTesting.py` is **not** used here (it needs the full benchmark
  datasets and a removed `scipy.misc` API). Use `infer_single.py`.
