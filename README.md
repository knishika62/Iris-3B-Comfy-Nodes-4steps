# Iris-3B Comfy Nodes (4steps)

ComfyUI nodes for [Iris-3B](https://huggingface.co/speridlabs/iris-3b), a 3B pixel-space text-to-image model from Sperid Labs, plus its 4x upscaler and depth model. Iris has no VAE, so everything goes straight to IMAGE.

**This fork adds the 4-step Turbo LoRAs** ([pentacoxian-dev/iris-3b-turbo-lora](https://huggingface.co/pentacoxian-dev/iris-3b-turbo-lora)): merge one from `models/loras` and generate a 1024x1024 image in **4 forward passes**, guidance baked in. On the card below that's about 1.5 s instead of 54 s (30 steps). [Turbo details.](#turbo-4-step)

**Fast:** about 1.5x faster than the other Iris node pack on a fresh start, and 4x to 9x faster on repeat runs on a 16 GB card, using about 5 GB less VRAM. [Numbers below.](#speed)

<p align="center">
  <img src="assets/fox_girl_4x.jpg" width="49%" alt="fox girl holding a sign, upscaled 4x">
  <img src="assets/bottle.png" width="49%" alt="galaxy in a bottle">
</p>

<p align="center">
  <img src="assets/fox_girl_depth.png" width="49%" alt="depth map of the fox girl, grayscale">
  <img src="assets/bottle_depth.png" width="49%" alt="depth map of the bottle, inferno colors">
</p>

Top left: 832x1216 at 50 steps, then the 4x upscaler (3328x4864). Top right: ComfyUI's classic bottle prompt. Bottom: their depth maps from the Iris depth model, grayscale and inferno. Every PNG here has its workflow embedded, so drag [`fox_girl.png`](assets/fox_girl.png), [`bottle.png`](assets/bottle.png), [`fox_girl_depth.png`](assets/fox_girl_depth.png) or [`bottle_depth.png`](assets/bottle_depth.png) into ComfyUI to get the exact graph and settings.

## Install

Search for **Iris-3B 4steps** in ComfyUI-Manager, or clone into `ComfyUI/custom_nodes`:

```
git clone https://github.com/knishika62/Iris-3B-Comfy-Nodes-4steps
```

Nothing else to install. It needs transformers 4.57 or newer, which recent ComfyUI already has.

### Don't install it next to the base Iris-3B pack

This pack is a superset of [Iris-3B-Comfy-Nodes](https://github.com/bani4kaskashka/Iris-3B-Comfy-Nodes): it ships the same nodes plus the Turbo ones. Keep only one of the two in `custom_nodes`. Both vendor a package named `iris3b`, and Python keeps whichever one was imported first, so with both installed the one that loads second fails at startup with:

```
ModuleNotFoundError: No module named 'iris3b.lora'
Cannot import .../custom_nodes/Iris-3B-Comfy-Nodes-4steps module for custom nodes
```

The file is there; the other pack's `iris3b` (which has no `lora.py`) simply won the name. To check which one wins, run this from the ComfyUI folder:

```
python3 -c "
import sys, importlib.util
sys.path.insert(0, 'custom_nodes/Iris-3B-Comfy-Nodes-4steps/vendor')
print(importlib.util.find_spec('iris3b').origin)"
```

Fix: disable or remove the other pack (this one covers everything it does), then restart ComfyUI.

## Workflows

Download one and drag it into ComfyUI, or open it from Templates once the pack is installed.

- [Iris-3B Text to Image (Turbo 4 steps)](example_workflows/Iris-3B%20Text%20to%20Image%20%28Turbo%204%20steps%29.json): the few-step path, 4 forwards per image
- [Iris-3B Text to Image (Simple)](example_workflows/Iris-3B%20Text%20to%20Image%20%28Simple%29.json): resolution and quality presets
- [Iris-3B Text to Image (Advanced)](example_workflows/Iris-3B%20Text%20to%20Image%20%28Advanced%29.json): every sampler setting
- [Iris-3B Image to Image](example_workflows/Iris-3B%20Image%20to%20Image.json): restyle or rework an image you load
- [Iris-3B Upscale 4x](example_workflows/Iris-3B%20Upscale%204x.json): 4x upscale of any image
- [Iris-3B Depth](example_workflows/Iris-3B%20Depth.json): depth map of any image, plus a 3D view you can orbit

Both text-to-image workflows end with a 4x upscale group (1024 to 4096) that's bypassed by default. Select the group and press Ctrl+B to turn it on.

## Models

When a workflow opens with a model missing, ComfyUI lists it under Missing Models with a Download button. The loaders also download anything still missing on first run, so you only fetch what your workflow uses (about 44 GB for everything), so the workflows run as-is either way.

```
models/diffusion_models/iris-3b.safetensors            text to image (12 GB)
models/diffusion_models/iris-3b-upscaler.safetensors   4x upscaler (12 GB)
models/diffusion_models/iris-3b-depth.safetensors      depth (12 GB)
models/loras/iris-3b-turbo-r2-1000.safetensors         4-step Turbo adapter (460 MB, optional)
models/loras/iris-3b-turbo-r1-1500.safetensors         first-run adapter (460 MB, optional)
models/text_encoders/Qwen3-VL-4B-Instruct/             text encoder, the whole HF repo (8.3 GB)
```

To get them yourself, the three Iris files are `model.safetensors`, `upscaler/model.safetensors` and `depth/model.safetensors` from [speridlabs/iris-3b](https://huggingface.co/speridlabs/iris-3b), renamed as above. The configs come with this pack. The Turbo adapters are `r2-1000/student.safetensors` and `r1-1500/student.safetensors` from [pentacoxian-dev/iris-3b-turbo-lora](https://huggingface.co/pentacoxian-dev/iris-3b-turbo-lora), renamed as above. Any LoRA-named Turbo adapter also works: the loaders list every file in `models/loras` (subfolders included) whose metadata marks it a Turbo adapter, so a Hub-layout `models/loras/r2-1000/student.safetensors` shows up under that name. The auto-download names are just what the loaders fetch when nothing else is there.

```
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir models/text_encoders/Qwen3-VL-4B-Instruct
```

The text encoder has to be the original HF folder. Single-file or fp8 Qwen3-VL text encoders won't work, because Iris reads 12 hidden layers from it.

## Nodes

- **Load Iris-3B Model**: loads the DiT. `bf16` uses 6 GB of VRAM. `fp32` is the shipped precision but needs a lot more than 16 GB. Optional `turbo_lora` and `lora_strength` merge a Turbo adapter from `models/loras` into the weights at load time.
- **Load Iris Turbo LoRA**: merge a Turbo adapter into an already loaded model, so one model load covers several adapters. The weights change in place, so connect this node's output, not the loader's.
- **Load Iris LoRA (ai-toolkit)**: merge a LoRA trained with [ai-toolkit](https://github.com/ostris/ai-toolkit) (`arch: iris3b`) into a loaded model, read directly: its PEFT keys (`diffusion_model.*.lora_A/lora_B.weight`) are renamed on load, the rank is inferred, and strength 1.0 matches the trained strength (ai-toolkit trains with alpha = rank). Drop the `<name>_LoRA_<step>.safetensors` file in `models/loras` (subfolders included) and it's listed. Layers this Iris model lacks are skipped with a warning. Merge it before any Turbo LoRA: like the Turbo merge it is permanent.
- **Load Iris Text Encoder (Qwen3-VL-4B)**: lists HF folders in `text_encoders`.
- **Iris Text Encode**: positive and negative prompt. Leave the negative empty to use the null prompt the model was trained with.
- **Iris Turbo Sampler (4 steps)**: the few-step sampler for a model loaded with a Turbo LoRA. Mode, resolution, image strength, seed and batch size, no steps or cfg: those come from the adapter's own metadata (4 steps, shift 4, guidance baked in).
- **Iris Sampler (Simple)**: mode (text to image or image to image), resolution (1 MP sizes from 21:9 to 9:21, or match the input image), quality (fast = 30 steps, balanced = 50, best = 100), prompt adherence (cfg 2, 3 or 4.5), image strength (denoise 0.3, 0.5 or 0.7), seed and batch size.
- **Iris Sampler (Advanced)**: every setting: steps, cfg, seed, batch size, mode, denoise, shift, solver order, CFG interval, any width/height in multiples of 16, and keep_model_on_gpu. It also takes an optional `turbo_lora` to merge, though for a Turbo model you'll want the Turbo Sampler.
- **Load Iris Upscaler** and **Iris Upscale**: Iris fine-tuned for restoration and 4x upscaling. It's a single pass per 1024 px tile with overlapping tiles and a wavelet color fix. It works on any IMAGE, not just Iris output. With `limit_input_size` on, big inputs are shrunk first (short side 512, long side 1024), so the output stays at or under 2048x4096. Turn it off for a full 4x of any size.
- **Load Iris Depth Model** and **Iris Depth**: Iris fine-tuned for monocular depth, one pass per image. Outputs a depth IMAGE (near = white, ready for depth ControlNets, or an inferno colormap) and the same map as a MASK. It's relative depth, not metric, scaled to each image's 2nd to 98th percentile. `max_side` caps the size the model sees (default 1024).
- **Iris Depth to 3D**: turns the depth into a colored point cloud or mesh (`.ply` in your output folder). Connect it to ComfyUI's **Preview 3D** to orbit around it. `fov` is the photo's assumed field of view.

Both base samplers use DPM-Solver++ on the flow schedule. The defaults match the model card (cfg 3, shift 4). Live previews follow ComfyUI's preview setting and cost nothing, since Iris predicts pixels directly.

### Turbo (4 step)

The [Turbo adapters](https://huggingface.co/pentacoxian-dev/iris-3b-turbo-lora) are DMD2-distilled LoRAs: with one merged, Iris makes a 1024x1024 image in 4 steps, one forward pass each, because the CFG 3 guidance is baked in. They're plain LoRA files, so they live in `models/loras` like any other LoRA, subfolders included, and the loaders list everything there that carries the adapter metadata. `iris-3b-turbo-r2-1000.safetensors` (recommended) and `iris-3b-turbo-r1-1500.safetensors` are downloaded there on first use.

Two nodes:

1. **Load Iris-3B Model** with `turbo_lora` set (or **Load Iris Turbo LoRA** after the loader). The adapter is folded into the weights, so a step costs exactly a base forward pass.
2. **Iris Turbo Sampler (4 steps)** instead of the DPM++ samplers. Steps (4) and shift (4) come from the adapter's safetensors metadata, not from widgets.

The Turbo loop is not DPM++ with 4 steps. It runs the distilled schedule t = 1.0, 0.923, 0.8, 0.571 and re-noises the predicted clean image with fresh noise between steps, which is how the student was trained; running a Turbo model through DPM++ at 30 steps with CFG gives washed-out, double-guided results (the Advanced sampler logs a warning if you do). Image to image works too: the image is noised to a grid time and the last few student steps run over it.

Notes from the adapter card: distilled at about 1 MP, so stay near 1024x1024 and its aspect ratios; variety is narrower than the base sampler's; typography is weak. `r1-1500` occasionally merges subjects at some seeds.

### Image to image

Set the sampler's mode to image to image and connect an IMAGE. The image is noised partway and diffused back, so lower strength keeps more of it. It takes the same time as a normal run with the same steps.

Two things about the strength settings, true of the base pack too:

**With a resolution preset, the image is scaled to cover and center-cropped.** A 16:9 input into 1:1 (1024x1024) loses about 40% of its width. Use `match input image` to keep the framing, or pick a preset near the input's aspect ratio.

**The presets are coarse and the base sampler keeps the step count, not the compute.** Steps stays at 30/50/100 and the grid is stretched to `steps / denoise`, so subtle, medium and strong all take the same time and land on the same start times (t = 0.63, 0.80, 0.90), and 0.5 and 0.7 look close. The Advanced sampler's free `denoise` slider gives finer control. On the Turbo sampler the effect is real but harsh: strength picks how many of the 4 student steps run over the image (0.3 = 1 step, 0.5 = 2, 0.7 = 3), and none of them starts at pure noise, which is where the distilled student is at its best. Turbo img2img is inherently softer than its text to image; for a quality restyle, generate text to image and upscale instead.

Source, then subtle (0.3), medium (0.5) and strong (0.7), prompted as an oil painting:

![source, then denoise 0.3, 0.5 and 0.7](assets/image_to_image.jpg)

### Depth

![image and its Iris depth map](assets/depth.jpg)

The Depth workflow also builds a 3D point cloud and opens it in Preview 3D. Click Fit to Viewer, then drag to orbit and scroll to zoom. Photo, depth and the 3D view:

![photo, depth map and 3D point cloud](assets/depth_3d.jpg)

### Upscale

Bicubic on the left, Iris 4x on the right:

![bicubic vs Iris 4x](assets/upscale.jpg)

## Speed

Measured on an RX 9060 XT (16 GB, ROCm on Windows), 1 MP images, against [the other Iris node pack](https://github.com/envy-ai/ComfyUI-Iris) with its default settings:

| | This pack | Other pack |
|---|---|---|
| First run | 1.8 s/step | 2.8 s/step |
| Second run | 1.8 s/step | 7.3 s/step |
| Third run | 1.8 s/step | 16.4 s/step |
| Peak VRAM | 7.7 GB | 12.5 GB |

Each model here visits the GPU only for its own step and sits in system RAM otherwise, so nothing piles up and spills over on a 16 GB card. Expect to need around 22 GB of free system RAM with all three models loaded. 4x upscaling 1024 to 4096 is 49 tiles, about 47 s.

With a Turbo adapter the same card does 4 forwards instead of 30: at 1.8 s/step that's about 7 s per 1024x1024 image end to end, against about 54 s at the fast preset. The upstream fork measured 0.68 s/image on an RTX 5090 and 10-11 s on an M3 Max.

On AMD, start ComfyUI the normal way (`main.py`), which turns on the AOTriton flash attention kernels. Without them attention falls back to a slow path.

## Credits

The model code in `vendor/iris3b` is from [speridlabs/iris-3b](https://github.com/speridlabs/iris-3b) (Apache-2.0, see `vendor/LICENSE` and `vendor/NOTICE`). Only the inference parts are included. `vendor/iris3b/lora.py` comes from the [pentaCoxian/iris-3b-turbo](https://github.com/pentaCoxian/iris-3b-turbo) fork (Apache-2.0), which also trained the [Turbo adapters](https://huggingface.co/pentacoxian-dev/iris-3b-turbo-lora) this pack runs. This pack is also Apache-2.0.
