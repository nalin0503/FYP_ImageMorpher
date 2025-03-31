# Metamorph: Advanced Image Morphing Pipeline

<div align="center">
   <img src="img/metamorphLogo_nobg.png" alt="Metamorph Logo" width="200"/>
   <br>
   <br>
   
   [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3129/)
   [![Pages Build Deployment](https://img.shields.io/badge/pages--build--deployment-passing-brightgreen)](https://nalin0503.github.io/FYP_ImageMorpher/)
   [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/nalin0503/metamorph)
   [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
   
   <p align="center">
      <i>NTU Final Year Project on Advanced Image Morphing via Diffusion Models with Frame Interpolation for Large Motion (FILM)</i>
   </p>
   
   <div style="display: flex; justify-content: center; gap: 20px;">
      <video width="290" autoplay loop muted playsinline>
         <source src="img/interpolated.mp4" type="video/mp4">
         Your browser does not support the video tag.
      </video>
      <video width="290" autoplay loop muted playsinline>
         <source src="sample_runs/sample_run_1/film_output_20250327_102142/output_video_20250327_102705.mp4" type="video/mp4">
         Your browser does not support the video tag.
      </video>
   </div>
</div>

## 🌟 Overview

Metamorph is a state-of-the-art image morphing framework that combines diffusion-based generative models with frame interpolation techniques to create smooth, high-quality transitions between images. The system integrates:

- **DiffMorpher**: Creates keyframe sequences using latent space interpolation in diffusion models
- **LCM-LoRA**: Accelerates the diffusion process with almost no quality loss
- **FILM**: Performs advanced frame interpolation for ultra-smooth transitions
- **Web Interface**: User-friendly Streamlit application for easy morphing

<div align="center">
   <img src="assets/pipeline_diagram.png" alt="Metamorph Pipeline" width="800"/>
</div>

## 📁 Repository Structure

<div align="center">
   <img src="assets/repo_hierarchy.png" alt="Repository Hierarchy" width="700"/>
</div>

- **Makefile**: Build automation and dependency management
- **run_morphing.py**: Controller layer that orchestrates keyframe generation and interpolation
- **DiffMorpher/**: Submodule for keyframe generation
  - **main.py**: Entry point for LCM-LoRA integration and memory optimizations
  - **utils.py**: Contains the DiffMorpherPipeline and model definitions
- **FILM.py**: Implements recursive frame interpolation for smoothing keyframes
- **app.py**: UI layer (Streamlit) for interactive morphing

## ✨ Features

- **High-Quality Morphing**: Generate seamless transitions between any two images
- **Model Selection**: Choose from multiple diffusion models for different aesthetic results
- **Acceleration Options**: Toggle LCM-LoRA to significantly reduce processing time
- **Advanced Interpolation**: Apply recursive FILM processing for smoother results
- **Text Guidance**: Optional text prompts to guide the semantic transition
- **Improved Keyframes**: Enhanced with AdaIN and rescheduled sampling for better coherence
- **User-Friendly Interface**: Simple web application to control all parameters

## 🛠️ Installation

### Prerequisites

- Python 3.12
- NVIDIA GPU with CUDA support
- Git

### FILM Dependencies

As recommended by [FILM](https://github.com/google-research/frame-interpolation):
- Anaconda Python 3.9
- CUDA Toolkit 11.2.1
- cuDNN 8.1.0

> **Note**: If you encounter dependency conflicts, you may try: `conda install -c conda-forge cudatoolkit=11.8.0 cudnn=9.3.0`

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/username/FYP_ImageMorpher.git
   cd FYP_ImageMorpher
   ```

2. Initialize and update submodules:
   ```bash
   make init
   ```
   
   This will:
   - Initialize the DiffMorpher submodule
   - Synchronize submodule URLs
   - Update submodules to the latest commits
   - Install required Python packages

## 🚀 Usage

### Option 1: Web Interface (Recommended)

Run the Streamlit application:
```bash
streamlit run app.py
```

For NTU SLAB GPU cluster users, select the "Using SLAB GPU Cluster" option in the interface.

### Option 2: Direct CLI Command

Use `run_morphing.py` as the overall orchestration script:
```bash
python run_morphing.py \
  --image_path_0 ./assets/Trump.jpg \
  --prompt_0 "A photo of an American man" \
  --image_path_1 ./assets/Biden.jpg \
  --prompt_1 "A photo of an American man" \
  --output_path ./results/Trump_Biden \
  --use_adain \
  --use_reschedule \
  --save_inter \
  --num_frames 16 \
  --duration 100 \
  --use_film \
  --film_fps 30 \
  --film_num_recursions 3
```

### Option 3: Using Makefile

```bash
make morph
```

### Option 4: DiffMorpher Only (Without FILM)

Use the base DiffMorpher for keyframe generation only:
```bash
cd DiffMorpher
python main.py \
  --image_path_0 ./assets/Trump.jpg \
  --image_path_1 ./assets/Biden.jpg \
  --prompt_0 "A photo of an American man" \
  --prompt_1 "A photo of an American man" \
  --output_path "./results/Trump_Biden" \
  --use_adain \
  --use_reschedule \
  --save_inter \
  --use_lcm
```

For NTU SLAB GPU cluster users:
```bash
cd DiffMorpher
srun -p rtx3090_slab -w slabgpu05 --gres=gpu:1 \
  --job-name=test --kill-on-bad-exit=1 python main.py \
  --image_path_0 ./assets/Trump.jpg \
  --image_path_1 ./assets/Biden.jpg \
  --prompt_0 "A photo of an American man" \
  --prompt_1 "A photo of an American man" \
  --output_path "./results/Trump_Biden" \
  --use_adain \
  --use_reschedule \
  --save_inter \
  --use_lcm
```

## 📊 Components

### DiffMorpher

DiffMorpher leverages latent consistency models to create intermediate keyframes between source images. Enhanced with:

- **Adaptive Instance Normalization (AdaIN)**: Adjusts statistical properties of interpolated latent spaces
- **Reschedule Sampling**: Creates non-linear sampling schedules based on perceptual distances
- **LCM-LoRA Acceleration**: Reduces sampling steps while preserving quality

<div align="center">
   <img src="assets/keyframe_diagram.png" alt="DiffMorpher Keyframe Generation" width="700"/>
</div>

### FILM Interpolation

Frame Interpolation for Large Motion (FILM) is particularly effective at handling large displacements between frames:

- **Recursive Application**: Create exponentially more intermediate frames with each pass
- **Smooth Motion**: Handle large changes between frames that traditional interpolation struggles with
- **Detail Preservation**: Maintain fine details during the interpolation process

<div align="center">
   <img src="assets/film_diagram.png" alt="FILM Interpolation Process" width="700"/>
</div>

## 📖 Documentation

For detailed parameter explanations and usage guidelines, visit our [GitHub Pages documentation](https://username.github.io/FYP_ImageMorpher/).

## 🔗 Links

- [GitHub Repository](https://github.com/username/FYP_ImageMorpher)
- [DiffMorpher Submodule](https://github.com/username/DiffMorpher)
- [Hugging Face Demo](https://huggingface.co/spaces/username/metamorph)
- [FILM Repository](https://github.com/google-research/frame-interpolation)

## 📋 Citation

If you use Metamorph in your research, please cite our work:

```bibtex
@misc{author2025metamorph,
  title={Metamorph: Enhancing Image Morphing with Diffusion Models and Frame Interpolation},
  author={Author, A.},
  year={2025},
  howpublished={Nanyang Technological University, Final Year Project}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) for the foundation models
- [Google Research FILM](https://github.com/google-research/frame-interpolation) for the interpolation framework
- [Latent Consistency Models](https://github.com/lucidrains/latent-consistency-model) for acceleration techniques
- Nanyang Technological University for supporting this research