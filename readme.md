# Enhancing Empathetic Interaction: A DreamBooth-Based Approach for Personalized Visual Therapy

**Authors:**
- **Shweta Sharma**  
  Department of Computer Science, Drexel University, Philadelphia, PA, USA  
  [ss5792@drexel.edu](mailto:ss5792@drexel.edu)
- **Khushboo Patel**  
  Department of Computer Science, Drexel University, Philadelphia, PA, USA  
  [kp3329@drexel.edu](mailto:kp3329@drexel.edu)
- **Milan Varghese**  
  Department of Computer Science, Drexel University, Philadelphia, PA, USA  
  [mv644@drexel.edu](mailto:mv644@drexel.edu)

---

## Overview

This repository illustrates a **DreamBooth-based** workflow for fine-tuning **Stable Diffusion** to generate personalized, context-sensitive images for **visual therapy**. It combines:
1. **Diffusion-based Generative Modeling**
2. **Latent Space Compression**
3. **Natural Language Conditioning**
4. **U-Net–based Denoising**

A **Therapist Agent** acts as a conversational interface, providing empathetic text interactions and automatically creating text prompts to steer the model. **Google Colab** is the recommended environment for easier GPU access and streamlined setup.

---

## Repository Contents

- **`downloader.ipynb`**  
  A notebook to download required project files into Colab or your local environment.

- **`Final Code.ipynb`**  
  The main notebook that:
  - Installs dependencies
  - Sets up DreamBooth fine-tuning with Hugging Face **diffusers** and **accelerate**
  - Includes evaluation (CLIPScore, SSIM)
  - Demonstrates inference using your fine-tuned model

- **`requirements.txt`**  
  Python libraries needed for this project.

- **`backend.py`**  
  Implements the `StableDiffusionBackend` class for image inference and evaluation.

- **`therapist_agent.py`**  
  Implements the `TherapistAgent` class, which leverages GPT (via the OpenAI API) for empathetic text responses and prompt generation.

- **`app.py`**  
  A **Streamlit** web application that combines the therapist agent and the Stable Diffusion backend for real-time interaction.

---

## Quick Start Guide

### 1. Setup on Google Colab

1. **Clone or Download** the repository:
   - Use `downloader.ipynb` to pull project files or clone this repo into Colab.
2. Ensure you’re using a **GPU runtime**:
   - Go to **Runtime** → **Change runtime type** and set **Hardware accelerator** to **GPU**.
3. **Install Dependencies**:
   ```bash
   !pip install -r requirements.txt
   ```
4. (Optional) Validate CUDA:
   ```python
   import torch
   print("GPU Available:", torch.cuda.is_available())
   ```

### 2. DreamBooth Fine-Tuning

1. Open **`Final Code.ipynb`** in Colab.
2. Run the cells sequentially:
   - **Mount** Google Drive (if desired) for storing logs and model checkpoints.
   - Configure parameters such as `instance_prompt` (e.g., `<bruno>`), `learning_rate`, and `max_train_steps`.
   - Ensure that the file paths to the dataset and model directories are correctly updated everywhere in the notebook.
   - Execute the training loop with Hugging Face's **Accelerator**.
3. The notebook will save your fine-tuned model to the specified output directory.

### 3. Inference Example

After fine-tuning, test inference in **`Final Code.ipynb`** or another script:

```python
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

# Load your fine-tuned model
scheduler = DPMSolverMultistepScheduler.from_pretrained(
    "path/to/fine-tuned-model", subfolder="scheduler"
)
pipe = StableDiffusionPipeline.from_pretrained(
    "path/to/fine-tuned-model",
    scheduler=scheduler,
    torch_dtype=torch.float16
).to("cuda")

# Generate an image
prompt = "a photo of <bruno> having a picnic in the park"
image = pipe(prompt, guidance_scale=7.5).images[0]
image.show()
```

Ensure that the file path to your fine-tuned model directory is correctly set in all relevant cells or scripts.

---

## Streamlit Web Application

1. **Prepare the Code Files:**  
   Ensure that `backend.py`, `therapist_agent.py`, and `app.py` are present in your environment. If not, run the corresponding cells in your notebook to create these files.

2. **Launch the App:**  
   In a Colab cell, run:
   ```bash
   !streamlit run /content/app.py &>/content/logs.txt & npx localtunnel --port 8501 & curl https://loca.lt/mytunnelpassword
   ```
   - This command starts Streamlit on port 8501 and exposes it via LocalTunnel. A URL will be provided—open it in your browser.
3. **Interact with the App:**  
   - Fill out the sidebar form (Name, Age, Occupation, Pet Keyword).
   - Enter your current mood or any message and click **Chat**.
   - The app will display an empathetic text response and generate a custom image from your fine-tuned model.

---

## Evaluation

- **CLIPScore:**  
  Evaluates semantic alignment between the text prompt and the generated image.
- **SSIM:**  
  Measures structural similarity between images (if a reference is available).

These metrics are integrated into **`Final Code.ipynb`** for convenience.

---

## Contributing

We welcome improvements, including:
- Bug fixes and feature requests
- Enhancements to training or inference pipelines
- Documentation and deployment improvements

Please open an issue or submit a pull request with your suggestions.

---

## Acknowledgments

- **Hugging Face** for providing `diffusers` and DreamBooth implementations.
- **OpenAI** for powering the GPT-based text generation in our therapist agent.
- **Streamlit** for the interactive web application framework.
- **Drexel University** for the academic environment and resources that supported this project.

---

**Thank you for exploring our DreamBooth-Based approach for Personalized Visual Therapy!**  
Feel free to reach out with any questions or suggestions.

