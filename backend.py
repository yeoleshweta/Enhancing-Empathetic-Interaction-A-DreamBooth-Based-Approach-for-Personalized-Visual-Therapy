import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity

class StableDiffusionBackend:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16
        ).to(self.device)

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def generate_image(self, prompt, num_inference_steps=100, guidance_scale=7.5):
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
        return image

    def calculate_clip_score(self, image, prompt):
        inputs = self.clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(self.device)
        outputs = self.clip_model(**inputs)
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1)
        return probs[0][0].item() * 100

    def calculate_ssim(self, real_image, generated_image):
        real_np = np.array(real_image.resize((512, 512))).astype(np.float32) / 255.0
        gen_np = np.array(generated_image.resize((512, 512))).astype(np.float32) / 255.0
        ssim_score = structural_similarity(real_np, gen_np, channel_axis=-1, data_range=1.0)
        return ssim_score * 100

    def evaluate(self, prompt, real_image_path):
        generated_image = self.generate_image(prompt)
        real_image = Image.open(real_image_path).convert("RGB")
        clip_score = self.calculate_clip_score(generated_image, prompt)
        ssim_score = self.calculate_ssim(real_image, generated_image)

        return {
            'generated_image': generated_image,
            'clip_score': clip_score,
            'ssim_score': ssim_score
        }
