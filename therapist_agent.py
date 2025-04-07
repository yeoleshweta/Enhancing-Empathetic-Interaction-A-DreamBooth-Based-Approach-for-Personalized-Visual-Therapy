import os
from openai import OpenAI
from backend import StableDiffusionBackend

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class TherapistAgent:
    def __init__(self, sd_backend: StableDiffusionBackend):
        self.sd_backend = sd_backend
        self.chat_system_prompt = (
            "You are a compassionate therapist chatbot helping users manage stress, anxiety, "
            "or sadness through empathetic and concise supportive conversation."
        )
        self.image_system_prompt = (
            "You generate very short, simple, and clear prompts for image generation involving the user's pet (named <bruno>). "
            "Examples: 'A photo of <bruno> playing in the garden', 'A cute picture of <bruno> sleeping.'"
        )

    def chat(self, user_message):
        # Generate empathetic text response
        chat_messages = [
            {"role": "system", "content": self.chat_system_prompt},
            {"role": "user", "content": user_message}
        ]

        chat_response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=chat_messages,
            temperature=0.8
        ).choices[0].message.content.strip()

        # Separately generate short and simple image prompt
        image_messages = [
            {"role": "system", "content": self.image_system_prompt},
            {"role": "user", "content": user_message}
        ]

        image_prompt_response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=image_messages,
            temperature=0.6,
            max_tokens=25
        ).choices[0].message.content.strip()

        generated_image = self.sd_backend.generate_image(image_prompt_response)

        return {
            "response_text": chat_response,
            "image_prompt": image_prompt_response,
            "generated_image": generated_image
        }
