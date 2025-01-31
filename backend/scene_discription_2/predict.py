from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
from gtts import gTTS
import os
import uuid
from googletrans import Translator  # Add this import

# Load the model, feature extractor, and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Configuration for text generation
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Initialize the translator
translator = Translator()

def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)

    # Add padding to handle images of different sizes
    pixel_values = feature_extractor(images=images, return_tensors="pt", padding=True).pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

def translate_text(text, dest_language="hi"):
    """
    Translate text to the specified language.

    Args:
        text (str): The text to translate.
        dest_language (str): The language code to translate to (e.g., "hi" for Hindi).

    Returns:
        str: The translated text.
    """
    try:
        translated = translator.translate(text, dest=dest_language)
        return translated.text
    except Exception as e:
        print(f"Translation failed: {e}")
        return text  # Return the original text if translation fails

def text_to_speech(text, output_folder="static", language="en"):
    """
    Convert text to speech and save it as an MP3 file.

    Args:
        text (str): The text to convert to speech.
        output_folder (str): The folder to save the audio file.
        language (str): The language code (e.g., "en" for English, "hi" for Hindi).

    Returns:
        str: The path to the saved audio file.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Generate a unique filename
    unique_id = uuid.uuid4().hex
    output_file = os.path.join(output_folder, f"output_{unique_id}.mp3")

    # Convert text to speech in the specified language
    tts = gTTS(text=text, lang=language)
    tts.save(output_file)
    return output_file