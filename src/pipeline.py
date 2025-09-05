import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from qdrant_client import QdrantClient
from openai import OpenAI
import os
from dotenv import load_dotenv
import streamlit as st

# -------------------------------
# Config
# -------------------------------
USE_GPU = True  # set to False to force CPU always

def get_device():
    if USE_GPU and torch.cuda.is_available():
        try:
            _ = torch.empty(1, device="cuda")  # test allocation
            return torch.device("cuda")
        except torch.cuda.OutOfMemoryError:
            print("⚠️ CUDA OOM → using CPU instead.")
            torch.cuda.empty_cache()
    return torch.device("cpu")

device = get_device()

# -------------------------------
# Model loading (cached for Streamlit)
# -------------------------------
@st.cache_resource
def load_models(device):
    # ResNet for embeddings
    resnet = models.resnet50(pretrained=True)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # remove fc
    resnet.eval().to(device)

    # BLIP for captions
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    blip = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16
    ).to(device)

    return resnet, processor, blip

model, processor_blip, model_blip = load_models(device)

# -------------------------------
# Preprocessing
# -------------------------------
preprocess = transforms.Compose([
    transforms.Resize(224),   # safe for ResNet
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# -------------------------------
# Embeddings
# -------------------------------
def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        try:
            embedding = model(input_tensor).squeeze().cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            print("⚠️ OOM in ResNet → retrying on CPU")
            torch.cuda.empty_cache()
            input_tensor = input_tensor.to("cpu")
            embedding = model.to("cpu")(input_tensor).squeeze().cpu().numpy()

    torch.cuda.empty_cache()
    return embedding

# -------------------------------
# Captions
# -------------------------------
def generate_caption_blip(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor_blip(images=image, return_tensors="pt")
    inputs = {k: v.to(device, dtype=torch.float16) for k, v in inputs.items()}

    with torch.no_grad():
        try:
            out = model_blip.generate(**inputs)
        except torch.cuda.OutOfMemoryError:
            print("⚠️ OOM in BLIP → retrying on CPU")
            torch.cuda.empty_cache()
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            out = model_blip.to("cpu").generate(**inputs)

    caption = processor_blip.decode(out[0], skip_special_tokens=True)
    torch.cuda.empty_cache()
    return caption

client = QdrantClient(host="localhost", port=6333)
collection_name = "flower_images"

# -------------------------------
# Qdrant
# -------------------------------
def get_top_neighbors(query_embedding, client, collection_name, top_n=10):
    """
    Get top nearest neighbors excluding the query image itself.

    Parameters:
        query_embedding: np.array
        client: QdrantClient
        collection_name: str
        top_n: number of neighbors to return
        query_filepath: str or None, path of the query image to exclude

    Returns:
        List of dicts: [{"filepath": ..., "caption": ...}, ...]
    """
    # Fetch extra neighbors in case the query image appears
    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=top_n + 5
    )

    neighbors = []

    for p in results[1:]:  # skip the first result (closest, likely query image)
        filepath = p.payload.get("filepath")
        caption = p.payload.get("caption", "")

        if filepath:  # skip invalid filepaths
            neighbors.append({"filepath": filepath, "caption": caption})

        if len(neighbors) >= top_n:
            break

    return neighbors


# -------------------------------
# OpenAI
# -------------------------------
load_dotenv()
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def query_gpt4(prompt, max_tokens=800):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful botanist."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()
