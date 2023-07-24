import clip

def download_model():
    model, transform = clip.load("ViT-B/32", device="cpu")

if __name__ == "__main__":
    download_model()