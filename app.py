from potassium import Potassium, Request, Response
import clip
import json
from run import encode_image, encode_text, get_similarity
import requests
import base64
from io import BytesIO

app = Potassium("clip")

@app.init
def init():
    model,_ = clip.load("ViT-B/32", device="cuda:0")

    context = {
        "model": model
    }

    return context

@app.handler()
def handler(context: dict, request: Request) -> Response:

    model = context.get("model")
    prompt = request.json.get("prompt")

    image_byte_string = prompt.get('imageByteString', None)
    image_url = prompt.get('imageURL', None)
    text = prompt.get('text', None)
    texts = prompt.get('texts', None)


    if image_byte_string == None and image_url == None:
        return Response(
            json={ "Error": "No image provided"},
            status=400
        )
    
    if text == None and texts ==  None:
        return Response(
            json={ "Error": "No text/texts provided"},
            status=400
        )
    
    image_bytes = None

    if image_url != None:
        response = requests.get(image_url)
        image_bytes = BytesIO(response.content)

    if image_byte_string != None:
        image_encoded = image_byte_string.encode('utf-8')
        image_bytes = BytesIO(base64.b64decode(image_encoded))

    
    image_encoding = encode_image(image_bytes,model)

    response = {}

    if texts != None:
        sims = []
        for t in texts:
            text_encoding = encode_text(t,model)
            sim = get_similarity(text_encoding, image_encoding)
            sims.append(sim)
            response["similarities"] = sims

    if text != None:
        text_encoding = encode_text(text,model)
        sim = get_similarity(text_encoding, image_encoding)
        response['similarity'] = sim

    print(type(response))

    Response(
        json=json.dump(response),
        status=200
    )

if __name__ == "__main__":
    app.serve()