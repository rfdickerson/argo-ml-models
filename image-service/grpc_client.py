import logging
import os

import grpc.aio
import numpy as np
from torchvision import transforms

from tritonclient.grpc import service_pb2_grpc
from tritonclient.grpc import service_pb2

from fastapi import FastAPI, File, UploadFile

from PIL import Image

MODEL_NAME = 'skin'
WIDTH = 450
HEIGHT = 600
CHANNELS = 3
BATCH_SIZE = 1

app = FastAPI()

TRITON_URL = os.environ.get('TRITON_URL', 'localhost:8001')
FORMAT = "%(levelname)s:%(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

log = logging.getLogger("my-api")

log.debug("Connecting to Triton inference engine at %s", TRITON_URL)

channel = grpc.aio.insecure_channel(TRITON_URL)
stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)


def _bytes_to_numpy(raw):
    # convert bytes to a numpy array
    b = b''.join(list(raw))
    return np.frombuffer(b, dtype='float32')


def _make_request(img_data, id='42'):
    return service_pb2.ModelInferRequest(
        model_name=MODEL_NAME,
        id=id,
        inputs=[
            service_pb2.ModelInferRequest.InferInputTensor(
                name='image',
                datatype='FP32',
                shape=[BATCH_SIZE, CHANNELS, WIDTH, HEIGHT],
                contents=service_pb2.InferTensorContents(
                    # requires a flattened 1-d array
                    fp32_contents=img_data.flatten()
                )
            )
        ],
    )

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
def healthcheck():
    return {"status": "healthy"}

@app.post("/uploadfile/")
async def create_upload_file(image_file: UploadFile):
    result = await process_image(stub, image_file.file)
    return result


def preprocess(image_file) -> np.array:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    image_transforms = transforms.Compose([
        transforms.CenterCrop([WIDTH,HEIGHT]),
        transforms.ToTensor(),
        normalize,
    ])

    with Image.open(image_file) as im:
        return image_transforms(im).numpy()


def postprocess(results):

    results = np.exp(results)

    class_name = ['actinic keratosis',
                  'basal cell carcinoma',
                  'benign keratosis',
                  'dermatofibroma',
                  'melanoma',
                  'melanocytic nevi',
                  'vascular lesions',]

    c = np.argmax(results)

    return {
        "dx": class_name[c],
        "probabilities": dict(zip(class_name, results.tolist()))
    }


async def process_image(stub, image_file):

    img_data = preprocess(image_file)

    request = _make_request(img_data)
    response = await stub.ModelInfer(request)
    contents = _bytes_to_numpy(response.raw_output_contents)

    result = postprocess(contents)

    log.debug(result)

    return result



