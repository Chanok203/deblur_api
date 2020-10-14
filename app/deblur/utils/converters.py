import base64
from PIL import Image
import io
import numpy as np


def numpy_to_base64(arr):
    image = Image.fromarray(arr)
    buffered = io.BytesIO()
    image.save(buffered, format="png")
    base64_string = base64.b64encode(buffered.getvalue())
    return base64_string.decode("ascii")


def base64_to_numpy(base64_string):
    img_bin = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(img_bin))
    return np.array(image)
