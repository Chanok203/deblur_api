from flask import request, jsonify
from flask import current_app
from .utils import converters

from . import bp


@bp.route("/")
def test():
    print(current_app.config["DEBLUR_MODEL"])
    return "Deblur API is running normally."


@bp.route("/predict", methods=["POST"])
def predict():
    base64_string = request.json["base64_string"]
    image = converters.base64_to_numpy(base64_string)
    model = current_app.config["DEBLUR_MODEL"]
    result, time_process = model.predict(image)
    return jsonify({
        "result": converters.numpy_to_base64(result),
        "time_process": f"{time_process:.6f}",
    })