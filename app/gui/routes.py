from flask import request, current_app
from flask.templating import render_template
from . import bp
from app.deblur.utils import converters
from PIL import Image
import io
import numpy as np
import requests


@bp.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("home.html")
    elif request.method == "POST":
        buffered = io.BytesIO()
        image_file = request.files["image"]
        image_file.save(buffered)
        image_arr = np.array(Image.open(buffered))
        base64_string = converters.numpy_to_base64(image_arr)

        url = current_app.config["API_URL"]
        response = requests.post(
            url,
            json={
                "base64_string": base64_string,
            },
        )

        res_json = response.json()
        original = "data:image/png;base64, " + base64_string
        result = "data:image/png;base64, " + res_json["result"]
        time_process = res_json["time_process"]

        return render_template(
            "home.html",
            original=original,
            result=result,
            time_process=time_process,
        )
