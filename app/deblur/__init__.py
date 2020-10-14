from flask import Blueprint
from flask_cors import CORS

bp = Blueprint("deblur", __name__)
CORS(bp)

from . import routes # isort:skip
