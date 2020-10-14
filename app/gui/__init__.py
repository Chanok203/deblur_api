from flask import Blueprint

bp = Blueprint("gui", __name__)

from . import routes