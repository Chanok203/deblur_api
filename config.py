import os
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)

from app.deblur.models.deblur import Deblur

class Config(object):
    DEBLUR_MODEL = Deblur()
    API_URL = "http://0.0.0.0:5000/deblur_api/v1/predict"

