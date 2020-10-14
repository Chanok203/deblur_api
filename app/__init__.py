from flask import Flask

def create_app(config_class):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    from app.gui import bp as gui_bp
    app.register_blueprint(gui_bp, url_prefix="/")

    from app.deblur import bp as deblur_api_bp
    app.register_blueprint(deblur_api_bp, url_prefix="/deblur_api/v1")

    return app