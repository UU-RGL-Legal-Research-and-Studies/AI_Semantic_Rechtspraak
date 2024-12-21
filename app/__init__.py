from flask import Flask
import os
import tempfile

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()  # Gebruik de tijdelijke directory

from app import routes  # Importeer routes na het aanmaken van de app
