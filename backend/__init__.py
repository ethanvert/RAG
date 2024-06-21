from flask import Flask
from backend.config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__, template_folder='/Users/ethanvertal/Documents/RAG Site/frontend/templates')
app.config.from_object(Config)
db = SQLAlchemy(app)
migrate = Migrate(app, db)

from backend import routes, dataclass