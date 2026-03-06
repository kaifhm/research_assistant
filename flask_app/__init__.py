from dotenv import load_dotenv
from flask_app.db import db, Base
from sqlalchemy import create_engine
from flask_app.server import app

load_dotenv()

engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'], echo=True)
Base.metadata.create_all(engine)

db.init_app(app)