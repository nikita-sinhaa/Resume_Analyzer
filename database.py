from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os

# Initialize Flask app and database
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///resume_data.db'
db = SQLAlchemy(app)

# Ensure the uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
