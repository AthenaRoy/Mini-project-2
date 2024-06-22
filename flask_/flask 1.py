import logging
import subprocess
from flask import Flask, redirect, url_for, render_template, request, session, flash, send_from_directory
from datetime import datetime, timedelta
from flask import jsonify
from flask_sqlalchemy import SQLAlchemy

import sys
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField, PasswordField
from wtforms import FileField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
from dotenv import load_dotenv
from sqlalchemy.exc import IntegrityError


import firebase_admin
from firebase_admin import credentials, storage


# Initialize the app with a service account, granting admin privileges
cred = credentials.Certificate('C:/Users/athen/OneDrive/Documents/Mini project 2/mini-project-d9780-firebase-adminsdk-excc6-1f7073b6d8.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'mini-project-d9780.appspot.com'
})



load_dotenv()

app = Flask(__name__)
app.secret_key = "hello"
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['UPLOAD_FOLDER'] = 'static/files'

db = SQLAlchemy(app)

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired()])
    passcode = PasswordField('Password', validators=[InputRequired()])
    submit = SubmitField('Register')

class Users(db.Model):
    __tablename__='login_info'
    username = db.Column("username", db.String(100), primary_key=True)
    passcode = db.Column("passcode", db.String(100))

    def __init__(self, username, passcode):
        self.username = username
        self.passcode = passcode

class Videos(db.Model):
    __tablename__='uploaded_videos'
    username = db.Column("username", db.String(100), primary_key=True)
    product_name = db.Column("product_name", db.String(100))
    video = db.Column("video", db.String(200))
    review = db.Column(db.String(255))


    def __init__(self, username, product_name, video, review):
        self.username = username
        self.product_name = product_name
        self.video = video
        self.review = review

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")


@app.route('/static/<path:filename>')
def custom_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/reviews', methods=['GET', 'POST'])
def video_review():
    if request.method == 'POST':
        product_name = request.form['product_name']
    
        video = Videos.query.filter_by(product_name=product_name).all()
    
        if video:
            review = [video.review for video in video]
            return render_template('review.html', product_name=product_name, review=review)
        else:
            # Handle case where no reviews are found for the product_name
            return render_template('review.html', message=f'No reviews found for product "{product_name}".')

    return render_template('review.html')


@app.route("/upload", methods=["GET", "POST"])
def upload():
    username = session.get('username')
    if not username:
        return "Username not found in session."
    
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        upload_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'])
        os.makedirs(upload_folder, exist_ok=True)

        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        
        bucket = storage.bucket()
        blob = bucket.blob(f'videos/{filename}')
        blob.upload_from_filename(file_path)

    
        product_name = request.form.get("product_name")
        upload_video = Videos(username=username, product_name=product_name, video=file_path,review=None)
        db.session.add(upload_video)
        db.session.commit()
        
        return redirect(url_for('upload_success'))
    return render_template("upload_16june2024.html", form=form)
    
@app.route("/upload_success")
def upload_success():
    return render_template("upload_success.html")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/view")
def view():
    return render_template("view.html", values=Users.query.all())


@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        user = request.form["username"]
        passcode = request.form["passcode"]
        session['username'] = user

        found_user = Users.query.filter_by(username=user, passcode=passcode).first()
        print(user)
        

        if found_user:
            flash("Login successful!")
            return redirect(url_for("home"))

        else:
            flash("Invalid username or password")
            return redirect(url_for("register"))
        
         
    return render_template("login.html")


@app.route("/user", methods=["POST" , "GET"])
def user():
    if request.method == "POST":
        username = request.form["username"]
        passcode = request.form["passcode"]

        if request.method == "POST":
            passcode = request.form["passcode"]
            session["passcode"] = passcode
            found_user = Users.query.filter_by(name=user).first()
            found_user.passcode = passcode
            db.session.commit()
            flash("Email was saved!")
        else:
            if "passcode" in session:
                email = session["passcode"]
        return render_template("user.html", user=user)
    else:
        flash("You are not logged in!")
        return redirect(url_for("login"))


@app.route("/logout")
def logout():
    flash("You have been logged out!", "info")
    session.pop("user", None)
    session.pop("passcode", None)
    return redirect(url_for("login"))



@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        username = form.username.data
        passcode = form.passcode.data
        session['username'] = username

            # Create a new user object and add it to the database
        new_user = Users(username=username, passcode=passcode)
        db.session.add(new_user)
        db.session.commit()
        
        flash("Registration successful!")
        return redirect(url_for("upload"))
    
    return render_template('register.html', form=form)

# Route to execute the Python script
@app.route('/execute_script', methods=['POST'])
def execute_script():
    try:
        # Run the Python script using subprocess
        result = subprocess.run([sys.executable, 'detection.py'], capture_output=True, text=True, check=True)
        # Log the output
        logging.info(f"Execution Time: {datetime.now()}")
        logging.info(result.stdout)
        return jsonify({'success': True, 'output': result.stdout})
    except subprocess.CalledProcessError as e:
        logging.error(f"Execution Time: {datetime.now()}")
        logging.error(e)
        return jsonify({'success': False, 'error': str(e)}), 500

# Route to fetch logs
@app.route('/get_logs', methods=['GET'])
def get_logs():
    log_file_path = os.path.join(os.getcwd(), 'script.log')

# Check if the file exists
    if not os.path.exists(log_file_path):
    # Create an empty file
        with open(log_file_path, 'w') as log_file:
            log_file.write('')
    try:
        with open('script.log', 'r') as log_file:
            logs = log_file.read()
        return jsonify({'success': True, 'logs': logs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)

