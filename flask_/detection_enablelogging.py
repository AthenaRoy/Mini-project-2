import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import cv2
import numpy as np
import psycopg2
import logging

logging.basicConfig(filename='detection.log', level=logging.DEBUG)

#LOSS AND ACCURACY PLOT

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

#Data gfenerators


train_dir = 'C:/Users/athen/OneDrive/Documents/Mini project 2/archive (1)/images/images/train'
val_dir = 'C:/Users/athen/OneDrive/Documents/Mini project 2/archive (1)/images/images/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 2

datagen = ImageDataGenerator(rescale=1./255)
print(datagen)
train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')
validation_generator = datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

import firebase_admin
from firebase_admin import credentials, initialize_app, storage, get_app
from firebase_admin.exceptions import FirebaseError
import requests
# Initialize Firebase app if not already initialized
try:
    firebase_app = get_app()
except ValueError:

    try:  # Raised if get_app() cannot find the app
        cred = credentials.Certificate('C:/Users/athen/OneDrive/Documents/Mini project 2/mini-project-d9780-firebase-adminsdk-excc6-1f7073b6d8.json')
        initialize_app(cred, {
            'storageBucket': 'mini-project-d9780.appspot.com'
        })
        firebase_app = get_app()
        print("Firebase app initialized successfully.")
    except FileNotFoundError as e:
        print(f"Error: Firebase credentials file not found - {e}")
    except FirebaseError as e:
        print(f"Error initializing Firebase app: {e}")
    except Exception as e:
        print(f"Unexpected error initializing Firebase app: {e}")

def download_video(file_name):
    bucket = storage.bucket()
    blob = bucket.blob(f'path/to/your/videos/{file_name}')
    destination = f'./videos/{file_name}'
    blob.download_to_filename(destination)
    print(f'File {file_name} downloaded to {destination}.')

class EmotionDetector:
    def __init__(self, model):
        self.model = model

    
    def predict(self, frame):
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        frame = cv2.resize(frame, (48, 48))   
        frame = frame / 255.0 
        frame = frame.reshape(1, 48, 48, 1) 

        # Make predictions
        predictions = self.model.predict(frame)
        return predictions

    def detect_emotions(self,frame):
        predictions = self.predict(frame)
        
        # Convert predictions to emotion labels
        # This is a dummy implementation, replace with your actual logic
        emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        detected_emotions = [emotions[pred.argmax()] for pred in predictions]

        return detected_emotions 

class Video:
    def __init__(self, download_path):
        self.path = download_path
        print(f"Attempting to open video at: {download_path}")
        if not os.path.isfile(download_path):
            print(f"Error: Video file does not exist at {download_path}")
            return
        self.capture = cv2.VideoCapture(download_path)
        if not self.capture.isOpened():
            print("Error: Could not open video.")

    
    def analyse(self, detector, display=False, save_path="frames"):
        os.makedirs(save_path, exist_ok=True)
        frame_count = 0
        results1 = []
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break
    
            # Perform emotion detection on the frame
            result = detector.detect_emotions(frame) 
            if result is not None:  # Ensure result is not None before appending
                results1.append((frame_count, result))
            
            print(f"Frame {frame_count}: {result}")# Fixing the method call
    
            if display:
                try:
                    for emotion in result:
                        cv2.putText(frame, str(emotion), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    frame_filename = os.path.join(save_path, f"frame_{frame_count:04d}.jpg")
                    cv2.imwrite(frame_filename, frame)
                except Exception as e:
                    print(f"Error saving/displaying frame {frame_count}: {e}")
        
            frame_count += 1

        self.capture.release()
        print("Analysis complete.") 
        return results1
        
def ensure_directory_exists(video_directory):
    if not os.path.exists(video_directory):
        os.makedirs(video_directory)
        print(f"Created directory: {video_directory}")

def download_video_from_firebase(file_name):
    bucket = storage.bucket()
    blob = bucket.blob(file_name)
    
    # Ensure the videos directory exists
    video_directory = "./videos"
    ensure_directory_exists(video_directory)
    
    # Construct the download path
    download_path = os.path.join(video_directory, os.path.basename(file_name))
    try:
        blob.download_to_filename(download_path)
        print(f"Downloaded {file_name} to {download_path}")
        return download_path

    except FirebaseError as e:
        print(f"Error downloading {file_name}: {e}")
        return None
    except FileNotFoundError as e:
        print(f"Error creating or writing to {download_path}: {e}")
        return None

def process_video(download_path):
    if download_path is None:
        print("Error: download_path is None.")
        return None
    video_directory = "./videos"
    video_files = os.listdir(video_directory)
    if not video_files:
        print("Error: No video files found in directory.")
        return None
    video_file = video_files[0]
    download_path = f"{video_directory}/{video_file}"
    
    results2 = []
    try:

        # Process the video with your model
        print(f"Processing video: {download_path}")
        
        video = Video(download_path)
        model = Sequential()
        
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

        model.save('flask_/emotion_detection_model.h5')
        model = keras.models.load_model("flask_/emotion_detection_model.h5")
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        
        history = model.fit(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size)
        
        
        
        video = Video(download_path)
        emotion_detector = EmotionDetector(model)
        results2 = video.analyse(emotion_detector, display=True)
# (Your model processing code goes here)
        
        # Delete the video file after processing
        os.remove(download_path)
        print(f"Deleted video: {download_path}")
        return results2, download_path
    except Exception as e:
        print(f"Error processing video: {e}")
        return None, None
    
def fetch_video_filenames():
    # Example: Fetch filenames from Firebase Storage
    bucket = storage.bucket()
    blobs = bucket.list_blobs()
    video_filenames = [blob.name for blob in blobs if blob.name.endswith('.mp4')]
    
    return video_filenames
    return ["video1.mp4", "video2.mp4"]

try:
    logging.info("Detection script started.")
    # Fetch video filenames dynamically
    video_filenames = fetch_video_filenames()
    print(f"Fetched video filenames: {video_filenames}")
    current_results = []

    for filename in video_filenames:
        download_path = download_video_from_firebase(filename)
        current_results, download_path = process_video(download_path)
        print(f"Analyzing {filename}, got results: {current_results}")

    if current_results is None:
        print("Error: analyze method returned None")
    else:
        # Convert results to pandas DataFrame
        data = {"frame_count": [], "happy": [], "surprise": [], 
            "angry": [], "disgust": [], "fear": [], "sad": []}

    # Populate the DataFrame with results
        for frame_count, emotions in current_results:
            data["frame_count"].append(frame_count)
            data["happy"].append(emotions.count("happy"))
            data["surprise"].append(emotions.count("surprise"))
            data["angry"].append(emotions.count("angry"))
            data["disgust"].append(emotions.count("disgust"))
            data["fear"].append(emotions.count("fear"))
            data["sad"].append(emotions.count("sad"))
        emotions_df = pd.DataFrame(data)

            # Display the first few rows of the DataFrame
        print(emotions_df.head())
    #raise ValueError("Simulated error for testing.")  # Remove this line after testing
    logging.info("Detection script finished successfully.")
except Exception as e:
    logging.error(f"Error in detection script: {e}")
    raise

db_params = {
    'host' : 'ep-wild-scene-a1bnv5tq.ap-southeast-1.aws.neon.tech',
    'database' : 'miniproject',
    'user' : 'athena',
    'password' : '0hStGLnb5fDa',
    'port': '5432'
}

try:
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    print("Database connection established.")
except Exception as e:
    print(f"Error connecting to database: {e}")
    conn, cursor = None, None

def standardize_path(filename):
    return filename.replace('videos/','static\\files\\')

path_to_video = standardize_path(filename)

def update_results(cursor, conn, current_results, path_to_video):
    try:
        # Debug: Check if the video path exists in the database
        cursor.execute('SELECT * FROM uploaded_videos WHERE video = %s', (path_to_video,))
        video_exists = cursor.fetchone()
        print(f"Video exists: {video_exists}")

        if video_exists:
            cursor.execute('UPDATE uploaded_videos SET review = %s WHERE video = %s', (current_results, path_to_video))
            conn.commit()
            print(f"Update successful for video {path_to_video} with results {current_results}")
        else:
            print(f"No entry found for video {path_to_video}")
    except Exception as e:
        print(f"Error updating the database: {e}")
    finally:
        # Check if the transaction is in an error state
        if conn:
            print(f"Connection status: {conn.status}")

# Predict whether a person show interest in a topic or not

positive_emotions = sum(emotions_df.happy) + sum(emotions_df.surprise)
negative_emotions = sum(emotions_df.angry) + sum(emotions_df.disgust) + sum(emotions_df.fear) + sum(emotions_df.sad)

if positive_emotions > negative_emotions:
    print("Person is interested")
    update_results(cursor,conn,"Person is satisfied")
elif positive_emotions < negative_emotions:
    print("Person is not interested")
    update_results(cursor,conn,"Person is not satisfied",path_to_video)
else:
    print("Person is neutral")

if cursor:
    cursor.close()
if conn:
    conn.close()
