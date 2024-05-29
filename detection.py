
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
import os



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




import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, storage
import requests


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
    def __init__(self, path_to_video):
        self.path = path_to_video
        self.capture = cv2.VideoCapture(path_to_video)
        if not self.capture.isOpened():
            print("Error: Could not open video.")

    
    def analyze(self, detector, display=False, save_path="frames"):
        os.makedirs(save_path, exist_ok=True)
        frame_count = 0
        results = []
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break
    
            # Perform emotion detection on the frame
            result = detector.detect_emotions(frame) 
            results.append((frame_count, result))
            print(f"Frame {frame_count}: {result}")# Fixing the method call
    
            if display:
                for emotion in result:
                    cv2.putText(frame, str(emotion), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                frame_filename = os.path.join(save_path, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                frame_count += 1

        self.capture.release()
        print("Analysis complete.") 
        return results


path_to_video = 'C:/Users/athen/OneDrive/Documents/Mini project 2/static/files/Flexpressions.mp4'

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



model.save('emotion_detection_model.h5')
model = keras.models.load_model("emotion_detection_model.h5")
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=num_train // batch_size,
    epochs=num_epoch,
    validation_data=validation_generator,
    validation_steps=num_val // batch_size)





video = Video(path_to_video)
emotion_detector = EmotionDetector(model)
results = video.analyze(emotion_detector, display=True)



if results is None:
    print("Error: analyze method returned None")
else:
    # Convert results to pandas DataFrame
   data = {"frame_count": [], "happy": [], "surprise": [], 
        "angry": [], "disgust": [], "fear": [], "sad": []}

# Populate the DataFrame with results
for frame_count, emotions in results:
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



#connecting to neon
from sqlalchemy import create_engine

db_params = {
    'host' : 'ep-wild-scene-a1bnv5tq.ap-southeast-1.aws.neon.tech',
    'database' : 'miniproject',
    'user' : 'athena',
    'password' : '0hStGLnb5fDa',
    'port': '5432'
}
engine = create_engine(f'postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}')


import psycopg2
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

def standardize_path(path_to_video):
    return path_to_video.replace('files/','files\\')

path_to_video = standardize_path('static/files/Flexpressions.mp4')


def update_results(cursor, conn, results, path_to_video):
    try:
        # Debug: Check if the video path exists in the database
        cursor.execute('SELECT * FROM uploaded_videos WHERE video = %s', (path_to_video,))
        video_exists = cursor.fetchone()
        print(f"Video exists: {video_exists}")

        if video_exists:
            cursor.execute('UPDATE uploaded_videos SET review = %s WHERE video = %s', (results, path_to_video))
            conn.commit()
            print(f"Update successful for video {path_to_video} with results {results}")
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



