import cv2
import numpy as np
class Video:
    def __init__(self, path_to_video):
        self.path = path_to_video
        self.cap = cv2.VideoCapture(path_to_video)
    
    def analyze(self, detector, display=False):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Perform emotion detection on the frame
            result = detector.detect_emotions(frame)
            if display:
                # Display the frame with the detection results
                for emotion in result:
                    cv2.putText(frame, emotion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('Video', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        self.cap.release()
        cv2.destroyAllWindows()


class EmotionDetector:
    def __init__(self, model):
        self.model = model
    
    def detect_emotions(self, frame):
        # Preprocess the frame as needed for your model
        # Here is a dummy implementation
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        frame = cv2.resize(frame, (48, 48))   
        frame = frame / 255.0 
        frame = frame.reshape(1, 48, 48, 1) 

        # Make predictions
        predictions = self.model.predict(frame)
        
        # Convert predictions to emotion labels
        # This is a dummy implementation, replace with your actual logic
        emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        detected_emotions = [emotions[pred.argmax()] for pred in predictions]

        return detected_emotions 