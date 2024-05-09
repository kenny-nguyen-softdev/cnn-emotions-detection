import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import os
import logging
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Set logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/test'
IMAGE_SIZE = (48, 48)
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001

# Command line argument parser
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="train/display")
    return parser.parse_args()

# Plot model accuracy and loss curves
def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for metric, ax in zip(['accuracy', 'loss'], axs):
        ax.plot(model_history.history[metric], label='train')
        ax.plot(model_history.history['val_' + metric], label='val')
        ax.set_title('Model ' + metric.capitalize())
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
    plt.show()

# Load and compile the model
def load_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 1)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])
    return model

# Train the model
def train_model(model, train_generator, validation_generator):
    logger.info("Training the model...")
    model_info = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=len(validation_generator)
    )
    plot_model_history(model_info)
    model.save_weights('model.h5')

# Display emotions from webcam feed
def display_emotions(model):
    logger.info("Displaying emotions from webcam feed...")
    # Load the pre-trained model weights
    model.load_weights('model.h5')

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Detect faces in the frame
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        # Process each detected face
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, IMAGE_SIZE), -1), 0)
            prediction = model.predict(cropped_img)
            max_index = np.argmax(prediction)
            emotion = emotion_dict[max_index]
            cv2.putText(frame, emotion, (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_arguments()
    if args.mode == "train":
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            color_mode="grayscale",
            class_mode='categorical'
        )
        val_datagen = ImageDataGenerator(rescale=1./255)
        validation_generator = val_datagen.flow_from_directory(
            VAL_DIR,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            color_mode="grayscale",
            class_mode='categorical'
        )
        model = load_model()
        train_model(model, train_generator, validation_generator)
    elif args.mode == "display":
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        model = load_model()
        display_emotions(model)
    else:
        logger.error("Invalid mode. Please specify 'train' or 'display'.")
