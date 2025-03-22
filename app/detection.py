import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
import numpy as np
import os

class CropClassifier:
    def __init__(self, dataset_path="dataset/Crop_detection", model_path="model/crop_classification_model.keras"):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.img_size = (224, 224)
        self.batch_size = 32
        self.crop_classes = []  # Initialize empty list

        # Check if model exists
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print(f"Loaded trained model from {self.model_path}")
            
            # Initialize crop classes from dataset directory
            self._initialize_crop_classes()
        else:
            self.model = None
            print("No trained model found. Please train the model first.")

    def _initialize_crop_classes(self):
        """Initialize crop classes from dataset directory"""
        if os.path.exists(self.dataset_path):
            # Get class names from directory structure
            self.crop_classes = sorted([d for d in os.listdir(self.dataset_path) 
                                      if os.path.isdir(os.path.join(self.dataset_path, d))])
            if self.crop_classes:
                print(f"Initialized crop classes: {self.crop_classes}")
            else:
                print("Warning: No crop classes found in dataset directory.")
        else:
            print(f"Warning: Dataset path {self.dataset_path} not found.")

    def train_model(self, epochs=10):
        """Train the model and save it"""
        # Check for GPU availability
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print(f"Found {len(physical_devices)} GPU(s). Training on GPU.")
            # Configure memory growth to avoid memory allocation errors
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        else:
            print("No GPU found. Training on CPU.")
            
        # Data augmentation for training set
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=0.2
        )

        # Only rescale validation data
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )

        train_generator = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="training"
        )

        val_generator = val_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="validation"
        )

        self.class_indices = train_generator.class_indices
        self.crop_classes = list(self.class_indices.keys())
        print("Detected Crop Classes:", self.crop_classes)

        # Load Pretrained Model (MobileNetV2)
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        
        # Fine-tuning: Unfreeze some top layers for better performance
        # First freeze all layers
        base_model.trainable = False
        
        # Then unfreeze some top layers for fine-tuning
        for layer in base_model.layers[-20:]:
            layer.trainable = True

        # Build Model
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)  # Increased dropout rate
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.3)(x)
        predictions = Dense(len(self.crop_classes), activation="softmax")(x)

        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Use learning rate scheduler to reduce LR on plateau
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
        
        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        # Model checkpoint to save best model
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        # Train Model
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=[lr_scheduler, early_stopping, checkpoint]
        )

        # Save Model (if not already saved by checkpoint)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        if not os.path.exists(self.model_path):
            self.model.save(self.model_path)
        print(f"Model saved at {self.model_path}")

        return history

    def predict_crop(self, image_path):
        """Predict the crop type for a given image"""
        if self.model is None:
            # Try to train the model if dataset exists
            if os.path.exists(self.dataset_path) and os.listdir(self.dataset_path):
                print("Training model for first-time use...")
                self.train_model(epochs=10)
            else:
                print("No trained model found and no dataset available for training.")
                return None

        # Load and preprocess image
        try:
            img = load_img(image_path, target_size=self.img_size)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Expand dims for batch shape
            img_array = preprocess_input(img_array)

            # Prediction
            predictions = self.model.predict(img_array)
            confidence = np.max(predictions) * 100  # Convert to percentage
            predicted_class = self.crop_classes[np.argmax(predictions)]

            return {"crop": predicted_class, "confidence": confidence}
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return {"crop": "unknown", "confidence": 0.0}

def main():
    classifier = CropClassifier()

    # Train if model doesn't exist
    if not os.path.exists(classifier.model_path):
        classifier.train_model(epochs=50)

    # Example Prediction
    image_path = "/kaggle/input/crop-detection/Crop_detection/banana/sample.jpg"  # Change this to a real image path
    result = classifier.predict_crop(image_path)
    print(result)
# Example Usage
if __name__ == "__main__":
    classifier = CropClassifier()

    # Train if model doesn't exist
    if not os.path.exists(classifier.model_path):
        classifier.train_model(epochs=50)

    # Example Prediction
    image_path = "/kaggle/input/crop-detection/Crop_detection/banana/sample.jpg"  # Change this to a real image path
    result = classifier.predict_crop(image_path)
    print(result)