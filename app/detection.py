import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import json

class CropClassifier:
    def __init__(self, img_width=224, img_height=224, batch_size=32, epochs=50):
        """Initialize the CropClassifier with configurable parameters."""
        # Check for GPU availability and configure
        self.configure_gpu()
        
        # Model parameters
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.class_indices = None
    
    def configure_gpu(self):
        """Configure GPU if available."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Use the first GPU
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                # Allow memory growth to avoid taking all GPU memory
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print(f"Using GPU: {gpus[0]}")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("No GPU found. Using CPU.")
    
    def build_model(self, num_classes):
        """Build the CNN model architecture."""
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', padding='same', 
                   input_shape=(self.img_width, self.img_height, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            
            # Fourth convolutional block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            
            # Flatten and Dense layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),  # Dropout to prevent overfitting
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, train_dir, validation_dir, test_dir=None, model_save_path='crop_classifier.keras'):
        """Train the model on the provided dataset directories."""
        # Data augmentation for training set
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Only rescaling for validation and test sets
        valid_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load and prepare the datasets
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical'
        )

        validation_generator = valid_datagen.flow_from_directory(
            validation_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        # Store class indices for prediction
        self.class_indices = train_generator.class_indices
        self.inverted_class_indices = {v: k for k, v in self.class_indices.items()}
        
        # Save class indices to file for later loading
        with open('class_indices.json', 'w') as f:
            json.dump(self.class_indices, f)
        
        # Get number of classes
        num_classes = len(self.class_indices)
        print(f"Number of classes: {num_classes}")
        print(f"Class indices: {self.class_indices}")
        
        # Build the model
        self.model = self.build_model(num_classes)
        self.model.summary()
        
        # Set up callbacks to prevent overfitting and improve learning
        callbacks = [
            # Stop training when validation loss doesn't improve
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when validation accuracy plateaus
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            # Save the best model
            ModelCheckpoint(
                filepath='best_' + model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // self.batch_size,
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set if provided
        if test_dir:
            test_datagen = ImageDataGenerator(rescale=1./255)
            test_generator = test_datagen.flow_from_directory(
                test_dir,
                target_size=(self.img_width, self.img_height),
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False
            )
            
            # Evaluate the model on test data
            test_loss, test_acc = self.model.evaluate(test_generator)
            print(f"Test accuracy: {test_acc:.4f}")
            
            # Generate and plot metrics
            self.generate_metrics(test_generator, history)
        
        # Save the final model
        self.model.save(model_save_path)
        print(f"Model saved as '{model_save_path}'")
        
        return history
    
    def generate_metrics(self, test_generator, history):
        """Generate and plot performance metrics."""
        # Plot training history
        self.plot_training_history(history)
        
        # Make predictions on test data
        predictions = self.model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true classes
        true_classes = test_generator.classes
        
        # Print classification report
        from sklearn.metrics import classification_report, confusion_matrix
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, 
                                    target_names=list(self.inverted_class_indices.values())))
        
        # Create and plot confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(self.inverted_class_indices))
        plt.xticks(tick_marks, self.inverted_class_indices.values(), rotation=90)
        plt.yticks(tick_marks, self.inverted_class_indices.values())
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.show()
    
    def plot_training_history(self, history):
        """Plot training and validation accuracy/loss."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def load_model(self, model_path, class_indices_path=None):
        """Load a pre-trained model and class indices."""
        self.model = load_model(model_path)
        
        # Load class indices
        if class_indices_path:
            with open(class_indices_path, 'r') as f:
                self.class_indices = json.load(f)
        elif os.path.exists('class_indices.json'):
            with open('class_indices.json', 'r') as f:
                self.class_indices = json.load(f)
                
        if self.class_indices:
            self.inverted_class_indices = {v: k for k, v in self.class_indices.items()}
        else:
            raise ValueError("Class indices not found. Please provide path to class_indices.json")
        
        return self
    
    def preprocess_image(self, image):
        """
        Preprocess a single image for prediction.
        
        Args:
            image: Can be a PIL Image, numpy array, or file path
        """
        # Handle different input types
        if isinstance(image, str):  # It's a file path
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):  # It's a numpy array
            if image.ndim == 2:  # Grayscale
                img = Image.fromarray(image).convert('RGB')
            else:
                img = Image.fromarray(image)
        elif hasattr(image, 'convert'):  # It's already a PIL Image
            img = image.convert('RGB')
        else:
            raise ValueError("Unsupported image type. Please provide a PIL Image, numpy array, or file path.")
        
        # Resize and normalize
        img = img.resize((self.img_width, self.img_height))
        img_array = np.array(img) / 255.0  # Normalize
        return np.expand_dims(img_array, axis=0)  # Add batch dimension

    def predict(self, image):
        """
        Predict crop class from an image.
        
        Args:
            image: Can be a PIL Image, numpy array, or file path
                
        Returns:
            dict: JSON-compatible dictionary with crop name and confidence
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess the image
        processed_image = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(processed_image)
        
        # Get the predicted class index and confidence
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        
        # Get the class name
        if self.inverted_class_indices:
            crop_name = self.inverted_class_indices[predicted_class_index]
        else:
            crop_name = f"Class_{predicted_class_index}"
        
        # Return as JSON compatible dictionary
        return {
            "crop_name": crop_name,
            "confidence": confidence
        }
    
    def get_crops_names(self):
        return ["apple", "banana", "beetroot", "bell pepper", "cabbage", "capsicum", "carrot", "cauliflower", "chilli pepper", "corn", "cucumber", "eggplant", "garlic", "ginger", "grapes", "jalepeno", "kiwi", "lemon", "lettuce", "mango", "onion", "orange", "paprika", "pear", "peas", "pineapple", "pomegranate", "potato", "raddish", "soy beans", "spinach", "sweetcorn", "sweetpotato", "tomato", "turnip", "watermelon"]


# Example usage (can be imported elsewhere)
if __name__ == "__main__":
    # Example training
    classifier = CropClassifier(epochs=20)
    classifier.train(
        train_dir='train',
        validation_dir='validation',
        test_dir='test',
        model_save_path='crop_classifier.keras'
    )
    
    # Example prediction
    # result = classifier.predict("path/to/test_image.jpg")
    # print(json.dumps(result, indent=4))
    
    # Or to load a pre-trained model
    # classifier = CropClassifier().load_model('crop_classifier.keras')
    # result = classifier.predict("path/to/test_image.jpg")
    # print(json.dumps(result, indent=4))