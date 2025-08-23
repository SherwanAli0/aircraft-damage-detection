
import warnings
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import tarfile
import urllib.request
import shutil

warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuration
BATCH_SIZE = 32
N_EPOCHS = 5
IMG_SIZE = (224, 224, 3)
SEED_VALUE = 42

# Set random seeds for reproducibility
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

class AircraftDamageClassifier:
    """Complete pipeline for aircraft damage classification and captioning."""
    
    def __init__(self):
        self.model = None
        self.train_generator = None
        self.valid_generator = None
        self.test_generator = None
        self.history = None
        self.blip_processor = None
        self.blip_model = None
        
    def download_dataset(self):
        """Download and extract aircraft damage dataset."""
        url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar"
        tar_filename = "aircraft_damage_dataset_v1.tar"
        extracted_folder = "aircraft_damage_dataset_v1"
        
        if os.path.exists(extracted_folder):
            shutil.rmtree(extracted_folder)
            
        urllib.request.urlretrieve(url, tar_filename)
        with tarfile.open(tar_filename, "r") as tar_ref:
            tar_ref.extractall()
        
        print("Dataset downloaded and extracted successfully")
        return extracted_folder
    
    def setup_data_generators(self, data_path):
        """Create data generators for training, validation, and testing."""
        train_dir = os.path.join(data_path, 'train')
        valid_dir = os.path.join(data_path, 'valid')
        test_dir = os.path.join(data_path, 'test')
        
        # Create ImageDataGenerators
        train_datagen = ImageDataGenerator(rescale=1./255)
        valid_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE[:2],
            batch_size=BATCH_SIZE,
            seed=SEED_VALUE,
            class_mode='binary',
            shuffle=True
        )
        
        self.valid_generator = valid_datagen.flow_from_directory(
            valid_dir,
            target_size=IMG_SIZE[:2],
            batch_size=BATCH_SIZE,
            seed=SEED_VALUE,
            class_mode='binary',
            shuffle=False
        )
        
        self.test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=IMG_SIZE[:2],
            batch_size=BATCH_SIZE,
            seed=SEED_VALUE,
            class_mode='binary',
            shuffle=False
        )
        
        print(f"Data generators created:")
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.valid_generator.samples}")
        print(f"Test samples: {self.test_generator.samples}")
    
    def build_model(self):
        """Build VGG16-based model for aircraft damage classification."""
        # Load pre-trained VGG16 model
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=IMG_SIZE)
        
        # Add custom classification layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        # Create complete model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base VGG16 layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Model built with {len(self.model.layers)} layers")
        return self.model
    
    def train_model(self):
        """Train the aircraft damage classification model."""
        if not self.model:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.history = self.model.fit(
            self.train_generator,
            epochs=N_EPOCHS,
            validation_data=self.valid_generator
        )
        
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        
        print(f"Training completed!")
        print(f"Final training accuracy: {final_train_acc:.4f}")
        print(f"Final validation accuracy: {final_val_acc:.4f}")
        
        return self.history
    
    def evaluate_model(self):
        """Evaluate model performance on test set."""
        if not self.model:
            raise ValueError("Model not trained. Call train_model() first.")
        
        test_loss, test_accuracy = self.model.evaluate(
            self.test_generator, 
            steps=self.test_generator.samples // self.test_generator.batch_size
        )
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return test_loss, test_accuracy
    
    def plot_training_curves(self):
        """Plot training and validation accuracy/loss curves."""
        if not self.history:
            raise ValueError("No training history found. Train model first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def setup_blip_model(self):
        """Initialize BLIP model for image captioning."""
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print("BLIP model loaded for captioning")
    
    def generate_caption(self, image_path, task_type="caption"):
        """Generate caption or summary for an image using BLIP."""
        if not self.blip_processor or not self.blip_model:
            self.setup_blip_model()
        
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            
            # Set prompt based on task
            prompt = "This is a picture of" if task_type == "caption" else "This is a detailed photo showing"
            
            # Generate text
            inputs = self.blip_processor(images=image, text=prompt, return_tensors="pt")
            output = self.blip_model.generate(**inputs)
            result = self.blip_processor.decode(output[0], skip_special_tokens=True)
            
            return result
            
        except Exception as e:
            print(f"Error generating {task_type}: {e}")
            return f"Error processing image"
    
    def predict_and_caption(self, image_path):
        """Complete pipeline: classify damage and generate caption."""
        if not self.model:
            raise ValueError("Classification model not trained.")
        
        # Load and preprocess image for classification
        img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE[:2])
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Classify damage
        prediction = self.model.predict(img_array)[0][0]
        damage_type = "crack" if prediction > 0.5 else "dent"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        # Generate caption
        caption = self.generate_caption(image_path, "caption")
        summary = self.generate_caption(image_path, "summary")
        
        return {
            "damage_type": damage_type,
            "confidence": confidence,
            "caption": caption,
            "summary": summary
        }


def main():
    """Main execution pipeline."""
    classifier = AircraftDamageClassifier()
    
    # Setup data
    print("Setting up dataset...")
    data_path = classifier.download_dataset()
    classifier.setup_data_generators(data_path)
    
    # Build and train model
    print("\nBuilding model...")
    classifier.build_model()
    
    print("\nTraining model...")
    classifier.train_model()
    
    # Evaluate model
    print("\nEvaluating model...")
    classifier.evaluate_model()
    
    # Plot results
    print("\nPlotting training curves...")
    classifier.plot_training_curves()
    
    # Setup captioning
    print("\nSetting up BLIP for captioning...")
    classifier.setup_blip_model()
    
    # Demo prediction and captioning
    sample_image = "aircraft_damage_dataset_v1/test/dent/149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg"
    if os.path.exists(sample_image):
        print(f"\nTesting complete pipeline on sample image...")
        result = classifier.predict_and_caption(sample_image)
        
        print(f"Damage Type: {result['damage_type']} (confidence: {result['confidence']:.3f})")
        print(f"Caption: {result['caption']}")
        print(f"Summary: {result['summary']}")


if __name__ == "__main__":
    main()
