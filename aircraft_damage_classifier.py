
import warnings
import os
import math
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
import tempfile
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

DATASET_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar"
DATASET_DIRNAME = "aircraft_damage_dataset_v1"
DATASET_SPLITS = ("train", "valid", "test")

# Set random seeds for reproducibility
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)


def _is_dataset_dir(path):
    """True if path looks like the extracted dataset (train/valid/test folders)."""
    return all(os.path.isdir(os.path.join(path, split)) for split in DATASET_SPLITS)


def _safe_extract(tar, dest):
    """Extract a tar archive into dest, refusing members that would land outside it."""
    dest_root = os.path.realpath(dest)
    for member in tar.getmembers():
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"Refusing to extract non-regular archive member: {member.name}")
        target = os.path.realpath(os.path.join(dest_root, member.name))
        if os.path.commonpath([dest_root, target]) != dest_root:
            raise ValueError(f"Archive member would be extracted outside the target folder: {member.name}")
    tar.extractall(dest_root)


def decode_prediction(probability, class_indices):
    """Map a sigmoid output to (class name, confidence).

    class_indices is the Keras generator mapping {class name: index}. With
    class_mode='binary' the sigmoid output is the probability of the class whose
    index is 1; Keras scores a prediction as class 1 only when it is > 0.5, so
    an exact 0.5 is assigned to class 0 here too.
    """
    if not class_indices or sorted(class_indices.values()) != [0, 1]:
        raise ValueError(
            "Class mapping unavailable: call setup_data_generators() first "
            f"(got class_indices={class_indices!r})."
        )
    index_to_class = {index: name for name, index in class_indices.items()}
    index = int(probability > 0.5)
    confidence = probability if index == 1 else 1 - probability
    return index_to_class[index], float(confidence)


class AircraftDamageClassifier:
    """Complete pipeline for aircraft damage classification and captioning."""
    
    def __init__(self):
        self.model = None
        self.train_generator = None
        self.valid_generator = None
        self.test_generator = None
        self.class_indices = None
        self.history = None
        self.blip_processor = None
        self.blip_model = None
        
    def download_dataset(self, data_dir="."):
        """Download and extract the aircraft damage dataset into data_dir.
        
        Returns the path of the extracted dataset folder. A valid existing copy
        is reused. Anything else already at that path is left untouched and an
        error is raised. The archive is unpacked in a temporary folder inside
        data_dir and only moved into place once it looks like the dataset.
        """
        data_dir = os.path.abspath(data_dir)
        os.makedirs(data_dir, exist_ok=True)
        extracted_folder = os.path.join(data_dir, DATASET_DIRNAME)
            
        if os.path.isdir(extracted_folder) and _is_dataset_dir(extracted_folder):
            print(f"Dataset already present at {extracted_folder}; reusing it")
            return extracted_folder
        if os.path.lexists(extracted_folder):
            raise FileExistsError(
                f"{extracted_folder} exists but is not the dataset (expected train/valid/test "
                "folders). It was not modified; move it or pass a different data_dir."
            )

        staging = tempfile.mkdtemp(prefix=".dataset_download_", dir=data_dir)
        try:
            tar_path = os.path.join(staging, "aircraft_damage_dataset_v1.tar")
            urllib.request.urlretrieve(DATASET_URL, tar_path)
            unpacked = os.path.join(staging, "unpacked")
            with tarfile.open(tar_path, "r") as tar_ref:
                _safe_extract(tar_ref, unpacked)
            candidate = os.path.join(unpacked, DATASET_DIRNAME)
            if not _is_dataset_dir(candidate):
                raise RuntimeError(
                    f"Downloaded archive does not contain {DATASET_DIRNAME}/train, valid and test folders."
                )
            os.replace(candidate, extracted_folder)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        
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
        
        # Keras assigns class indices from the sorted folder names; keep the
        # training mapping so predictions are decoded the way the model was trained.
        self.class_indices = dict(self.train_generator.class_indices)
        for name, generator in (("validation", self.valid_generator), ("test", self.test_generator)):
            if dict(generator.class_indices) != self.class_indices:
                raise ValueError(
                    f"{name} class mapping {dict(generator.class_indices)} differs from "
                    f"training mapping {self.class_indices}"
                )

        print(f"Data generators created:")
        print(f"Class mapping: {self.class_indices}")
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
        """Evaluate model performance on the whole test set (every image once)."""
        if not self.model:
            raise ValueError("Model not trained. Call train_model() first.")
        
        generator = self.test_generator
        if generator.samples == 0:
            raise ValueError("Test generator contains no samples.")
        # Round up so the final partial batch is evaluated as well.
        steps = math.ceil(generator.samples / generator.batch_size)

        test_loss, test_accuracy = self.model.evaluate(generator, steps=steps)
        
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
        
        # Classify damage; the label comes from the mapping used in training
        prediction = self.model.predict(img_array)[0][0]
        damage_type, confidence = decode_prediction(prediction, self.class_indices)
        
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
    sample_image = os.path.join(
        data_path, "test", "dent",
        "149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg",
    )
    if os.path.exists(sample_image):
        print(f"\nTesting complete pipeline on sample image...")
        result = classifier.predict_and_caption(sample_image)
        
        print(f"Damage Type: {result['damage_type']} (confidence: {result['confidence']:.3f})")
        print(f"Caption: {result['caption']}")
        print(f"Summary: {result['summary']}")


if __name__ == "__main__":
    main()
