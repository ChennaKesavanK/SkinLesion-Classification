# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import csv

# Custom Callback to log metrics (including val_loss) to CSV
class MetricsLogger(Callback):
    def __init__(self, filepath):
        super(MetricsLogger, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            if epoch == 0:
                writer.writerow(['Epoch', 'Loss', 'Accuracy', 'Val_Loss', 'Val_Accuracy'])
            writer.writerow([epoch + 1, logs.get('loss'), logs.get('accuracy'), logs.get('val_loss'), logs.get('val_accuracy')])

# Paths to your dataset directories
train_dir = "C:/Users/chenna kesavan/OneDrive/Desktop/skin cancer detection/PRIE/train"
val_dir = "C:/Users/chenna kesavan/OneDrive/Desktop/skin cancer detection/PRIE/validation"
test_dir = "C:/Users/chenna kesavan/OneDrive/Desktop/skin cancer detection/PRIE/test"

# Image properties
img_size = (224, 224)
batch_size = 64  # Reduced batch size for potentially better performance

# Data generators (augmentation applied to training data)
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load data with data generators
train_generator = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=True)
val_generator = val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False)

# Load the ResNet50 model with pre-trained ImageNet weights
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

# Freeze the base model initially
for layer in base_model.layers:
    layer.trainable = False

# Add new layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)  # Dropout for regularization
predictions = Dense(train_generator.num_classes, activation='softmax', kernel_regularizer=l2(0.01))(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
metrics_logger = MetricsLogger(filepath='training_metrics.csv')

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# Calculate steps per epoch
steps_per_epoch = np.ceil(train_generator.samples / batch_size).astype(int)
validation_steps = np.ceil(val_generator.samples / batch_size).astype(int)

# Fine-tune the model
history_finetune = model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator,
    callbacks=[early_stopping, model_checkpoint, reduce_lr, metrics_logger],
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1
)

# Save the final model
model.save('skin_cancer_detection_model_finetuned.keras')

# Plot accuracy for visualization
plt.plot(history_finetune.history['accuracy'], label='Train Accuracy')
plt.plot(history_finetune.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predict classes on the test data
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# --- Multi-class confusion matrix ---
cm_multi = confusion_matrix(true_classes, predicted_classes)
disp_multi = ConfusionMatrixDisplay(confusion_matrix=cm_multi, display_labels=list(test_generator.class_indices.keys()))
disp_multi.plot(cmap=plt.cm.Blues)
plt.title("Multi-Class Confusion Matrix (8 Classes)")
plt.show()

# Define function to map multi-class predictions to binary (malignant/benign)
malignant_classes = ['Melanoma', 'Squamous cell carcinoma', 'Basal cell carcinoma']
benign_classes = ['Actinic Keratosis', 'Dermatofibroma', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis', 'Vascular lesion']

def map_to_binary_class(class_indices, class_indices_dict):
    mapped_classes = []
    for idx in class_indices:
        class_name = list(class_indices_dict.keys())[list(class_indices_dict.values()).index(idx)]
        mapped_classes.append(1 if class_name in malignant_classes else 0)
    return mapped_classes

# Map to binary classes
binary_true_classes = map_to_binary_class(true_classes, test_generator.class_indices)
binary_predicted_classes = map_to_binary_class(predicted_classes, test_generator.class_indices)

# Plot binary confusion matrix
cm = confusion_matrix(binary_true_classes, binary_predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Binary Classification Confusion Matrix")
plt.show()


