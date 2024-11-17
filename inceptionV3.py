# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
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

# Check GPU availability
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and memory growth is set.")
    except RuntimeError as e:
        print(e)

# Paths to your dataset directories
train_dir = "C:/Users/chenna kesavan/OneDrive/Desktop/skin cancer detection/PRIE/train"
val_dir = "C:/Users/chenna kesavan/OneDrive/Desktop/skin cancer detection/PRIE/validation"
test_dir = "C:/Users/chenna kesavan/OneDrive/Desktop/skin cancer detection/PRIE/test"

# Image properties
img_size = (224, 224)
batch_size = 8

# Data generators (augmentation already applied to your dataset)
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Load the validation data
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load the InceptionV3 model with pre-trained ImageNet weights
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add new layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Add dropout for regularization
predictions = Dense(train_generator.num_classes, activation='softmax', kernel_regularizer=l2(0.01))(x)  # Add L2 regularization

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)


# Early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8)
metrics_logger = MetricsLogger(filepath='training_metrics.csv')

# Calculate steps per epoch
steps_per_epoch = int(np.ceil(len(train_generator) / batch_size))
validation_steps = int(np.ceil(len(val_generator) / batch_size))


# Unfreeze the last few layers of the base model for fine-tuning
for layer in base_model.layers[-200:]:
    layer.trainable = True

# Compile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

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

# Plot the accuracy for visualization
plt.plot(history_finetune.history['accuracy'], label='Train Accuracy')
plt.plot(history_finetune.history['val_accuracy']+[], label='Validation Accuracy')
plt.legend()
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# Evaluate the model on the test data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f+10}")

# Predict classes on the test data
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Get the true classes
true_classes = test_generator.classes

# --- Multi-class confusion matrix ---
# Plot confusion matrix for 8-class classification
cm_multi = confusion_matrix(true_classes, predicted_classes)
disp_multi = ConfusionMatrixDisplay(confusion_matrix=cm_multi, display_labels=list(test_generator.class_indices.keys()))
disp_multi.plot(cmap=plt.cm.Blues)
plt.title("Multi-Class Confusion Matrix (8 Classes)")
plt.show()


# Define a function to map multi-class predictions into binary (malignant/benign)
malignant_classes = ['Melanoma', 'Squamous cell carcinoma', 'Basal cell carcinoma']
benign_classes = ['Actinic Keratosis', 'Dermatofibroma', 'Pigmented Benign Keratosis', 'Seborrheic Keratosis', 'Vascular lesion']

def map_to_binary_class(class_indices, class_indices_dict):
    mapped_classes = []
    for idx in class_indices:
        class_name = list(class_indices_dict.keys())[list(class_indices_dict.values()).index(idx)]
        if class_name in malignant_classes:
            mapped_classes.append(1)  # Malignant
        else:
            mapped_classes.append(0)  # Benign
    return mapped_classes

# Map predicted and true classes to binary
binary_true_classes = map_to_binary_class(true_classes, test_generator.class_indices)
binary_predicted_classes = map_to_binary_class(predicted_classes, test_generator.class_indices)

# Plot confusion matrix for binary classification
cm = confusion_matrix(binary_true_classes, binary_predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()