from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the path to your dataset
dataset_directory = 'D:/MP 1 FINAL/ML5/archive/food-101/food-101/images'  # Replace with the actual path

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Split 80% for training, 20% for validation
    horizontal_flip=True,
    zoom_range=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_directory,  # Path to the dataset
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(101, activation='softmax')(x)  # 101 food categories

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze initial layers and train only the top layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, validation_data=validation_generator, epochs=10)
calorie_mapping = {
    "pizza": 266,
    "burger": 295,
    # Add mappings for all 101 classes
}
predicted_class = "pizza"  # Example output from the model
calories = calorie_mapping.get(predicted_class, "Unknown")
print(f"Estimated calories: {calories} kcal")
