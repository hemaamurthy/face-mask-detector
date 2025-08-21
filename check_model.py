from keras.models import load_model

MODEL_PATH = r"C:\Users\91812\OneDrive\Desktop\face-mask-detector\models\mask_detector.h5"
model = load_model(MODEL_PATH)

print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)
