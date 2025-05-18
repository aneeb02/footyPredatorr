# compress_models.py
import joblib
import os

# paths to your original pickles
ORIG_MODEL = 'player_position_model.pkl'
ORIG_ENCODER = 'label_encoder.pkl'

# paths for the compressed outputs
COMP_MODEL = 'player_position_model_compressed.joblib'
COMP_ENCODER = 'label_encoder_compressed.joblib'

print(f"Loading original model from {ORIG_MODEL}…")
model = joblib.load(ORIG_MODEL)

print(f"Loading original encoder from {ORIG_ENCODER}…")
encoder = joblib.load(ORIG_ENCODER)

print("Dumping compressed model…")
joblib.dump(model, COMP_MODEL, compress=('gzip', 3))

print("Dumping compressed encoder…")
joblib.dump(encoder, COMP_ENCODER, compress=('gzip', 3))

print("Done! Compressed files saved as:")
print(f"  • {COMP_MODEL} ({round(os.path.getsize(COMP_MODEL)/1024/1024,2)} MB)")
print(f"  • {COMP_ENCODER} ({round(os.path.getsize(COMP_ENCODER)/1024/1024,2)} MB)")
