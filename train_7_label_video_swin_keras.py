import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("CPU Available:", tf.config.list_physical_devices('CPU'))
print("TensorFlow is using:", tf.test.gpu_device_name() or "CPU")
import tensorflow_hub as hub
import cv2
import logging
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling3D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple

# --- 1. Setup ---
logging.basicConfig(
    filename='video_classification.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- 2. Label Configuration (Updated) ---
LABEL_CONFIG = {
    'label1': {
        'classes': [0, 1, 2, 4, 5],
        'names': {
            0: "Undefined_Locomotion",
            1: "Stand",
            2: "Walk",
            4: "Sit",
            5: "Lie"
        },
        'csv_col': "label_1"
    },
    'label2': {
        'classes': [0, 101, 102, 103, 104, 105],
        'names': {
            0: "Undefined_HL_Activity",
            101: "Relaxing",
            102: "Coffee time",
            103: "Early morning",
            104: "Cleanup",
            105: "Sandwich time"
        },
        'csv_col': "label_2"
    },
    'label3': {
        'classes': [0] + list(range(201, 214)),
        'names': {0: "Undefined_LL_Left_Arm", **{
            201: "unlock", 202: "stir", 203: "lock", 204: "close",
            205: "reach", 206: "open", 207: "sip", 208: "clean",
            209: "bite", 210: "cut", 211: "spread", 212: "release",
            213: "move"
        }},
        'csv_col': "label_3"
    },
    'label4': {
        'classes': [0] + list(range(301, 324)),
        'names': {0: "Undefined_LL_Left_Arm_Object", **{
            301: "Bottle", 302: "Salami", 303: "Bread", 304: "Sugar",
            305: "Dishwasher", 306: "Switch", 307: "Milk", 308: "Drawer3",
            309: "Spoon", 310: "Knife cheese", 311: "Drawer2", 312: "Table",
            313: "Glass", 314: "Cheese", 315: "Chair", 316: "Door1",
            317: "Door2", 318: "Plate", 319: "Drawer1", 320: "Fridge",
            321: "Cup", 322: "Knife salami", 323: "Lazychair"
        }},
        'csv_col': "label_4"
    },
    'label5': {
        'classes': [0] + list(range(401, 414)),
        'names': {0: "Undefined_LL_Right_Arm", **{
            401: "unlock", 402: "stir", 403: "lock", 404: "close",
            405: "reach", 406: "open", 407: "sip", 408: "clean",
            409: "bite", 410: "cut", 411: "spread", 412: "release",
            413: "move"
        }},
        'csv_col': "label_5"
    },
    'label6': {
        'classes': [0] + list(range(501, 524)),
        'names': {0: "Undefined_LL_Right_Arm_Object", **{
            501: "Bottle", 502: "Salami", 503: "Bread", 504: "Sugar",
            505: "Dishwasher", 506: "Switch", 507: "Milk", 508: "Drawer3",
            509: "Spoon", 510: "Knife cheese", 511: "Drawer2", 512: "Table",
            513: "Glass", 514: "Cheese", 515: "Chair", 516: "Door1",
            517: "Door2", 518: "Plate", 519: "Drawer1", 520: "Fridge",
            521: "Cup", 522: "Knife salami", 523: "Lazychair"
        }},
        'csv_col': "label_6"
    },
    'label7': {
        'classes': [0, 406516, 406517, 404516, 404517, 406520, 404520, 
                   406505, 404505, 406519, 404519, 406511, 404511,
                   406508, 404508, 408512, 407521, 405506],
        'names': {
            0: "Undefined_ML_Both_Arms",
            406516: "Open Door 1", 406517: "Open Door 2",
            404516: "Close Door 1", 404517: "Close Door 2",
            406520: "Open Fridge", 404520: "Close Fridge",
            406505: "Open Dishwasher", 404505: "Close Dishwasher",
            406519: "Open Drawer 1", 404519: "Close Drawer 1",
            406511: "Open Drawer 2", 404511: "Close Drawer 2",
            406508: "Open Drawer 3", 404508: "Close Drawer 3",
            408512: "Clean Table", 407521: "Drink from Cup",
            405506: "Toggle Switch"
        },
        'csv_col': "label_7"
    }
}

# --- 1. Enhanced Video Loader ---
def load_video_frames_fast(video_path: str, num_frames: int = 32, img_size: int = 224) -> np.ndarray:
    """Fast video loader: read all frames once, then sample"""
    try:
        print(f"Loading video: {video_path}")
        if not os.path.exists(video_path):
            print(f"[ERROR] File not found: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            return None

        # Read all frames at once
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
            all_frames.append(frame / 255.0)

        cap.release()
        total_frames = len(all_frames)
        print(f"Total frames read: {total_frames}")

        if total_frames == 0:
            print("[ERROR] Video contains 0 frames")
            return None

        # Sampling or padding
        if total_frames < num_frames:
            print(f"[INFO] Padding {total_frames} → {num_frames}")
            all_frames += [np.zeros((img_size, img_size, 3))] * (num_frames - total_frames)
            selected_frames = all_frames
        else:
            indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
            selected_frames = [all_frames[i] for i in indices]

        video_array = np.stack(selected_frames)
        print(f"Final shape: {video_array.shape}")
        return video_array

    except Exception as e:
        print(f"[CRITICAL] Error: {str(e)}")
        return None

# --- 2. Build Model with Feature Extractor ---
def build_multihead_model_with_feature_extractor(model_path: str):
    # Configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Input layer for raw video frames
    input_layer = Input(shape=(32, 224, 224, 3), name="video_input")  # [T, H, W, C]
    
    # Load TensorFlow Hub model as a Keras Layer
    feature_extractor = hub.KerasLayer(
        model_path,
        trainable=False,
        name="video_swin_transformer"
    )
    
    # Lambda layer to transpose dimensions from [B, T, H, W, C] to [B, C, T, H, W]
    def extract_features(x):
        x = tf.transpose(x, perm=[0, 4, 1, 2, 3])  # [B, C, T, H, W]
        return feature_extractor(x)
    
    # Apply extraction through Lambda layer
    x = tf.keras.layers.Lambda(extract_features, name="swin_feature_extractor")(input_layer)
    
    # Global pooling to reduce dimensions
    x = GlobalAveragePooling3D()(x)
    
    # Shared layers
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Task-specific outputs
    outputs = []
    for name, cfg in LABEL_CONFIG.items():
        y = Dense(512, activation='relu', name=f"{name}_dense")(x)
        y = Dropout(0.3)(y)
        outputs.append(
            Dense(len(cfg['classes']), activation='softmax', name=name)(y)
        )
    
    return Model(inputs=input_layer, outputs=outputs)

def compile_multihead_model(model):
    losses = {f'label{i+1}': 'sparse_categorical_crossentropy' for i in range(7)}
    metrics = {f'label{i+1}': ['accuracy'] for i in range(7)}
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9
        )
    )
    
    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=metrics
    )

    print("Model successfully compiled")
    print(f"Loss functions: {list(losses.keys())}")
    print(f"Metrics: {list(metrics.keys())}")
    
    return model

# --- 3. Debug-Enhanced Data Pipeline ---
def prepare_dataset(csv_path: str, video_dir: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Prepares dataset with proper input/output structure"""
    print("\n" + "="*50)
    print("START DATASET PREPARATION")
    print("="*50)
    
    try:
        # 1. Load CSV
        print(f"\n[STEP 1] Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"CSV loaded successfully. Entries: {len(df)}")
        print("Sample:\n", df.head(3))

        # 2. Process labels
        print("\n[STEP 2] Processing labels")
        for name, cfg in LABEL_CONFIG.items():
            col = cfg['csv_col']
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in CSV")
            
            df[col] = df[col].fillna(0).astype(int)
            unique_vals = df[col].unique()
            print(f"Label {name} - Unique values: {unique_vals}")

        # 3. Validate labels
        print("\n[STEP 3] Validating labels")
        valid_mask = np.ones(len(df), dtype=bool)
        for name, cfg in LABEL_CONFIG.items():
            valid = df[cfg['csv_col']].isin(cfg['classes'])
            valid_mask &= valid
            print(f"Label {name} - Invalid values: {sum(~valid)}")
        
        df = df[valid_mask].copy()
        print(f"Valid videos: {len(df)}")

        # 4. Extract video frames (no feature extraction yet)
        print("\n[STEP 4] Loading video frames")
        frames_list, labels = [], {name: [] for name in LABEL_CONFIG}
        success = 0

        for idx, row in df.iterrows():
            video_path = os.path.join(video_dir, row['filename'])
            print(f"\nProcessing video {idx+1}/{len(df)}: {video_path}")
            
            if not os.path.exists(video_path):
                print("Skipping - video file not found")
                continue
                
            frames = load_video_frames_fast(video_path)
            if frames is None:
                print("Skipping - frame extraction failed")
                continue
            
            try:
                frames_list.append(frames)
                
                for name, cfg in LABEL_CONFIG.items():
                    labels[name].append(cfg['classes'].index(row[cfg['csv_col']]))
                
                success += 1
                print("Successfully processed")
            except Exception as e:
                print(f"Processing failed: {str(e)}")
                continue

        # 5. Final validation
        print("\n[STEP 5] Final validation")
        if success == 0:
            raise ValueError("No videos could be processed")
        
        print(f"Success rate: {success}/{len(df)}")
        frames_array = np.stack(frames_list)
        labels_dict = {k: np.array(v) for k, v in labels.items()}
        
        # Return as dictionary matching model input/output names
        data_dict = {
            'video_input': frames_array,
            **{f'label{i+1}': labels_dict[f'label{i+1}'] for i in range(7)}
        }
        
        print("\nFinal shapes:")
        print(f"Video frames: {frames_array.shape}")
        for name, arr in labels_dict.items():
            print(f"{name}: {arr.shape}")
        
        return data_dict

    except Exception as e:
        print(f"Critical error: {str(e)}")
        raise

def plot_training_history(history):
    plt.figure(figsize=(15, 10))
    
    # Extrahiere alle Label-Namen aus den history-Keys
    label_names = set()
    for key in history.history.keys():
        if key.endswith('_loss') and not key.startswith('val_'):
            name = key[:-5]  # Entfernt '_loss'
            label_names.add(name)
    
    # Plot losses
    plt.subplot(2, 1, 1)
    for name in label_names:
        plt.plot(history.history[f'{name}_loss'], label=f'{name} train')
        if f'val_{name}_loss' in history.history:
            plt.plot(history.history[f'val_{name}_loss'], label=f'{name} val')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(2, 1, 2)
    for name in label_names:
        if f'{name}_accuracy' in history.history:
            plt.plot(history.history[f'{name}_accuracy'], label=f'{name} train')
        if f'val_{name}_accuracy' in history.history:
            plt.plot(history.history[f'val_{name}_accuracy'], label=f'{name} val')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()  # Zeigt den Plot an (optional)
    plt.close()
# --- 4. Main Training Pipeline ---
def train_model():
    print("\n" + "="*50)
    print("START TRAINING PIPELINE")
    print("="*50)
    
    # Configuration
    CSV_PATH = os.path.abspath("dataset/labels_s.csv")
    VIDEO_DIR = os.path.abspath("dataset/snippets")
    MODEL_PATH = os.path.abspath("video_swin_model/saved_model")
    
    print("\nConfiguration:")
    print(f"CSV: {CSV_PATH}")
    print(f"Video Dir: {VIDEO_DIR}")
    print(f"Model Path: {MODEL_PATH}")
    
    # Validate paths
    print("\nValidating paths...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    if not os.path.exists(VIDEO_DIR):
        raise FileNotFoundError(f"Video directory not found: {VIDEO_DIR}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    print("All paths validated")
    
    # Prepare dataset
    print("\nPreparing dataset...")
    data = prepare_dataset(CSV_PATH, VIDEO_DIR)
    
    # Split data
    print("\nSplitting data...")
    train_idx, val_idx = train_test_split(range(len(data['video_input'])), test_size=0.2, random_state=42)
    
    train_data = {
        'video_input': data['video_input'][train_idx],
        **{f'label{i+1}': data[f'label{i+1}'][train_idx] for i in range(7)}
    }
    
    val_data = {
        'video_input': data['video_input'][val_idx],
        **{f'label{i+1}': data[f'label{i+1}'][val_idx] for i in range(7)}
    }
    
    print(f"Training samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    
    # Build and compile model
    print("\nBuilding model...")
    model = build_multihead_model_with_feature_extractor(MODEL_PATH)
    print("\nCompile model...")
    model = compile_multihead_model(model)
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(log_dir='./logs')     # Open with tensorboard --logdir="your_Path" in CMD
    ]
    
    # Start training
    print("\nStarting training...")
    history = model.fit(
        x=train_data['video_input'],
        y={k: v for k, v in train_data.items() if k != 'video_input'},
        validation_data=(
            val_data['video_input'],
            {k: v for k, v in val_data.items() if k != 'video_input'}
        ),
        epochs=50,
        batch_size=8,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save and evaluate
    print("\nSaving final model...")
    model.save("7label_video_classifier_123.keras")
    
    print("\nEvaluating...")
    results = model.evaluate(
        val_data['video_input'],
        {k: v for k, v in val_data.items() if k != 'video_input'},
        verbose=0
    )
    
    print("\nTraining complete!")
    print("Final validation metrics:")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")
    
    # Plot training history
    plot_training_history(history)

if __name__ == "__main__":
    train_model()