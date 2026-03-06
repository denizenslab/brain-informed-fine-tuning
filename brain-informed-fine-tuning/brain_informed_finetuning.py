# %%
import torch
import torch.optim as optim
import numpy as np
from transformers import BertTokenizer, BertModel
from datasets import Dataset
import os
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from utils_resp import ResponseUtils
import torch.nn as nn
from utils_resp import ResponseUtils
from scipy.stats import pearsonr
from torch.utils.data import Dataset
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils_ridge.utils_stim import get_story_wordseqs
from config import DATA_PATH, GRIDS_DIR, TRFILES_DIR

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# %%
# ---- CONFIGURATION ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
LR = 5e-5  # Learning rate 
EPOCHS = 20
utilresonses = ResponseUtils()
# %%
RANDOM_SEED = 42
def set_seed(seed=RANDOM_SEED):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_brain_data(subject, modality):

    # Load response data for training
    rresp_en_subj1, rresp_test_subj1_en = utilresonses.load_subject_fMRI(DATA_PATH, subject, modality)
    return rresp_en_subj1, rresp_test_subj1_en

def create_20_word_contexts(text_array, seq_len=20):
    """
    Create context windows from a word sequence.
    
    - For the first `seq_len` words, create growing contexts (1 to seq_len words).
    - Then create fixed-size sliding windows of length `seq_len`.

    Args:
        text_array: List of words (length N)
        seq_len: Context window size (default: 20)

    Returns:
        context_windows: List of word lists. Each is a sequence of ≤ seq_len words.
    """
    context_windows = []

    # Phase 1: Growing context
    for i in range(1, seq_len + 1):
        context_windows.append(text_array[:i])

    # Phase 2: Fixed sliding context
    for i in range(seq_len, len(text_array)):
        context = text_array[i - seq_len + 1 : i + 1]
        context_windows.append(context)

    return context_windows

class BERTContextDataset(Dataset):
    def __init__(self, context_windows, tokenizer, max_length=20):
        self.contexts = context_windows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        context = list(self.contexts[idx])  # Ensure it's a List[str]

        tokenized = self.tokenizer(
            context,
            is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0)
        }

def nt_xent_loss(
    z1, 
    z2, 
    model=None, 
    temperature=0.5
):
    """
    Robust Normalized Temperature-scaled Cross Entropy Loss
    
    Args:
        z1 (torch.Tensor): First set of embeddings
        z2 (torch.Tensor): Second set of embeddings
        model (nn.Module, optional): Model (not used)
        temperature (float): Temperature scaling parameter
    
    Returns:
        torch.Tensor: Contrastive loss
    """
    # Ensure inputs are tensors with float32 type
    z1 = z1.float() if isinstance(z1, torch.Tensor) else torch.tensor(z1, dtype=torch.float32)
    z2 = z2.float() if isinstance(z2, torch.Tensor) else torch.tensor(z2, dtype=torch.float32)
    
    # Reshape if needed (ensure 2D tensor)
    if z1.dim() > 2:
        z1 = z1.reshape(z1.size(0), -1)
    if z2.dim() > 2:
        z2 = z2.reshape(z2.size(0), -1)
    
    # Normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Ensure same batch size
    batch_size = min(z1.size(0), z2.size(0))
    z1 = z1[:batch_size]
    z2 = z2[:batch_size]
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(z1, z2.T) / temperature
    
    # Compute positive similarities (diagonal)
    pos_sim = torch.diag(sim_matrix)
    
    # Compute negative similarities (all pairs except diagonal)
    neg_sim = sim_matrix.sum(1) - pos_sim
    
    # Compute loss
    loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim) + 1e-8)).mean()
    
    return loss

# 3️⃣ **Spatial Correlation Loss**
def spatial_correlation_loss(predicted, actual, model=None):
    """
    Computes correlation loss across voxels using PyTorch.
    """
    pred_mean = predicted.mean(dim=0, keepdim=True)
    actual_mean = actual.mean(dim=0, keepdim=True)
    
    pred_std = predicted.std(dim=0, keepdim=True) + 1e-6
    actual_std = actual.std(dim=0, keepdim=True) + 1e-6
    
    numerator = ((predicted - pred_mean) * (actual - actual_mean)).sum(dim=0)
    denominator = pred_std * actual_std * predicted.shape[0]

    correlation = numerator / denominator
    return -correlation.mean()  # Negative correlation for minimization

def hybrid_loss(predicted, actual, model=None, lambda_ridge=0.01, alpha_corr=0.5):
    """
    Combines Spatial Correlation Loss and MSE without explicit regularization.
    - alpha_corr: Weight for spatial correlation (0.5 is balanced).
    """
    # Compute Spatial Correlation Loss
    pred_mean = predicted.mean(dim=0, keepdim=True)
    actual_mean = actual.mean(dim=0, keepdim=True)
    
    pred_std = predicted.std(dim=0, keepdim=True) + 1e-6
    actual_std = actual.std(dim=0, keepdim=True) + 1e-6
    
    correlation = ((predicted - pred_mean) * (actual - actual_mean)).sum(dim=0) / (pred_std * actual_std)
    spatial_corr_loss = -correlation.mean()  # Negative correlation for minimization
    
    # MSE Loss
    mse_loss = torch.nn.functional.mse_loss(predicted, actual)
    
    # Ridge Regularization (L2)
    l2_reg = sum(torch.sum(param ** 2) for param in model.parameters())

    # Final Loss: Balance all components
    return (alpha_corr * spatial_corr_loss) + ((1 - alpha_corr) * mse_loss) + (lambda_ridge * l2_reg)

# 2️⃣ **Ridge Loss (MSE + L2 Regularization)**
def ridge_loss(output, target, model, lambda_ridge=0.01):
    mse_loss = F.mse_loss(output, target)  # Standard MSE loss
    l2_reg = sum(torch.sum(param ** 2) for param in model.parameters())  # L2 penalty
    return mse_loss + lambda_ridge * l2_reg  # Combined loss

# 1️⃣ **Mean Squared Error (MSE) Loss**
def mse_loss(predicted, actual, model=None):
    return F.mse_loss(predicted, actual)

def compute_cka_mean_pooling(weights1, weights2):
    """
    Compute CKA using mean-pooled layer activations.
    """
    X = weights1.detach().cpu().numpy()
    Y = weights2.detach().cpu().numpy()
    print(X.shape, Y.shape)

    # Mean pool along the feature dimension (reduce dimensionality)
    X = np.mean(X, axis=0, keepdims=True)
    Y = np.mean(Y, axis=0, keepdims=True)

    # Compute Gram Matrices
    K = np.dot(X, X.T)
    L = np.dot(Y, Y.T)

    # Centering
    K -= np.mean(K, axis=0, keepdims=True)
    L -= np.mean(L, axis=0, keepdims=True)

    # Compute CKA
    numerator = np.trace(np.dot(K, L))
    denominator = np.sqrt(np.trace(np.dot(K, K)) * np.trace(np.dot(L, L)))
    
    return numerator / denominator if denominator > 0 else 0.0  # Avoid division by zero

def sinc(x):
    """Differentiable sinc function with proper handling of x=0"""
    # Handle x=0 case to avoid NaN
    x = x.clone()  # Create a copy to avoid modifying the input
    mask = (x == 0)
    x[mask] = 1.0  # Temporarily set to non-zero
    result = torch.sin(torch.pi * x) / (torch.pi * x)
    result[mask] = 1.0  # Set correct value for x=0
    return result

def lanczos_kernel(x, a=3):
    """Differentiable Lanczos kernel implementation"""
    # Handle out-of-window cases
    mask = (torch.abs(x) < a)
    result = torch.zeros_like(x)
    x_window = x[mask]
    
    # Calculate the kernel only for values within the window
    if x_window.numel() > 0:
        result[mask] = sinc(x_window) * sinc(x_window / a)
    
    return result

def differentiable_lanczosinterp2D(data, source_times, target_times, a=3):
    """
    Differentiable PyTorch implementation of Lanczos interpolation.
    
    Args:
        data (torch.Tensor): Input tensor [time_points, features]
        source_times (np.ndarray or torch.Tensor): Original time points
        target_times (np.ndarray or torch.Tensor): Target time points for resampling
        a (int): Lanczos kernel parameter
    
    Returns:
        torch.Tensor: Interpolated data at target times
    """
    # Ensure inputs are PyTorch tensors
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    if not isinstance(source_times, torch.Tensor):
        source_times = torch.tensor(source_times, dtype=torch.float32, device=data.device)
    if not isinstance(target_times, torch.Tensor):
        target_times = torch.tensor(target_times, dtype=torch.float32, device=data.device)
    
    n_targets = len(target_times)
    n_sources = len(source_times)
    
    # Scale factor between source and target sampling rates
    scale = (source_times[-1] - source_times[0]) / (target_times[-1] - target_times[0]) if target_times[-1] != target_times[0] else 1.0
    
    # Initialize interpolation matrix
    interpmat = torch.zeros((n_targets, n_sources), device=data.device)
    
    # Compute interpolation weights
    for i in range(n_targets):
        t = target_times[i]
        
        # Compute normalized distances to the target point
        dists = (t - source_times) * scale
        
        # Apply Lanczos kernel to get weights
        weights = lanczos_kernel(dists, a=a)
        
        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        interpmat[i] = weights
    
    # Apply interpolation
    result = torch.matmul(interpmat, data)
    return result

def differentiable_make_delayed(features, delays=[0, 1, 2, 3]):
    """
    Create delayed versions of features using PyTorch operations.
    
    Args:
        features (torch.Tensor): Input features [time_points, feature_dim]
        delays (list): List of delay steps
    
    Returns:
        torch.Tensor: Concatenated delayed features
    """
    n_samples, n_features = features.shape
    device = features.device
    delayed_features = []
    
    for delay in delays:
        if delay == 0:
            # No delay
            delayed_features.append(features)
        else:
            # Create delayed version
            padding = torch.zeros(delay, n_features, device=device)
            if delay < n_samples:
                # Shift and pad with zeros
                delayed = torch.cat([padding, features[:-delay]], dim=0)
            else:
                # All padding if delay exceeds sample count
                delayed = torch.zeros_like(features)
            delayed_features.append(delayed)
    
    # Concatenate along feature dimension
    return torch.cat(delayed_features, dim=1)


# ---- BERT MODEL WITH PROJECTION ----
class BertToBrain(nn.Module):
    def __init__(self, model_hf_path=None, output_dim=100, dropout_rate=0.2, delay_factor=4):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_hf_path)
        self.bert.gradient_checkpointing_enable()
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(dropout_rate)
        self.regression = nn.Linear(768 * delay_factor, output_dim)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_token_vec = outputs.last_hidden_state[:, -1, :]  # last token representation
        x = self.dropout(last_token_vec)
        return x

# ---- END-TO-END MODEL ----
class EndToEndBertBrain(nn.Module):
    """
    End-to-end model for BERT to brain mapping with differentiable operations.
    Wraps the original BertToBrain model and adds differentiable operations.
    """
    def __init__(self, bert_brain_model):
        super().__init__()
        self.model = bert_brain_model
        self.device = next(bert_brain_model.parameters()).device
    
    def process_story(self, input_ids, attention_mask, word_seqs, story_name):
        """
        Process a single story with differentiable operations.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            word_seqs (dict): Word sequence information
            story_name (str): Name of the story
        
        Returns:
            torch.Tensor: Processed features
        """
        # Get embeddings from BERT
        embeddings = self.model(input_ids, attention_mask=attention_mask)
        
        # Convert data_times and tr_times to tensors for interpolation
        data_times = torch.tensor(
            word_seqs[story_name].data_times, 
            dtype=torch.float32, 
            device=self.device
        )
        tr_times = torch.tensor(
            word_seqs[story_name].tr_times, 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Differentiable downsampling
        downsampled = differentiable_lanczosinterp2D(
            embeddings, data_times, tr_times
        )
        
        # Apply trimming
        trim = 5
        trimmed = downsampled[5+trim:-trim-5]
        
        # Apply normalization (similar to zscore in the original code)
        mean = torch.mean(trimmed, dim=0, keepdim=True)
        std = torch.std(trimmed, dim=0, keepdim=True) + 1e-6  # Avoid div by zero
        normalized = (trimmed - mean) / std
        
        return normalized
    
    def forward(self, stories_input_ids, stories_attention_mask, 
                word_seqs, story_names):
        """
        Forward pass through the entire model while maintaining gradients.
        
        Args:
            stories_input_ids (list): List of input_ids tensors for each story
            stories_attention_mask (list): List of attention_mask tensors for each story
            word_seqs (dict): Word sequence information for all stories
            story_names (list): Names of stories to process
        
        Returns:
            torch.Tensor: fMRI predictions
            torch.Tensor: Processed features before regression
        """
        # Process each story with gradients maintained
        processed_stories = []
        
        for i, story_name in enumerate(story_names):
            story_features = self.process_story(
                stories_input_ids[i],
                stories_attention_mask[i],
                word_seqs,
                story_name
            )
            processed_stories.append(story_features)
        
        # Combine all stories
        if processed_stories:
            all_features = torch.cat(processed_stories, dim=0)
            
            # Create delayed features
            delayed_features = differentiable_make_delayed(all_features)
            
            # Apply regression to get fMRI predictions
            fmri_preds = self.model.regression(delayed_features)
            
            return fmri_preds, delayed_features
        else:
            # Handle empty case
            return None, None

def predict_on_test_data(model, test_loader, subject, modality, story_name):
    """
    Generate predictions on test data using the trained model and DataLoader.
    
    Args:
        model (BertToBrain): Trained model
        test_loader (DataLoader): DataLoader containing test context windows
        subject (str): Subject identifier
        modality (str): Data modality (e.g. 'listening', 'reading')
        story_name (str): Name of the test story
    
    Returns:
        torch.Tensor: Processed test features ready for evaluation
        torch.Tensor: Model predictions for test data
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Load story word sequences (current dataset uses English stimuli)
    word_seqs = get_story_wordseqs([story_name], GRIDS_DIR, TRFILES_DIR)
    
    print(f"Processing test story: {story_name}")
    
    # Get all test embeddings in order
    all_embeddings = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            
            # Get embeddings from the model
            embeddings = model(input_ids, attention_mask=attn_mask)
            all_embeddings.append(embeddings)
    
    # Combine all embeddings
    story_tensor = torch.cat(all_embeddings, dim=0)
    print(f"Extracted test embeddings: {story_tensor.shape}")
    
    # Downsample to fMRI time resolution
    data_times = torch.tensor(word_seqs[story_name].data_times, dtype=torch.float32, device=device)
    tr_times = torch.tensor(word_seqs[story_name].tr_times, dtype=torch.float32, device=device)
    
    print(f"Downsampling test data:")
    print(f"  Story tensor shape: {story_tensor.shape}")
    print(f"  Data times shape: {data_times.shape}")
    print(f"  TR times shape: {tr_times.shape}")
    
    # Use differentiable interpolation
    downsampled = differentiable_lanczosinterp2D(story_tensor, data_times, tr_times)
    
    # Apply trimming
    trim = 5
    trimmed = downsampled[5+trim:-trim-5]
    
    # Apply normalization
    mean = torch.mean(trimmed, dim=0, keepdim=True)
    std = torch.std(trimmed, dim=0, keepdim=True) + 1e-6
    normalized = (trimmed - mean) / std
    
    # Create delayed features
    delayed_features = differentiable_make_delayed(normalized)
    print(f"Processed test features shape: {delayed_features.shape}")
    
    # Generate predictions
    with torch.no_grad():
        predictions = model.regression(delayed_features)
        print(f"Test predictions shape: {predictions.shape}")
    
    return delayed_features, predictions

# ---- TRAINING FUNCTION ----
def train_model_end_to_end(
    model, train_loader, fmri_data, story_window_counts, 
    loss_fn=spatial_correlation_loss, modality='reading', subject='Subj1', 
    tune='all', batch_size=4, epochs=EPOCHS, 
    lr=1e-4, weight_decay=1e-3, val_split=0.2, patience=5
):
    """End-to-end training function for BertToBrain model with full gradient flow."""
    set_seed()  # Ensure reproducibility
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Set up parameter gradients based on tuning strategy
    if tune == 'all':
        print("Enabling gradients for all parameters (full fine-tuning)")
        for param in model.parameters():
            param.requires_grad = True
    elif tune == 'last':
        print("Fine-tuning only the last transformer layer and regression layer")
        for name, param in model.bert.named_parameters():
            if "encoder.layer.11" not in name:
                param.requires_grad = False
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, 
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Load story information
    print("Preparing word sequences...")
    stim_language = 'en'
    Rstories = ResponseUtils().stories[stim_language]
    Pstories = ResponseUtils().test_story
    allstories = Rstories + Pstories
    word_seqs = get_story_wordseqs(allstories, GRIDS_DIR, TRFILES_DIR)
    
    # First, track which stories each batch belongs to
    input_features = []
    feature_index_map = []
    story_start_indices = [0]
    cumulative_sum = 0
    
    for i in range(len(story_window_counts)):
        cumulative_sum += story_window_counts[i]
        story_start_indices.append(cumulative_sum)
    
    # Train/validation split for fMRI data
    Y_full = torch.tensor(fmri_data, dtype=torch.float32)
    total_samples = len(fmri_data)
    split_idx = int(total_samples * (1 - val_split))
    train_indices = np.arange(split_idx)
    val_indices = np.arange(split_idx, total_samples)
    Y_train = Y_full[train_indices].to(device)
    Y_val = Y_full[val_indices].to(device)
    
    # Training tracking variables
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch+1}/{epochs} =====")
        model.train()
        epoch_loss = 0.0
        
        # Clear previous data
        input_features = []
        feature_index_map = []
        
        # Phase 1: Forward pass through BERT to get embeddings for each story
        print("Phase 1: Processing story features...")
        batch_idx = 0
        
        # Process each batch and track which story it belongs to
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            
            # Track which story each batch output belongs to
            batch_size = input_ids.size(0)
            batch_start = batch_idx
            batch_end = batch_idx + batch_size
            
            # Find which story(ies) this batch belongs to
            for i in range(len(story_start_indices)-1):
                if batch_start < story_start_indices[i+1] and batch_end > story_start_indices[i]:
                    # This batch contains elements from story i
                    # Record which positions in this batch belong to which story
                    for pos in range(batch_size):
                        global_pos = batch_start + pos
                        if global_pos >= story_start_indices[i] and global_pos < story_start_indices[i+1]:
                            feature_index_map.append((len(input_features), pos, i, global_pos - story_start_indices[i]))
            
            # Store the input batch for later use (without detaching - keep gradients!)
            input_features.append((input_ids, attn_mask))
            batch_idx += batch_size
            
            print(f"Processed batch: {batch_idx}/{sum(story_window_counts)}", end="\r")
        
        print("\nPhase 2: Processing stories one by one with backpropagation...")
        
        # Group features by story
        story_features = [[] for _ in range(len(story_window_counts))]
        for batch_idx, pos_in_batch, story_idx, pos_in_story in feature_index_map:
            story_features[story_idx].append((batch_idx, pos_in_batch, pos_in_story))
        
        # Process each story independently and maintain gradients
        all_downsampled_features = []
        
        # Set batch accumulation for optimization
        accumulation_steps = 1
        optimizer.zero_grad()
        
        for story_idx in range(len(story_features)):
            if not story_features[story_idx]:
                continue
                
            print(f"Processing story {story_idx+1}/{len(story_features)}: {Rstories[story_idx]}")
            
            # Forward pass for this story's features
            story_embeddings = []
            for batch_idx, pos_in_batch, pos_in_story in story_features[story_idx]:
                input_ids, attn_mask = input_features[batch_idx]
                
                # Extract just this position's inputs (maintain the original tensors)
                single_input_ids = input_ids[pos_in_batch:pos_in_batch+1]
                single_attn_mask = attn_mask[pos_in_batch:pos_in_batch+1]
                
                # Forward pass with gradients preserved
                with torch.amp.autocast(device_type='cuda'):
                    embedding = model(single_input_ids, attention_mask=single_attn_mask)
                    story_embeddings.append(embedding)
            
            # Concatenate all embeddings for this story
            story_tensor = torch.cat(story_embeddings, dim=0)
            
            # Downsample this story's features (differentiable version)
            # Convert time points to tensors
            story = Rstories[story_idx]
            data_times = torch.tensor(word_seqs[story].data_times, dtype=torch.float32, device=device)
            tr_times = torch.tensor(word_seqs[story].tr_times, dtype=torch.float32, device=device)
            
            # Print shapes for debugging
            print(f"Story tensor shape: {story_tensor.shape}")
            print(f"Data times shape: {data_times.shape}")
            print(f"TR times shape: {tr_times.shape}")
            
            # Use differentiable interpolation
            downsampled = differentiable_lanczosinterp2D(story_tensor, data_times, tr_times)
            
            # Apply trimming
            trim = 5
            trimmed = downsampled[5+trim:-trim-5]
            
            # Apply normalization
            mean = torch.mean(trimmed, dim=0, keepdim=True)
            std = torch.std(trimmed, dim=0, keepdim=True) + 1e-6
            normalized = (trimmed - mean) / std
            
            all_downsampled_features.append(normalized)
        
        # Combine all downsampled features
        if all_downsampled_features:
            all_features = torch.cat(all_downsampled_features, dim=0)
            
            # Create delayed features (differentiable version)
            print("Creating delayed features...")
            delayed_features = differentiable_make_delayed(all_features)
            
            # Split for train/validation
            X_train_indices = train_indices
            X_val_indices = val_indices
            
            # Make sure indices don't exceed the available data
            X_train_indices = X_train_indices[X_train_indices < delayed_features.shape[0]]
            X_val_indices = X_val_indices[X_val_indices < delayed_features.shape[0]]
            
            # Extract features for training/validation
            X_train_features = delayed_features[X_train_indices]
            Y_train_batch = Y_train[:len(X_train_indices)]
            
            # Forward pass through regression layer
            print("Phase 3: Full end-to-end backpropagation through entire model...")
            with torch.amp.autocast(device_type='cuda'):
                predictions = model.regression(X_train_features)
                loss = loss_fn(predictions, Y_train_batch, model)
            
            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Track loss
            epoch_loss = loss.item()
            
            # Validation
            print("Phase 4: Validation...")
            model.eval()
            with torch.no_grad():
                X_val_features = delayed_features[X_val_indices]
                Y_val_batch = Y_val[:len(X_val_indices)]
                
                with torch.amp.autocast(device_type='cuda'):
                    val_predictions = model.regression(X_val_features)
                    val_loss = loss_fn(val_predictions, Y_val_batch, model)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss.item():.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"⛔ Early stopping at epoch {epoch+1}")
                    break
        
        # Force garbage collection after each epoch
        # Free memory
        del X_train_features, X_val_features, delayed_features, all_features
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    # Load best model
    if best_model_state:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print("✅ Loaded best model (lowest validation loss).")
    
    return model

# %%
# ------------------- CORRELATION FUNCTION -------------------
def compute_correlations(matrix1, matrix2):
    """
    Compute Pearson correlation coefficients between corresponding columns of two matrices.
    
    Args:
        matrix1 (np.ndarray): First matrix of shape (n_samples, n_features).
        matrix2 (np.ndarray): Second matrix of shape (n_samples, n_features).
    
    Returns:
        np.ndarray: A 1D array of correlation coefficients (size n_features).
    """
    if matrix1.shape != matrix2.shape:
        raise ValueError("Input matrices must have the same shape.")

    mean1 = matrix1.mean(axis=0)
    mean2 = matrix2.mean(axis=0)
    std1 = matrix1.std(axis=0)
    std2 = matrix2.std(axis=0)

    numerator = ((matrix1 - mean1) * (matrix2 - mean2)).sum(axis=0)
    denominator = std1 * std2 * matrix1.shape[0]

    correlations = numerator / denominator
    return np.nan_to_num(correlations)

# %%
# ---- EVALUATION FUNCTION ----
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on test data.
    Computes Pearson Correlation, Spatial Correlation, and Mean Absolute Error (MAE).
    """
    model.eval()  # Set to evaluation mode
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)  # (Test Size, 4, 768)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)  # (Test Size, 70k+)    with torch.no_grad():
    predicted_fmri = model(X_test_tensor)  # Predict fMRI responses    # Convert to NumPy
    predicted_fmri_np = predicted_fmri.detach().cpu().numpy()
    y_test_np = y_test_tensor.detach().cpu().numpy()    # Compute Pearson Correlation for each voxel
    print(predicted_fmri_np.shape, y_test_np.shape)
    pearson_corrs = compute_correlations(predicted_fmri_np, y_test_np[:, :])
    mean_pearson_corr = np.mean(pearson_corrs)  
    print(f"Test Pearson Correlation (mean over voxels): {mean_pearson_corr:.6f}")
    return mean_pearson_corr, pearson_corrs


# ---- MAIN FUNCTION ----
def main_end_to_end(all_contexts, fmri_data, loss_fn, modality, model_name, subject, fine_tune_layers='all'):
    """
    Main function to run the end-to-end training.
    
    Args:
        all_contexts (list): List of context windows for all stories
        fmri_data (np.ndarray): Target fMRI data
        loss_fn (callable): Loss function
        modality (str): Data modality (e.g. 'listening', 'reading')
        model_name (str): Name of the model
        subject (str): Subject identifier
        fine_tune_layers (str): Tuning strategy ('all' or 'last')
    
    Returns:
        BertToBrain: Trained model and test data
    """
    # Prepare training data
    train_contexts = []
    train_indices = [0, 1, 2, 3, 4, 6, 9]  # Your specific indices
    test_indices = [10]
    
    for i in train_indices:
        train_contexts.extend(all_contexts[i])
    
    story_window_counts = [len(all_contexts[i]) for i in train_indices]
    test_contexts = all_contexts[test_indices[0]]
    
    # Create datasets and dataloaders
    train_dataset = BERTContextDataset(train_contexts, tokenizer)
    test_dataset = BERTContextDataset(test_contexts, tokenizer)
    
    train_batch_size = 32  # Start small to manage memory
    test_batch_size = 32   # Can be larger for inference
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=train_batch_size, shuffle=False)
    
    # Initialize model
    output_dim = fmri_data.shape[1]  # Number of voxels
    
    model_hf_path = model_name
        
    bert_brain_model = BertToBrain(output_dim=output_dim, model_hf_path=model_hf_path).to(DEVICE)
    
    # Ensure gradient checkpointing is enabled
    bert_brain_model.bert.gradient_checkpointing_enable()
    
    # Train with end-to-end gradient flow
    trained_model = train_model_end_to_end(
        bert_brain_model, train_loader, fmri_data, story_window_counts,
        loss_fn, modality, subject, fine_tune_layers,
        batch_size=train_batch_size  # Start with small batch size
    )
    
    # Save the trained weights
    os.makedirs('weights', exist_ok=True)
    save_finetuned_model_weights(
        trained_model,
        f'weights/{model_name}_finetuned_end2end_{fine_tune_layers}_{loss_fn.__name__.split("_")[0]}_{subject}_{modality}.pth'
    )

    # Get the test story name
    test_story_name = ResponseUtils().test_story[0]
    
    # Process the test data using the test loader
    test_features, test_predictions = predict_on_test_data(
        trained_model, 
        test_loader, 
        subject, 
        modality,
        test_story_name
    )
    
    # Load test data for evaluation
    # This is a placeholder - you'll need to adapt this to how your test data is structured
    X_test = [np.zeros((100, 768))]  # Replace with actual test data generation
    
    return trained_model, test_features, test_predictions

# ------------------- PLOT FUNCTION -------------------
def plot_voxel_correlation(voxcorrs, subject_name, xfm_name, vmax, vmin, file_name):
    """
    Plot voxel correlation using cortex Volume and save as image.

    Args:
        voxcorrs (np.ndarray): Voxel correlations to visualize.
        subject_name (str): Subject identifier for cortex Volume.
        xfm_name (str): Transformation name for cortex Volume.
        vmax (float): Maximum value for colormap.
        vmin (float): Minimum value for colormap.
        file_name (str): Name of the file to save the plot.
    """
    plot_args = dict(
    with_curvature=True,
    with_rois=True,
    with_labels=True,
    with_colorbar=True,
    colorbar_location='right',
    with_sulci=False,
    recache=False,
    nanmean=True,
    with_dropout=False, 
    cmap='inferno'
    )
    vol = cortex.Volume(voxcorrs, subject_name, xfmname=xfm_name, vmax=vmax, vmin=vmin, cmap='inferno')
    cortex.quickshow(vol, **plot_args)
    plt.savefig(file_name, format='jpg')
    plt.close()

def save_finetuned_model_weights(model, output_path="weights/fine_tuned_bert"):
    """
    Saves the BERT part of a FineTunedBERT model using save_pretrained
    (excludes the projection layer).
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Save just the BERT component using save_pretrained
    model.bert.save_pretrained(output_path)
    
    print(f"✅ Fine-tuned BERT weights (without projection layer) saved to {output_path}")

# ---- RUN SCRIPT ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "CheXpert NN argparser")
    parser.add_argument("-i","--input_dir", help="Input dir with stimulus words", type = str, required=True)
    parser.add_argument("-seq","--sequence_length", help="Choose context", type = int)
    parser.add_argument("-s","--subject", help="Choose subject", type = str, required=True)
    parser.add_argument("-m","--modality", help="Choose modality (e.g. listening, reading)", type = str, required = True)
    parser.add_argument("-t","--tune", help="Choose tuning", type = str, choices=['last','all'], required = True) #'last or all'
    parser.add_argument("-f","--loss", help="Choose loss function", type = str, choices=['mse','ridge','spatial', 'contrastive','hybrid'], required = True)
    parser.add_argument("-lm", "--model", help="Language model name as on huggingface", type = str, required = True)
    
    args = parser.parse_args()
    print(args.input_dir)
    # Load and label all context windows
    all_contexts = []
    if os.path.isdir(args.input_dir):
        for story_idx, story_file in enumerate(sorted(os.listdir(args.input_dir))):
            # Skip hidden files, system files like .DS_Store, and non-text files
            if story_file.startswith(".") or not story_file.lower().endswith(".txt"):
                continue

            print(story_file)
            file_path = os.path.join(args.input_dir, story_file)
            # Be tolerant to occasional encoding glitches in stimuli files
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                words = np.array(f.read().strip().split("\n"))
                story_contexts = create_20_word_contexts(words, seq_len=args.sequence_length)

                all_contexts.append(story_contexts)
    print(f"Total number of context windows from all stories: {len(all_contexts)}")
    
    # Select loss function dynamically
    if args.loss == "mse":
        loss_fn = mse_loss
    elif args.loss == "ridge":
        loss_fn = ridge_loss
    elif args.loss == "spatial":
        loss_fn = spatial_correlation_loss
    elif args.loss == "contrastive":
        loss_fn = nt_xent_loss
    elif args.loss == "hybrid":
        loss_fn = hybrid_loss
    else:
        raise ValueError(f"Invalid loss function: {args.loss}")

    print(f"\nFine-tuning {args.tune} layers of {args.model}...")
    rresp_en_subj1, rresp_test_subj1_en = load_brain_data(args.subject, args.modality)
    
    trained_model_all, test_features, test_predictions = main_end_to_end(
        all_contexts,
        rresp_en_subj1,
        loss_fn,
        args.modality,
        model_name=args.model,
        subject=args.subject,
        fine_tune_layers=args.tune,
    )
    # Add shuffle status to output file names
    np.save('monolingual_brain_data',np.nan_to_num(test_predictions[:291,:].detach().cpu().numpy()))
    correlation_voxels = compute_correlations(np.nan_to_num(test_predictions[:291,:].detach().cpu().numpy()), np.nan_to_num(rresp_test_subj1_en))
    
    #mean_corr, correlation_voxels = evaluate_model(trained_model_all, np.nan_to_num(test_features.detach().cpu().numpy(),), np.nan_to_num(rresp_test_subj1_en))
    
    os.makedirs('correlation_plots', exist_ok=True)
    np.save(f'correlation_plots/{args.model}_tuning_{args.subject}', correlation_voxels)

# Optional: Add a function to analyze and compare shuffled vs non-shuffled results
def compare_real_vs_shuffled(real_corrs, shuffled_corrs, subject, model_name, output_dir='comparison_plots'):
    """
    Compare and visualize the difference between real and shuffled brain data results.
    
    Args:
        real_corrs (np.ndarray): Correlation values from normal (non-shuffled) training
        shuffled_corrs (np.ndarray): Correlation values from shuffled training
        subject (str): Subject identifier
        model_name (str): Model name for file naming
        output_dir (str): Directory to save comparison plots
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Histogram comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(real_corrs, bins=50, alpha=0.7, color='blue', label='Real')
    plt.hist(shuffled_corrs, bins=50, alpha=0.7, color='red', label='Shuffled')
    plt.xlabel('Correlation Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Voxel Correlations')
    plt.legend()
    
    # 2. Difference map visualization
    diff_corrs = real_corrs - shuffled_corrs
    
    plt.subplot(1, 2, 2)
    plt.hist(diff_corrs, bins=50, color='purple')
    plt.xlabel('Correlation Difference (Real - Shuffled)')
    plt.ylabel('Frequency')
    plt.title('Difference Between Real and Shuffled Correlations')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_{subject}_real_vs_shuffled_histogram.jpg')
    plt.close()
    
    # 3. Scatter plot of real vs shuffled correlations
    plt.figure(figsize=(8, 8))
    plt.scatter(real_corrs, shuffled_corrs, alpha=0.3, s=1)
    
    # Add diagonal line
    max_val = max(np.max(real_corrs), np.max(shuffled_corrs))
    min_val = min(np.min(real_corrs), np.min(shuffled_corrs))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
    
    plt.xlabel('Real Data Correlation')
    plt.ylabel('Shuffled Data Correlation')
    plt.title('Real vs Shuffled Voxel Correlations')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_{subject}_real_vs_shuffled_scatter.jpg')
    plt.close()
    
    # 5. Statistical summary
    print("--- Statistical Comparison Real vs. Shuffled ---")
    print(f"Real data - Mean: {np.mean(real_corrs):.4f}, Median: {np.median(real_corrs):.4f}, Std: {np.std(real_corrs):.4f}")
    print(f"Shuffled data - Mean: {np.mean(shuffled_corrs):.4f}, Median: {np.median(shuffled_corrs):.4f}, Std: {np.std(shuffled_corrs):.4f}")
    print(f"Difference - Mean: {np.mean(diff_corrs):.4f}, Median: {np.median(diff_corrs):.4f}, Std: {np.std(diff_corrs):.4f}")
    
    # Return summary statistics
    return {
        'real_mean': float(np.mean(real_corrs)),
        'real_median': float(np.median(real_corrs)),
        'real_std': float(np.std(real_corrs)),
        'shuffled_mean': float(np.mean(shuffled_corrs)),
        'shuffled_median': float(np.median(shuffled_corrs)),
        'shuffled_std': float(np.std(shuffled_corrs)),
        'diff_mean': float(np.mean(diff_corrs)),
        'diff_median': float(np.median(diff_corrs)),
        'diff_std': float(np.std(diff_corrs))
    }