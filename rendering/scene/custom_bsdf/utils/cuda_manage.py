import torch
import gc

def print_cuda_memory_info(stage=""):
    """
    Print current CUDA memory usage information
    
    Args:
        stage: Current execution stage description
    """
    if torch.cuda.is_available():
        # Get current CUDA memory information
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # Convert to GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
        
        print(f"[{stage}] CUDA Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")
    else:
        print(f"[{stage}] CUDA is not available")

def clear_all_cache():
    """
    Clear all possible caches
    """
    # Clear PyTorch CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()
    
    print("All caches cleared")