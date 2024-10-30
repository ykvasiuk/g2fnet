import custom_radius_cuda
import torch

def create_batch_pointers(batch):
    # Get unique batch IDs and their first occurrences (start of each batch)
    unique_batches, counts = torch.unique_consecutive(batch, return_counts=True)
    ptr = torch.zeros(unique_batches.size(0) + 1, device=batch.device, dtype=torch.long)
    ptr[1:] = torch.cumsum(counts, dim=0)
    return ptr


def periodic_radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=256, L=1.0):
    """
    Wrapper function to handle batch assignments and call the CUDA radius function.

    Parameters:
    - x: Tensor of shape [N, D] for N points in D dimensions
    - y: Tensor of shape [M, D] for M points in D dimensions
    - r: Radius within which to search for neighbors
    - batch_x: Batch tensor for x of shape [N], optional
    - batch_y: Batch tensor for y of shape [M], optional
    - max_num_neighbors: Maximum number of neighbors to return
    - L: Length of the periodic boundary (for periodic boundary conditions)

    Returns:
    - Tensor with neighbor assignments.
    """
    # Determine the batch size
    if batch_x is not None and batch_y is not None:
        batch_size = max(batch_x.max().item(), batch_y.max().item()) + 1
    else:
        batch_size = 1

    # If batch_size > 1, convert batch_x and batch_y into batch pointers
    if batch_size > 1:
        assert batch_x is not None, "batch_x must be provided for multiple batches"
        assert batch_y is not None, "batch_y must be provided for multiple batches"
        
        # Create batch pointers from batch_x and batch_y
        ptr_x = create_batch_pointers(batch_x)
        ptr_y = create_batch_pointers(batch_y)
    else:
        # Treat all points as a single batch if batch_x or batch_y is not provided
        ptr_x = None
        ptr_y = None

    # Now call the CUDA radius function (assuming it's wrapped in PyTorch)
    result = custom_radius_cuda.radius_periodic(x, y, ptr_x=ptr_x, ptr_y=ptr_y, r=r, max_num_neighbors=max_num_neighbors, L=L)
    
    return result