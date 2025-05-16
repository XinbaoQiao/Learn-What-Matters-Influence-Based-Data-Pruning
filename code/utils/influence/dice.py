import torch

def dice_one_hop_influence(model_j, neighbors, P, node_idx, batch, optimizer, q_j=1.0, q_ks=None):
    """
    Calculate one-hop DICE influence for a node, following the DICE paper formula.

    Args:
        model_j: The current node's model (torch.nn.Module).
        neighbors: List of neighbor node models (torch.nn.Module).
        P: Weight matrix (torch.Tensor, shape=[num_nodes, num_nodes]).
        node_idx: Index of the current node (int).
        batch: Tuple (inputs, targets) for the current batch.
        optimizer: Optimizer for the current node (to get parameter update).
        q_j: Weight for the current node (float, default=1.0).
        q_ks: List of weights for neighbor nodes (default=None, will use 1.0 for each).

    Returns:
        dice_influence: The DICE one-hop influence value (float).
    """
    model_j.eval()
    inputs, targets = batch
    device = next(model_j.parameters()).device
    inputs, targets = inputs.to(device), targets.to(device)

    # 1. Compute loss and gradient for current node
    outputs = model_j(inputs)
    loss = torch.nn.functional.cross_entropy(outputs, targets)
    grads_j = torch.autograd.grad(loss, model_j.parameters(), create_graph=True)
    grads_j = [g.detach() for g in grads_j]

    # 2. Compute parameter update Δ_j (simulate one step SGD)
    lr = optimizer.param_groups[0]['lr']
    delta_j = [-lr * g for g in grads_j]

    # 3. First term: -q_j * grad^T * Δ_j
    first_term = 0.0
    for g, d in zip(grads_j, delta_j):
        first_term += torch.sum(g * d).item()
    first_term = -q_j * first_term

    # 4. Second term: sum over neighbors
    second_term = 0.0
    if q_ks is None:
        q_ks = [1.0 for _ in neighbors]
    for idx, neighbor in enumerate(neighbors):
        neighbor.eval()
        # Neighbor's weight to this node
        W_kj = P[idx, node_idx]
        # Use the same batch for neighbor (can be adjusted as needed)
        outputs_k = neighbor(inputs)
        loss_k = torch.nn.functional.cross_entropy(outputs_k, targets)
        grads_k = torch.autograd.grad(loss_k, neighbor.parameters(), create_graph=True)
        grads_k = [g.detach() for g in grads_k]
        # Use Δ_j from current node
        dot = 0.0
        for gk, dj in zip(grads_k, delta_j):
            dot += torch.sum(gk * dj).item()
        second_term += q_ks[idx] * W_kj * dot

    dice_influence = first_term - second_term
    return dice_influence

def dice_evaluation(
    model_j,
    neighbors,
    P,
    node_idx,
    dataloader,
    optimizer,
    pruning_ratio=0.0,
    q_j=1.0,
    q_ks=None
):
    """
    Evaluate DICE influence for each batch in the dataloader, remove the least influential batches, and return a new dataloader.

    Args:
        model_j: The current node's model.
        neighbors: List of neighbor node models.
        P: Weight matrix.
        node_idx: Index of the current node.
        dataloader: The original dataloader.
        optimizer: Optimizer for the current node.
        pruning_ratio: Ratio of batches to remove based on influence.
        q_j, q_ks: Weight parameters.

    Returns:
        pruned_loader: New dataloader with the remaining data.
    """
    batches = list(dataloader)
    influences = []
    # Compute influence for each batch
    for batch in batches:
        influence = dice_one_hop_influence(model_j, neighbors, P, node_idx, batch, optimizer, q_j, q_ks)
        influences.append(influence)
    # Sort by influence and remove the least influential batches
    sorted_indices = torch.argsort(torch.tensor(influences))
    num_to_remove = int(len(batches) * pruning_ratio)
    keep_indices = sorted_indices[num_to_remove:]
    kept_batches = [batches[i] for i in keep_indices]
    # Merge remaining batches and rebuild dataloader
    # Assume batch is (inputs, targets)
    inputs_cat = torch.cat([b[0] for b in kept_batches])
    targets_cat = torch.cat([b[1] for b in kept_batches])
    pruned_dataset = torch.utils.data.TensorDataset(inputs_cat, targets_cat)
    pruned_loader = torch.utils.data.DataLoader(
        pruned_dataset,
        batch_size=dataloader.batch_size,
        shuffle=getattr(dataloader, 'shuffle', True)
    )
    return pruned_loader
