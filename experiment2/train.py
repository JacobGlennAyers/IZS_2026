import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train(model, train_indices, train_dataset, criterion, optimizer, experiment_parameters):
    model.train()
    train_batch_count = 0
    train_total_loss = 0

    for train_ndx in tqdm(train_indices, desc="Training: "):
        train_dataset.select_clip(train_ndx.item())
        dataloader = DataLoader(train_dataset, batch_size=experiment_parameters["batch_size"], shuffle=False,
            num_workers=4,           # Parallel data loading
            pin_memory=True,         # Faster GPU transfer
            persistent_workers=False, # Keep workers alive between epochs
            prefetch_factor=4        # Prefetch batches
        )
        
        for batch in dataloader:
            x, y = batch
            x, y = x.to(experiment_parameters["device"]), y.to(experiment_parameters["device"])
            x = model.reshape_input(x)
            y = model.reshape_output(y)
            pred_y = model(x)
            #if experiment_parameters["model"] in ["Transformer", "DecoderTransformer"]:
            #    pred_y = model(x, y)  # Pass target for teacher forcing
            #else:
            #    pred_y = model(x)
            loss = criterion(pred_y, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_batch_count += 1
            train_total_loss += loss.item()

    return train_total_loss / train_batch_count
