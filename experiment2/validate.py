import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def validate(model, validation_indices, train_dataset, criterion, experiment_parameters):
    model.eval()
    val_batch_count = 0
    val_total_loss = 0

    with torch.no_grad():
        for val_ndx in tqdm(validation_indices, desc="Validation: "):
            train_dataset.select_clip(val_ndx.item())
            dataloader = DataLoader(train_dataset, batch_size=experiment_parameters["batch_size"], shuffle=False)

            for batch in dataloader:
                x, y = batch
                x, y = x.to(experiment_parameters["device"]), y.to(experiment_parameters["device"])
                x = model.reshape_input(x)
                y = model.reshape_output(y)
                pred_y = model(x)
                #if experiment_parameters["model"] in ["Transformer", "DecoderTransformer"]:
                #    pred_y = model(x, y=None)  # Inference mode
                #else:
                #    pred_y = model(x)
                loss = criterion(pred_y, y)

                val_batch_count += 1
                val_total_loss += loss.item()

    return val_total_loss / val_batch_count
