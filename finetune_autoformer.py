import torch
import json
import argparse

from transformers import AutoformerForPrediction, AutoformerConfig

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, classification_report

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class PowerDemandDataset(Dataset):
    
    """A power demand dataframe wrapper with some neccesary transformations."""
    
    def __init__(self, dataframe, transform=None):    
        
        self.transform = transform
        self.dataframe = self.transform(dataframe)


    def find_number_peaks(self, index, daily_distance=104):
        
        """Find the number of local maxima in one week.
    
        If there are 5 it is a normal timeseries if not it is anomalous.

        daily_distance: The number of datapoints in one day
        """

        future_values = self.dataframe["future_values"].iloc[index]

        number_peaks = len(find_peaks(future_values, distance=daily_distance)[0])

        return 1 if number_peaks >= 5 else 0



    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index): 
        
        past_values = torch.Tensor(self.dataframe.iloc[index]["past_values"])
        future_values = torch.Tensor(self.dataframe.iloc[index]["future_values"])
        
        past_time_features = torch.repeat_interleave(torch.Tensor([index]), repeats=2190).unsqueeze(-1)

        future_time_features = torch.repeat_interleave(torch.Tensor([index]), repeats=730).unsqueeze(-1)

        # Unsqueeze because only one time feature (month) is present
        # Else also concat all past time features together
        # The number of repeats is equal to the sequence length 
        # This is equal to the total number of values for three weeks (see autoformer docs)
        # NOTE: At this point lags are the first 7 time points of a month

        total_past_records = len(past_values)
        past_observed_mask = torch.ones(total_past_records)

        label = self.find_number_peaks(index=index)
        # The label is set by hand after plotting the time series
        # The plot is produced by the function plot_monthly_data

        # When the label is zero the time series is normal
        # When the label is one the time series is anomalous

        return {"past_values": past_values,
                "future_values": future_values,
                "past_time_features": past_time_features,
                "past_observed_mask": past_observed_mask,
                "future_time_features": future_time_features,
                "label": label}

class ToTimeSeries(object):
    """Transfroms a panda series of scalar values to multiple non overlapping time series.
    
    Each resulting time series contains approximately one months data. 
    """
    
    def __init__(self, window_size:int, is_plot=False):

        self.is_plot = is_plot
        self.window_size = window_size

    def __call__(self, dataframe, grouped_indices=2920):

        # Grouped indices is the result of the number of records in the dataframe 
        # divided by the number of months in a year.

        rolling_windows = dataframe["demand"].rolling(window=grouped_indices)#, step=self.window_size)

        windows = [{"yearly_demand":list(window)}
        for window in rolling_windows if len(window) == grouped_indices]

        dataframe_final = pd.DataFrame(windows)
        breakpoint()
        # dataframe["month"] = dataframe.index // grouped_indices

        # dataframe_final = dataframe.groupby("month")["demand"].apply(list).to_frame()
        # dataframe_final = dataframe_final[:-1] if len(dataframe_final) == 10 else dataframe_final[1:] if not self.is_plot else dataframe_final
        
        # The last record in the dataframe (for month 9) has less past_values than the other records
        # So the last record can either be dropped or upsampled (batch requirement: Same shape per item)
        # TODO: Try upsampling as well

        dataframe_final["past_values"] = dataframe_final["yearly_demand"].apply(lambda x: x[:-730])
        dataframe_final["future_values"] = dataframe_final["yearly_demand"].apply(lambda x: x[-730:])

        dataframe_len = len(dataframe_final)

        # dataframe_final = dataframe_final.sample(0.75 * dataframe_len, random_state=42)
        # breakpoint()
        return dataframe_final

def plot_monthly_data(dataset: PowerDemandDataset):
    """ Subplot all 12 weeks of the dataset"""

    fig, ax = plt.subplots(4, 3, sharey=True)

    dataframe = dataset.dataframe 

    for i in range(4):
        for j in range(3):
            monthly_demand = dataframe.iloc[i]["future_values"]
            ax[i, j].plot(monthly_demand)
            ax[i, j].set_title(f"Energy demand month {3 * i + j + 1}")


    plt.savefig("weekly_data.png")



def train_one_epoch(train_dataloader: DataLoader,
        model:AutoformerForPrediction, lr:int,
        epoch:int, writer:SummaryWriter):

    torch.cuda.empty_cache()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    running_loss = 0
    last_loss = 0

    
    num_batches = len(train_dataloader)
    model.train()
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        past_values = batch['past_values'].to(device)
        future_values = batch['future_values'].to(device)
        past_time_features = batch["past_time_features"].to(device)
        past_observed_mask = batch["past_observed_mask"].to(device)
        future_time_features = batch["future_time_features"].to(device)

        outputs = model(past_values=past_values, future_values=future_values,
        past_time_features=past_time_features, past_observed_mask=past_observed_mask,
        future_time_features=future_time_features)

        loss = outputs.loss

        writer.add_scalar("Loss/Train",loss.item(), i)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if (i + 1)% (num_batches) == 0:
            last_loss = running_loss / num_batches  # loss per batch
            print('epoch {}  batch {} loss: {}'.format(epoch + 1, i + 1, last_loss))
            running_loss = 0.

    
def process_dataset(batch_size: int, window_size: int):
        
    dataframe = pd.read_csv("power_data.txt", names=["demand"])

    train_dataframe, test_dataframe = train_test_split(dataframe, test_size=0.3, shuffle=False)


    dataset = PowerDemandDataset(dataframe, transform=ToTimeSeries(window_size=window_size))
    train_dataset = PowerDemandDataset(train_dataframe, transform=ToTimeSeries(window_size=window_size))
    test_dataset = PowerDemandDataset(test_dataframe, transform=ToTimeSeries(window_size=window_size))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

def evaluate_model(model: AutoformerForPrediction, test_loader: DataLoader, writer: SummaryWriter):
    torch.cuda.empty_cache()
    model.eval()
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            past_values = batch['past_values'].to(device)
            future_values = batch['future_values'].to(device)
            past_time_features = batch["past_time_features"].to(device)
            past_observed_mask = batch["past_observed_mask"].to(device)
            future_time_features = batch["future_time_features"].to(device)
            labels = batch["label"].to(device)

            model.to(device)
            outputs = model.generate(past_values=past_values, 
                            past_time_features=past_time_features,
                            past_observed_mask=past_observed_mask,
                            future_time_features=future_time_features)
    
            time_series_mean = outputs.sequences.mean(dim=1)
            time_series_std = outputs.sequences.std(dim=1)

            predictions = (torch.mean(torch.abs(future_values - time_series_mean) / (2 *  time_series_std), axis = 1) < 1).type(torch.LongTensor)
            if 1 in labels:
                print(f"batch:{i}, labels:{labels}, score:{precision_score(labels.cpu(), predictions.cpu())}")
                print("Classification Report/Test", classification_report(labels.cpu(), predictions.cpu()))

            # writer.add_text("Classification Report/Test", classification_report(labels.cpu(), predictions.cpu()))

if __name__ == "__main__":

    hyperparams_file = open("hyperparameters.json")
    hyperparams = json.load(hyperparams_file)

    lr = hyperparams["learning_rate"]
    epochs = hyperparams["epochs"]
    batch_size = hyperparams["batch_size"]
    window_size = hyperparams["window_size"]
    available_gpu = hyperparams["available_gpu"]

    device = f"cuda:{available_gpu}" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", dest="lr", help="Learning Rate", type=float)
    parser.add_argument("--epochs", dest="epochs", help="Number of training epochs", type=int)
    parser.add_argument("--batch_size", dest="batch_size", help="Number of records in one batch", type=int)

    # args = parser.parse_args()

    # lr = args.lr
    # epochs = args.epochs
    # batch_size = args.batch_size


    log_dir =  "experiments/" + "/".join([f"{param}={value}" for param, value in hyperparams.items()])
    writer = SummaryWriter(log_dir=log_dir)
    train_dataloader, test_dataloader = process_dataset(batch_size=batch_size, window_size=window_size)

    for batch in train_dataloader: 
        if 1 in batch["label"]: print(batch["label"])

   
  

#    configuration = AutoformerConfig(prediction_length=730, context_length=2183, num_time_features=1)
#    model = AutoformerForPrediction.from_pretrained("kashif/autoformer-electricity-hourly", config=configuration, ignore_mismatched_sizes=True).to(device) 
#
#    for param in model.parameters():
#        param.requires_grad_(False)
#    
#    model.parameter_projection.requires_grad_()
#
#   # for i, param in enumerate(model.parameters()):
#   #     if i >= 10:
#   #         param.requires_grad_()
#
#   # This works without having to iterate
#   # NOTE: Not all layers shown from print(model) are callable
#   # TODO: Investigate
#
#    for epoch in range(epochs):
#         train_one_epoch(
#                 train_dataloader=train_dataloader, writer=writer,
#                 model=model, lr=lr, epoch=epoch)
#   # 
#
#   # torch.save(model.state_dict(), "./finetuned_autoformer.pt")
#    evaluate_model(model, test_dataloader, writer)
#
#    hyperparams_file.close()
#
