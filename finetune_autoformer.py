import torch
from transformers import AutoformerForPrediction, AutoformerConfig
import evaluate

import pandas as pd 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class PowerDemandDataset(Dataset):
    
    """A power demand dataframe wrapper with some neccesary transformations."""
    
    def __init__(self, dataframe, transform=None):    
        
        self.transform = transform
        self.dataframe = self.transform(dataframe)

        

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

       return self.dataframe.iloc[index]

class ToTimeSeries(object):
    """Transfroms a panda series of scalar values to multiple non overlapping time series.
    
    Each resulting time series contains approximately one months data. 
    """
    
    def __init__(self):
        pass

    def __call__(self, dataframe):
        
        grouped_indices = len(dataframe) // 12

        dataframe["month_demand"] = dataframe.index // grouped_indices
        
        dataframe = dataframe.groupby("month_demand")["demand"].apply(list)

        past_values = dataframe.apply(lambda x: x[:int(0.75 * len(x))])
        future_values = dataframe.apply(lambda x: x[:int(0.75 * len(x)):])

        dataframe_final = pd.concat([past_values, future_values], names=["past_values", "fututre_values"], axis=1)
        return torch.Tensor(dataframe_final)

dataframe = pd.read_csv("power_data.txt", names=["demand"])
train_dataframe, test_dataframe = train_test_split(dataframe, test_size=0.2)

train_dataset = PowerDemandDataset(train_dataframe, transform=ToTimeSeries())
test_dataset = PowerDemandDataset(test_dataframe, transform=ToTimeSeries())


train_dataloader = DataLoader(train_dataset, batch_size=2000, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2000, shuffle=True)

configuration = AutoformerConfig(prediction_length=730)

model = AutoformerForPrediction(
    config=configuration).from_pretrained("kashif/autoformer-electricity-hourly")


for batch in train_dataloader:

    past_values = batch["past_values"]
    future_values = batch["future_values"]
    model(past_values=past_values, future_values=future_values)

#----------------------------------------------------------------------------------------------------------------------------

mase_metric = evaluate.load("evaluate-metric/mase")
