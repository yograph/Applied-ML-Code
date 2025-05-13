from convnet_one import Creating_Convnet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class full_training_of_model():
    """
    Inheret from creating_convnet??
    """
    def __init__(self,dataset):
        """
        dataset is properly preprocessed into a tensor array with dataloader
        
        """
        self.dataset = dataset
        self.models_to_store = {}

    
    def starting_models(self, number_of_models):
        """
        
        
        """
        for i in range(number_of_models):
            self.models_to_store["model_" + str(i)] = Creating_Convnet.creating_single_model()
        
        ### Create sanity checks

        assert len(self.models_to_store) != 0
        print(self.models_to_store)

    def train_models(self):
        """
        Here we train all the models required
        """
        loader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        for index, model in enumerate(self.models_to_store):
            current_model = self.models_to_store[index]
            current_model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-4)

            # one epoch
            for images, labels in loader:
                optimizer.zero_grad()
                outputs = model(images)          # forward
                loss = criterion(outputs, labels)
                loss.backward()                  # backward
                optimizer.step()                 # update

            print("Done one training epoch, loss =", loss.item())


        

    

        