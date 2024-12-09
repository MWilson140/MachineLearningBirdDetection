import os
import torch
import torch.nn.functional as F

class VisualTrainer:

  def __init__(self,tdata, vdata, device):
    self.tdata = tdata
    self.vdata = vdata
    self.device = device
    self.lowestValidLoss = float('inf')
    self.lowestEpoch = 0

  def fit(self, model,epochs,savedir, tb):
      model.to(self.device)
      self.optimizer = model.configure_optimizers()
      self.model = model
      os.makedirs(savedir, exist_ok=True)
      self.lowestEpoch = 0
      self.lowestValidLoss = 10

      for epoch in range(epochs):
          self.fit_epoch(epochs)
          self.validate()
          
          # Logging metrics to TensorBoard
          tb.add_scalar("Training Loss", self.tloss, epoch)
          tb.add_scalar("Valid Loss", self.vloss, epoch)
          tb.add_scalar("Training Accuracy", self.accuracy, epoch)
          tb.add_scalar("Validation Accuracy", self.training_accuracy, epoch)
          if self.vloss < self.lowestValidLoss:
            self.lowestValidLoss = self.vloss
            print(f"new lowest {self.lowestValidLoss}")
            lowest_path = os.path.join(savedir, "lowest.pth")
            torch.save(self.model.state_dict(), lowest_path)
          if self.vloss > (self.lowestValidLoss + 0.15):
            print(f"quit {self.lowestValidLoss} {self.vloss}")
            print(f"Validation loss exceeded threshold at epoch {epoch + 1}. Stopping training.")
            break
          if (epoch + 1) % 2 == 0:
            save_path = os.path.join(savedir, f"model_epoch_{epoch + 1}.pth")
            if self.vloss < self.lowestValidLoss:
              self.lowestValidLoss = self.vloss
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1} to {save_path}")

      print("Training process has finished")

  def fit_epoch(self, epoch):
    correct = 0
    current_loss = 0.0
    self.tloss = 0.0
    self.model.train()
    total_batches = 0;
    self.accuracy = 0.0
    total_predictions = 0
    for i, tdata in enumerate(self.tdata):
      inputs, target, file_id = tdata
      inputs, target = inputs.to(self.device), target.to(self.device)
      self.optimizer.zero_grad()
      outputs = self.model(inputs)
      loss = self.model.loss(outputs, target)
      probability = F.sigmoid(outputs)
      probability = (probability > 0.5).int().squeeze()
      correct += (probability == target).sum()
      total_predictions += target.size(0)
      loss.backward()
      self.optimizer.step()
      current_loss += loss.item()
      self.tloss += loss.item()
      total_batches +=1
    self.tloss = self.tloss / total_batches
    self.accuracy = correct / total_predictions
    print(f"total training loss {self.tloss:.3f}")
    print(f"Accuracy {self.accuracy}")



  def validate(self):
    correct = 0
    predictions = 0
    proability = 0
    self.vloss = 0
    current_loss = 0.0
    self.training_accuracy = 0.0
    total_samples = 0;
    total_predictions = 0
    self.model.eval()
    total_batches = 0;
    with torch.no_grad():
      for i, vdata in enumerate(self.vdata):
        inputs, target, file_id= vdata
        inputs, target = inputs.to(self.device), target.to(self.device)
        outputs = self.model(inputs)
        loss = self.model.loss(outputs, target)
        self.vloss += loss.item()
        probability = F.sigmoid(outputs)
        probability = (probability > 0.5).int().squeeze()
        correct += (probability == target).sum()
        total_predictions += target.size(0)
        total_batches +=1
    self.training_accuracy = correct/total_predictions
    self.vloss = self.vloss / total_batches
    print(f"Validation Loss: {self.vloss:.3f}")
    print(f"Validation accuracy: {self.training_accuracy}")
    print()