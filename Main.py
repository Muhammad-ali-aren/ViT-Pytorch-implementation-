
import torch
import torch.nn as nn
import yaml
from dataset import train_loader
from schedular import get_lr_schedular
from Model import ViT
from torch.optim import AdamW
from train import train_and_test
from torch.utils.data import DataLoader
from dataset import train_dataset


with open('config.yaml','r') as file:
    config = yaml.safe_load(file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)

torch.manual_seed(42)

epochs = config['Epochs']
steps_per_epoch = len(train_loader)
total_steps = epochs * steps_per_epoch
warmup_steps = int(0.1 * total_steps)

vit_model = ViT(embed_dim=config['embed_dim'], num_encoders=config['num_encoders'], hidden_dim=config['hidden_dim'],num_head=config['num_heads'],num_classes=config['num_classes'])

train_loader =  DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True,drop_last=True)

optimizer = AdamW(vit_model.parameters(),lr=config['base_lr'],betas=(config['beta1'],config['beta2']), eps=float(config['eps']),weight_decay=config['weight_decay'], )
loss_fn = nn.CrossEntropyLoss()
schedular = get_lr_schedular(total_steps=total_steps,warmup_steps=warmup_steps,optimizer=optimizer,schedular_type=config['scheduler_type'])


if __name__ == "__main__":
    results, train_model = train_and_test(vit_model,train_loader=train_loader,optimizer=optimizer,loss_fn=loss_fn,schedular=schedular,epochs=epochs)

