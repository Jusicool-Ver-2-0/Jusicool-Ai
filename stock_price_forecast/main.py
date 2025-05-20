from omegaconf import DictConfig, OmegaConf

from dataloader import *
from model import *
from trainer import *
import hydra
import torch
from torch.utils.tensorboard import SummaryWriter
@hydra.main(version_base=None, config_path='./conf', config_name='config.yaml')
def main(cfg):
    OmegaConf.to_yaml(cfg)
    train_loader, val_loader, x_test, y_test = get_dataset()
    input_size = x_test.shape[2]
    writer = SummaryWriter()
    model=GRU(cfg.model.num_classes, input_size, cfg.model.hidden_size, cfg.model.num_layers, seq_length=3)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.params.lr)
    train(model, train_loader, val_loader, criterion,optimizer, cfg.params.epoch, writer)
    writer.close()
    test(model, x_test, y_test, criterion)
if __name__=='__main__':
    main()