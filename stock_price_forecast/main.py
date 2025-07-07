from omegaconf import DictConfig, OmegaConf

from dataloader import *
from model import *
from trainer import *
import hydra
import torch
from torch.utils.tensorboard import SummaryWriter
from measurement import *
@hydra.main(version_base=None, config_path='./conf', config_name='config.yaml')
def main(cfg):
    OmegaConf.to_yaml(cfg)
    train_loader, val_loader, x_test, y_test = get_dataset()
    for x, y in val_loader:
        print(x.shape, y.shape)
        break
    print(x_test.shape, y_test.shape)
    input_size = x_test.shape[2]

    writer = SummaryWriter()
    model=GRU(cfg.model.num_classes, input_size, cfg.model.hidden_size, cfg.model.num_layers, seq_length=14)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.params.lr)
    #print_cpu_memory_usage()
    #print_gpu_usage()
    train(model, train_loader, val_loader, criterion,optimizer, cfg.params.epoch, writer)
    #print_cpu_memory_usage()
    #print_gpu_usage()
    writer.close()
    test(model, x_test, y_test, criterion)
if __name__=='__main__':
    main()