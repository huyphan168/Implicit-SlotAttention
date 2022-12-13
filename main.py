import os
import os.path as osp
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from src.slot_attention import build_model
from src.data import build_dataset

def parse_arguments():
    parser = argparse.ArgumentParser("Slot-Attention-Iterative-Refinement")
    parser.add_argument("--cfg", type=str, default="configs/baseline.yaml")
    parser.add_argument("--output-path", type=str, default="results/")
    args = parser.parse_args()
    return args

def main() -> None:
    args = parse_arguments()
    cfg = OmegaConf.load(args.cfg)
    cfg.output_path = args.output_path
    if not osp.exists(cfg.output_path):
        os.makedirs(cfg.output_path)

    # set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build dataset
    train_dataset = build_dataset(cfg.training, "train")
    val_dataset = build_dataset(cfg.testing, "val")
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.testing.batch_size, shuffle=False, num_workers=cfg.testing.num_workers)

    # build model
    model = build_model(cfg.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    # train
    for epoch in range(cfg.training.epochs):
        model.train()
        for idx, data in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            loss = model(data)
            loss.backward()
            optimizer.step()
            if idx % cfg.training.log_interval == 0:
                print(f"Epoch: {epoch}, Iter: {idx}, Loss: {loss}")
            if idx % cfg.training.save_interval == 0:
                torch.save(model.state_dict(), osp.join(args.output_path, f"model_{epoch}_{idx}.pth"))

        # validation
        model.eval()
        for data in tqdm(val_loader):
            loss = model(data)
            print(f"Validation Loss: {loss}")



if __name__ == "__main__":
    main()