import os
import torch
import argparse
import hydra
import logging
from hydra import compose, initialize
import pandas as pd
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from models.flow_module import FlowModule
from data.ab_datasets import AbDataModule
from utils import experiments_utils as eu

log = eu.get_pylogger(__name__)

def load_state_dict(weights_path, device='cpu'):
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    return state_dict

class Predictor:
    def __init__(self, cfg: DictConfig, weights_path: str, inference_dir: str):
        self._cfg = cfg
        self._module = FlowModule(self._cfg)
        self._module._inference_dir = inference_dir
        
        log.info(f"Loading weights from {weights_path}")
        state_dict = load_state_dict(weights_path)

        first_key = list(state_dict.keys())[0]
        if first_key.startswith('module.') and not hasattr(self._module.model, 'module'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        clean_state_dict = {}
        for k, v in state_dict.items():
            name = k
            while name.startswith('model.') or name.startswith('module.'):
                if name.startswith('model.'):
                    name = name.replace('model.', '', 1)
                if name.startswith('module.'):
                    name = name.replace('module.', '', 1)
            clean_state_dict[name] = v
        
        missing, unexpected = self._module.model.load_state_dict(clean_state_dict, strict=False)
        log.info(f"Missing keys : {missing}")
        log.info(f"Unexpected keys : {unexpected}")
        self._module.model.float()

        self._datamodule = AbDataModule(data_cfg=cfg.data)
        self._datamodule.setup(stage='test')

        self._trainer = Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            logger=False,
        )

    def predict(self):
        log.info("Starting Inference...")
        results = self._trainer.predict(
            model=self._module, 
            dataloaders=self._datamodule.test_dataloader()
        )
        return results

def main():

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("weights_path", type=str, help="Path to the model checkpoint")
    parser.add_argument("inference_dir", type=str, help="Path to save samples")
    args = parser.parse_args()

    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="base")

    weights_path = args.weights_path
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    inference_dir = args.inference_dir
    os.makedirs(inference_dir, exist_ok=True)

    predictor = Predictor(cfg, weights_path, inference_dir)
    _ = predictor.predict()
    
    log.info("Inference finished.")

if __name__ == "__main__":
    main()