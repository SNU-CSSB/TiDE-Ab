import os
import torch
import hydra
import pandas as pd
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from models.flow_module import FlowModule
from data.ab_datasets import AbDataModule
from utils import experiments_utils as eu

log = eu.get_pylogger(__name__)

class Predictor:
    def __init__(self, cfg: DictConfig, weights_path: str):
        self._cfg = cfg
        self._module = FlowModule(self._cfg)
        
        log.info(f"Loading weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu')
        
        first_key = list(state_dict.keys())[0]
        if first_key.startswith('module.') and not hasattr(self._module.model, 'module'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        missing, unexpected = self._module.model.load_state_dict(state_dict, strict=False)
        log.info(f"Missing keys (expected): {missing}")
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

@hydra.main(version_base=None, config_path="configs", config_name="base.yaml")
def main(cfg: DictConfig):
    weights_path = "./weights.pt" 
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    predictor = Predictor(cfg, weights_path)
    _ = predictor.predict()
    
    log.info("Inference finished.")

if __name__ == "__main__":
    main()