import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Pytorch lightning imports
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from data.ab_datasets import AbDataModule
from models.flow_module import FlowModule
from utils import experiments_utils as eu
import wandb


log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')


class Experiment:

    def __init__(self, *, cfg: DictConfig):

        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment
        self._datamodule = AbDataModule(data_cfg=self._data_cfg)
        self._datamodule.setup()

        self._train_device_ids = self._exp_cfg.num_devices 
        log.info(f"Training with {self._train_device_ids} devices")
        self._module = FlowModule(self._cfg)


    def train(self):
        callbacks = []
        logger = WandbLogger(**self._exp_cfg.wandb)
        
        # Checkpoint directory.
        ckpt_dir = self._exp_cfg.checkpointer.dirpath
        log.info(f"Checkpoints saved to {ckpt_dir}")
        
        # Model checkpoints
        callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))
        
        # Save config only for main process.
        local_rank = os.environ.get('LOCAL_RANK', 0)
        if local_rank == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            cfg_path = os.path.join(ckpt_dir, 'config.yaml')
            with open(cfg_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f.name)
            cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
            flat_cfg = dict(eu.flatten_dict(cfg_dict))
            if isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
                logger.experiment.config.update(flat_cfg)

        trainer = Trainer(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=self._train_device_ids,
        )
        trainer.fit(
            model=self._module,
            datamodule=self._datamodule,
            ckpt_path=self._exp_cfg.warm_start
        )


@hydra.main(version_base=None, config_path="configs", config_name="base.yaml")
def main(cfg: DictConfig):

    if cfg.experiment.warm_start is not None and cfg.experiment.warm_start_cfg_override:
        # Loads warm start config.
        warm_start_cfg_path = os.path.join(os.path.dirname(cfg.experiment.warm_start), 'config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        # Warm start config may not have latest fields in the base config.
        # Add these fields to the warm start config.
        OmegaConf.set_struct(cfg.model, False)
        OmegaConf.set_struct(warm_start_cfg.model, False)
        cfg.model = OmegaConf.merge(cfg.model, warm_start_cfg.model)
        OmegaConf.set_struct(cfg.model, True)
        log.info(f'Loaded warm start config from {warm_start_cfg_path}')

    exp = Experiment(cfg=cfg)
    exp.train()

if __name__ == "__main__":
    main()
