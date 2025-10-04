"""
Created on Wed Mar  6 20:04:20 2024

@author: zcf
"""
import torch
from mmyolo.registry import LOOPS
from mmengine.runner import EpochBasedTrainLoop
import copy

@LOOPS.register_module()
class TrainValEpochBasedTrainLoop(EpochBasedTrainLoop):
    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')
        trainMetric_dataloader = copy.deepcopy(self.runner.cfg.val_dataloader)
        trainMetric_dataloader.dataset.ann_file = self.runner.cfg.train_dataloader.dataset.ann_file
        trainMetric_evaluator =  copy.deepcopy(self.runner.cfg.val_evaluator)
        trainMetric_evaluator.ann_file = self.runner.cfg.train_dataloader.dataset.data_root+self.runner.cfg.train_dataloader.dataset.ann_file
        self.trainMetric_loop = LOOPS.build(dict(type='ValLoop'), default_args=dict(
                                                                    runner=self.runner,
                                                                    dataloader=trainMetric_dataloader,
                                                                    evaluator=trainMetric_evaluator))
        while self._epoch < self._max_epochs and not self.stop_training:
            self.run_epoch()

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and self._epoch % self.val_interval == 0):
                self.trainMetric_loop.run() #----------------zcf
                self.runner.val_loop.run()
        self.runner.call_hook('after_train')
        return self.runner.model

