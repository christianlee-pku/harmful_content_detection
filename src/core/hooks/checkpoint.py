from src.core.registry import HOOKS


@HOOKS.register_module()
class CheckpointHook:
    """
    Save checkpoints periodically.
    """
    def __init__(self, interval: int = 1):
        self.interval = interval

    def after_train_epoch(self, runner):
        if runner.epoch % self.interval == 0:
            runner.save_checkpoint(f"epoch_{runner.epoch}.pth")
            
            # Optionally save 'latest.pth'
            runner.save_checkpoint("latest.pth")
