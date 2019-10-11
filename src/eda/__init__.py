import config as cfg
from .helper import Helper
from .plot import ClassificationPlotter

helper = Helper()
plotter = None
if cfg.task_type == 0:
	plotter = ClassificationPlotter()
else:
	pass
