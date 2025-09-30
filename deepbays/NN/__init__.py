# from .training import train, regLoss
from .training import train, regLoss, test
# from .multitask_training import multitaskMSE, multitaskTrain, multitaskLoss
# from .training import test
# from .training import LangevinOpt
# from .training_maybe_faster import LangevinOpt
from .training import LangevinOpt
from .networks import FCNet
from .networks import ConvNet