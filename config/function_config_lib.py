# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from config.collections import AttrDict

# --------------------------------------------------------------------------- #
# optimizer options:
# --------------------------------------------------------------------------- #

adamax_opt = AttrDict()
adamax_opt.lr = 0.01
adamax_opt.weight_decay = 0
adamax_opt.eps = 0.00000001

Adam_par = AttrDict()
Adam_par.lr = 2.5e-4



OPTIMIZER = {
    'Adam': Adam_par,
    'Adamax': adamax_opt
}



# --------------------------------------------------------------------------- #
# loss options:
# --------------------------------------------------------------------------- #

LOSS = {
    'logit': AttrDict(),
    'softmax': AttrDict()
}


# SUMMARY of all model parameters
MODEL_TYPE_PAR_DICT = {
    'Adamax': adamax_opt,
    'Adam': Adam_par,
}


class ModelParPair(AttrDict):

    IMMUTABLE = '__immutable__'

    def __init__(self, model_type):
        super(ModelParPair, self).__init__()

        self.method = model_type
        if self.method not in MODEL_TYPE_PAR_DICT:
            exit("unkown model type %s, please check \
                 config/function_config_lib.py for allowed options")
        self.par = MODEL_TYPE_PAR_DICT[self.method]

    def update_type(self, updated_pair_type):
        if updated_pair_type != self.method:
            self.method = updated_pair_type
            self.par = MODEL_TYPE_PAR_DICT[self.method]

    def is_immutable(self):
        return self.__dict__[ModelParPair.IMMUTABLE]
