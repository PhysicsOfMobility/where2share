import pickle
from boosting_model import *

from where2share.models import ResidualBoostingModel, CustomModel

model = pickle.load(open("feature_restricted_model.pkl", "rb"))
model.__class__ = ResidualBoostingModel
model.theory_model.__class__ = CustomModel

pickle.dump(model, open("feature_restricted_model_w2s.pkl", "wb"))