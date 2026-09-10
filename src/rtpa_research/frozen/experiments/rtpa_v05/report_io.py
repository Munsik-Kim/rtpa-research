"""JSON scalar-type adapter only; frozen metric arithmetic and files stay unchanged."""
from .common import np, save

def builtin_scalars(value):
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,dict):
        return {key:builtin_scalars(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [builtin_scalars(item) for item in value]
    return value

def aggregate():
    from . import metrics
    original_save=metrics.save
    try:
        metrics.save=lambda path,value: save(path,builtin_scalars(value))
        return metrics.aggregate()
    finally:
        metrics.save=original_save
