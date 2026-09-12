"""Explicit R4 paired-rounding intervention; historical evaluation stays frozen.

This thin opt-in entrypoint extends the codec factory in this process only.
It changes neither the base execution source nor any historical result.
Both the new factory and codec source are included in its source freeze.
"""
from . import diag_r4_execution as execution
from .codec_r4_rne import CodecR4RNE

original_factory=execution.new_cache


def new_cache(engine,root,spec):
    if spec['codec']!='R4_OFFSET_RNE_V1':return original_factory(engine,root,spec)
    masks=execution.policy(root,spec,engine.device)
    return engine.cache('RTPA_DIAG',masks,layers=sorted(masks),codec=CodecR4RNE(str(engine.device)))


def main():
    execution.new_cache=new_cache
    execution.SOURCES=(*execution.SOURCES,'codec_r4_rne.py','diag_r4_repair_execution.py')
    execution.main()


if __name__=='__main__':main()
