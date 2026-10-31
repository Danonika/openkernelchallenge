"""Fusion B: guarded native-loss replay + versioned MLP saves + FP32 rotary.

Original component code remains byte-identical in nested packages. No source
kernel is relabeled as a new kernel; no component shares/savings are added.
"""
import json


def apply():
    from . import memory_component, head_component, rotary_component
    memory = memory_component.apply()
    head = head_component.apply()
    rotary_component.apply()
    result = {'candidate': 'soup-fusion-b-20261030', 'memory': memory,
              'head_installed': head, 'rotary_installed': True,
              'score': None, 'admission': 'UNKNOWN',
              'note': 'Installation metadata is not runtime coverage or target validation'}
    print('SOUP_FUSION_B_INSTALL ' + json.dumps(result, sort_keys=True), flush=True)
    return result
