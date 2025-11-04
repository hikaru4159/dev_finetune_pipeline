import onnx
import sys
from onnx import numpy_helper

path = sys.argv[1] if len(sys.argv) > 1 else 'base/supercombo.onnx'
model = onnx.load(path)
print('Model:', path)
print('Inputs:')
for inp in model.graph.input:
    name = inp.name
    shape = []
    try:
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            elif dim.dim_value:
                shape.append(dim.dim_value)
            else:
                shape.append('?')
    except Exception:
        shape = ['unknown']
    elem_type = inp.type.tensor_type.elem_type
    print(' -', name, 'dtype=', elem_type, 'shape=', shape)

print('\nInitializers:')
for init in model.graph.initializer:
    arr = numpy_helper.to_array(init)
    print(' -', init.name, 'shape=', arr.shape, 'dtype=', arr.dtype)

print('\nValue infos:')
for vi in model.graph.value_info:
    name = vi.name
    shape = []
    try:
        for dim in vi.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            elif dim.dim_value:
                shape.append(dim.dim_value)
            else:
                shape.append('?')
    except Exception:
        shape = ['unknown']
    print(' -', name, 'shape=', shape)
