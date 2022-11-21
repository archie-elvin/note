---
title: Mxnet 转 ONNX
description: Mxnet 转 ONNX
categories:
 - Python
tags:
 - deeplearn
---

### 以Retinaface_Cov模型为例

### 环境

| package         | version |
| --------------- | ------- |
| mxnet           | 1.9.1   |
| onnx            | 1.11.0  |
| onnxruntime     | 1.11.1  |
| onnx-simplifier | 0.3.10  |

### 修改模型Softmax算子

```json
// mnet_cov2-symbol.json
{
    "op": "SoftmaxActivation", 
    "name": "face_rpn_type_prob_stride8", 
    "attrs": {"mode": "channel"}, 
    "inputs": [[585, 0, 0]]
}, 
// 替换为
{
    "op": "softmax", 
    "name": "face_rpn_type_prob_stride8", 
    "attrs": {"axis": "1"}, 
    "inputs": [[585, 0, 0]]
},
```

```bash
:%s/SoftmaxActivation/softmax/g
:%s/"mode": "channel"/"axis": "1"/g
```



### 添加 `UpSampling`、`Crop`算子，替换`softmax`算子

```python
# {PYTHON_ENV}/site-packages/mxnet/onnx/mx2onnx/_op_translations.py

@mx_op.register("UpSampling")
def convert_upsample(node, **kwargs):
    """Map MXNet's UpSampling operator attributes to onnx's Upsample operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    sample_type = attrs.get('sample_type', 'nearest')
    sample_type = 'linear' if sample_type == 'bilinear' else sample_type
    scale = convert_string_to_list(attrs.get('scale'))
    scaleh = scalew = float(scale[0])
    if len(scale) > 1:
        scaleh = float(scale[0])
        scalew = float(scale[1])
    scale = np.array([1.0, 1.0, scaleh, scalew], dtype=np.float32)
    roi = np.array([], dtype=np.float32)
    node_roi=create_helper_tensor_node(roi, name+'roi', kwargs)
    node_sca=create_helper_tensor_node(scale, name+'scale', kwargs)
    node = onnx.helper.make_node(
        'Resize',
        inputs=[input_nodes[0], name+'roi', name+'scale'],
        outputs=[name],
        coordinate_transformation_mode='asymmetric',
        mode=sample_type,
        nearest_mode='floor',
        name=name
    )
    return [node_roi, node_sca, node]

@mx_op.register("Crop")
def convert_crop(node, **kwargs):
    """Map MXNet's crop operator attributes to onnx's Crop operator
    and return the created node.
    """
    name, inputs, attrs = get_inputs(node, kwargs)
    start=np.array([0, 0, 0, 0], dtype=np.int) #index是int类型
    start_node=create_helper_tensor_node(start, name+'__starts', kwargs)
    shape_node = create_helper_shape_node(inputs[1], inputs[1]+'__shape')
    crop_node = onnx.helper.make_node(
        "Slice",
        inputs=[inputs[0], name+'__starts', inputs[1]+'__shape'], #data、start、end
        outputs=[name],
        name=name
    )
    logging.warning(
        "Using an experimental ONNX operator: Crop. " \
        "Its definition can change.")
    return [start_node, shape_node, crop_node]

@mx_op.register("softmax")
def convert_softmax(node, **kwargs):
    """Map MXNet's softmax operator attributes to onnx's Softmax operator
    and return the created node.
    """
    name, input_nodes, attrs = get_inputs(node, kwargs)
    axis = int(attrs.get("axis", -1))
    c_softmax_node = []
    axis=-1
    transpose_node1 = onnx.helper.make_node(
        "Transpose",
        inputs=input_nodes,
        perm=(0,2,3,1), #NCHW--NHWC--(NHW,C)
        name=name+'_tr1',
        outputs=[name+'_tr1']
    )
    softmax_node = onnx.helper.make_node(
        "Softmax",
        inputs=[name+'_tr1'],
        axis=axis,
        name=name+'',
        outputs=[name+'']
    )
    transpose_node2 = onnx.helper.make_node(
        "Transpose",
        inputs=[name+''],
        perm=(0,3,1,2), #NHWC--NCHW
        name=name+'_tr2',
        outputs=[name+'_tr2']
    )
    c_softmax_node.append(transpose_node1)
    c_softmax_node.append(softmax_node)
    c_softmax_node.append(transpose_node2)
    return c_softmax_node

def create_helper_tensor_node(input_vals, output_name, kwargs):
    """create extra tensor node from numpy values"""
    data_type = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input_vals.dtype]
    tensor_node = onnx.helper.make_tensor_value_info(
        name=output_name,
        elem_type=data_type,
        shape=input_vals.shape
    )
    kwargs["initializer"].append(
        onnx.helper.make_tensor(
            name=output_name,
            data_type=data_type,
            dims=input_vals.shape,
            vals=input_vals.flatten().tolist(),
            raw=False,
        )
    )
    return tensor_node

def create_helper_shape_node(input_node, node_name):
    """create extra transpose node for dot operator"""
    trans_node = onnx.helper.make_node(
        'Shape',
        inputs=[input_node],
        outputs=[node_name],
        name=node_name
    )
    return trans_node
```

### 转换脚本

```python
import onnx
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet

if __name__ == '__main__':
    sym = './mnet_cov2-symbol.json'
    params = './mnet_cov2-0000.params'
    input_shape = [(1, 3, 1280, 1280)]
    onnx_file = './mnet_cov2.onnx'
    convert_path = onnx_mxnet.export_model(sym, params, input_shape, np.float32, onnx_file, dynamic=True, dynamic_input_shapes=[(1, 3, None, None)])
    print(convert_path)
    onnx.checker.check_model(onnx_file)
    print('onnx is checked!')
```

### 输入警告消除脚本

```python
import onnx

def remove_initializer_from_input():
    model = onnx.load('./mnet_cov2.onnx')
    if model.ir_version < 4:
        print("Model with ir_version below 4 requires to include initilizer in graph input")
        return
    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input
    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])
    onnx.save(model, './mnet_cov2_fixed.onnx')

if __name__ == "__main__":
    remove_initializer_from_input()
```

### 优化

```bash
onnxsim mnet_cov2_fixed.onnx mnet_cov2_sim.onnx --dynamic-input-shape
```

### 参考

https://zhuanlan.zhihu.com/p/166267806
