# onnx导出

1、导出模板分支：

```python
net = models.LightTrackM_Subnet('back_04502514044521042540+cls_211000022+reg_100000111_ops_32', 16)
net = load_pretrain(net, '../snapshot/LightTrackM/LightTrackM.pth')
dummy_input = (
    torch.randn(1, 3, 128, 128),
    )
torch.onnx.export(
    net.features,
    dummy_input,
    "lighttrack-z.onnx",
    verbose=True,
    opset_version=11,
    input_names=["z"],
    output_names=["zf"],
    )
print('----------finished exporting onnx-----------')
print('----------start simplifying onnx-----------')
model_sim, flag = onnxsim.simplify('lighttrack-z.onnx')
if flag:
    onnx.save(model_sim, 'lighttrack-z.onnx')
    print("---------simplify onnx successfully---------")
else:
    print("---------simplify onnx failed-----------")
```



2、修改源码中的 `lib/models/super_model_DP.py` 中 `class Super_model_DP_retrain` 的 `forward` 函数：

```python
def forward(self, zf, search):
    """backbone_index: which layer's feature to use"""
    # zf = self.features(template)
    xf = self.features(search)
    # Batch Normalization before Corr
    zf, xf = self.neck(zf, xf)
    # Point-wise Correlation
    feat_dict = self.feature_fusor(zf, xf)
    # supernet head
    oup = self.head(feat_dict)
    cls_score = nn.functional.sigmoid(oup['cls'])
    bbox_pred = oup['reg']
    return cls_score, bbox_pred
```

然后导出 搜索分支+neck+head：

```python
net = models.LightTrackM_Subnet('back_04502514044521042540+cls_211000022+reg_100000111_ops_32', 16)
net = load_pretrain(net, '../snapshot/LightTrackM/LightTrackM.pth')
dummy_input = (
        torch.randn(1, 96, 8, 8),
        torch.randn(1, 3, 256, 256),
    )
torch.onnx.export(
    net,
    dummy_input,
    "lighttrack-x-head.onnx",
    verbose=True,
    opset_version=11,
    input_names=["zf", "x"],
    output_names=["cls", "reg"],
)
print('----------finished exporting onnx-----------')
print('----------start simplifying onnx-----------')
model_sim, flag = onnxsim.simplify("lighttrack-x-head.onnx")
if flag:
    onnx.save(model_sim, "lighttrack-x-head.onnx")
    print("---------simplify onnx successfully---------")
else:
    print("---------simplify onnx failed-----------")
```





