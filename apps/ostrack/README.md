# onnx导出



1、修改 `lib/models/ostrack/ostrack.py` 文件中的 `class OSTrack` 的 `forward` 函数：

```python
# return out
# 修改为：
return out['score_map'], out['size_map'], out['offset_map']
```

2、在 `tracking` 文件夹上新建 `export_onnx.py` 文件如下：

```python
# 注意修改权重文件路径
import os
import argparse
import importlib
import sys
sys.path.append(os.getcwd())
from lib.train.base_functions import *
from lib.models.ostrack import build_ostrack
import onnx
import onnxsim

def parse_args():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('--script', type=str, default='ostrack')
    parser.add_argument('--config', type=str, default='vitb_256_mae_ce_ep300')
    parser.add_argument('--output_path', type=str, default='./ostrack.onnx')
    parser.add_argument('--prj_dir', type=str, default='/home/lsf/Object_Tracking/SOT/my_OSTrack')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config_module = importlib.import_module("lib.config.%s.config" % args.script)
    cfg = config_module.cfg
    cfg_file = os.path.join(args.prj_dir, 'experiments/%s/%s.yaml' % (args.script, args.config))
    config_module.update_config_from_file(cfg_file)

    print("New configuration is shown below.")
    for key in cfg.keys():
        print("%s configuration:" % key, cfg[key])
        print('\n')

    # Create network
    net = build_ostrack(cfg, training=False)
    net.cpu()
    checkpoint = os.path.join(args.prj_dir, args.checkpoint)
    net.load_state_dict(torch.load(checkpoint, map_location='cpu')['net'], strict=True)
    net.eval()

    dummy_input = (
        torch.randn(1, 3, 128, 128),
        torch.randn(1, 3, 256, 256),
    )

    torch.onnx.export(
        net,
        dummy_input,
        args.output_path,
        verbose=False,
        opset_version=15,
        input_names=["z", "x"],
        output_names=["score_map", "size_map", "offset_map"]
    )
    print('----------finished exporting onnx-----------')
    print('----------start simplifying onnx-----------')
    model_sim, flag = onnxsim.simplify(args.output_path)
    if flag:
        onnx.save(model_sim, args.output_path)
        print("---------simplify onnx successfully---------")
    else:
        print("---------simplify onnx failed-----------")

if __name__ == "__main__":
    main()
```

3、运行

```python
python tracking/export_onnx.py
```

