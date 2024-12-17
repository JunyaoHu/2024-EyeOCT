# 2024-Eye-OCT

2024年亚太眼科学会大数据竞赛 使用人工智能自動從眼底照生成OCT （25/40）baseline 代码

FVD: 1083.8656
SSIM: 0.1367
PSNR: 13.3087

# 基本思路

修改controlnet，把眼底照片作为一种条件（类似文本），给引入到diffusion中，让模型学习眼底照片和OCT之间的关系。

# 数据集读取

数据集原位置在

```
/home/pod/shared-nvme
    -----/data
            -----EyeOCT
                    -----train
                        CF
                        OCT
                        train.csv
                    -----val
                        CF
                        val.csv
```

设计了自定义的train和valid读取接口, 见 

```
/home/pod/project/code/EyeOCT/dataset_AE.py / dataset_DM.py (两个是一样的其实)
```

注意这里要对原数据集进行预处理, 做裁剪（有一部分图是包含了除了OCT以外的内容的）

```
if idx <= 728:
    target_i = target_i[:496,-768:]
```

# 训练

## ae模型
首先训练一个ae模型，因为原版的diffusion的AE是三通道的，我们这个要一次性生成六张OCT，可以把他当作一个六通道的图片，所以我修改了ae的config

```
/home/pod/project/code/EyeOCT/models/autoencoder_kl_32x32x4.yaml
```

在这里设置了 `in_channels: 6`

其他的都是按照原配置进行了

这一步实现了oct的压缩和重建

训练的方法是

```
/home/pod/project/code/EyeOCT/train_AE.py
```

训练的checkpoint在

```
/home/pod/shared-nvme/tensorboard/logs/OCT_AE
```

## dm模型
再训练一个dm模型，同理我们这个要一次性生成六张OCT，可以把他当作一个六通道的图片，修改dm的config

```
/home/pod/project/code/EyeOCT/models/cldm_v15.yaml
```

在这里设置了 

```
AutoencoderKL
in_channels: 6
out_ch: 6
```

然后还要修改cldm的交叉注意力，我们的眼底照片是作为条件输入的

对于原本的`BasicTransformerBlock`, 修改了`OCTTransformerBlock`

其中加了一个眼底照片的注意力，删去了原本的文本clip encoder的注意力

```
def _forward(self, x, cond_global=None):
    # print("cond_global", cond_global.shape)
    # QKV-OCT self-attention
    x = self.attn1(self.norm1(x), context=None) + x
    # Q-OCT KV-CF_global cross-attention
    x = self.attn2(self.norm2(x), context=cond_global) + x
    # FeedForward
    x = self.ff(self.norm3(x)) + x
    return x
```

如果要细看修改了哪些代码，请全局搜索`cond_global`即可，`cond_global`代表眼底图像作为条件的输入embedding，这个涉及的代码段还挺多的

```
/home/pod/project/code/EyeOCT/cldm/cldm.py
/home/pod/project/code/EyeOCT/ldm/modules/attention.py
```

因为torchlightning的代码真的很难绷，我后面会转diffuser了，大家都说这个很好用

其他的都基本是按照原配置进行了

```
/home/pod/project/code/EyeOCT/train_DM.py
```

训练的checkpoint在

```
/home/pod/shared-nvme/tensorboard/logs/OCT_DM
```

# 推理

使用下面的代码实现推理

```
/home/pod/project/code/EyeOCT/inference_DM.py
```

推理的图片会保存在

```
/home/pod/project/code/EyeOCT/log_valid/yyddHHMM-ddimxx
```

时间戳是自动的，ddim步数要手动设置

```
def shared_step_test(self, batch, batch_idx):
    ddim_steps = 10
    images = self.log_valid_images(batch, ddim_steps=ddim_steps)
    
    foldername = f"./log_valid/{time}-ddim{ddim_steps}/samples"
```
