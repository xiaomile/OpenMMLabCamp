_base_ = ['../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32.py', '../_base_/default_runtime.py']
model = dict(
    head=dict( 
        num_classes=5,
        topk = (1,)
    ))
data = dict(
    samples_per_gpu = 32,
    workers_per_gpu = 2,
    train = dict(
        data_prefix = 'data/flower/train',
        ann_file = 'data/flower/train.txt',
        classes = 'data/flower/classes.txt'
    ),
    val = dict(
        data_prefix = 'data/flower/val',
        ann_file = 'data/flower/val.txt',
        classes = 'data/flower/classes.txt'
    )
)
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    step=[1])
runner = dict(type='EpochBasedRunner', max_epochs=100)
# 预训练模型 
load_from = '/HOME/scz0bbt/run/mmclassification/mmclassification/checkpoints/resnet18_batch256_imagenet_20200708-34ab8f90.pth'
