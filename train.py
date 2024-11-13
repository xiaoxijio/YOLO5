import argparse
import logging
import os
import random
import shutil
import time
from pathlib import Path

import math
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test
import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import get_latest_run, check_file, increment_dir, check_img_size, labels_to_class_weights, \
    labels_to_image_weights, compute_loss, plot_images, fitness, strip_optimizer, plot_results
from utils.google_utils import attempt_download
from utils.torch_utils import select_device, init_torch_seeds, intersect_dicts, ModelEMA

logger = logging.getLogger(__name__)


def train(hyp, opt, device, tb_writer=None):
    logger.info(f'Hyperparameters {hyp}')
    log_dir = Path(tb_writer.log_dir) if tb_writer else Path(opt.logdir) / 'evolve'  # 日志位置 runs/exp
    wdir = log_dir / 'weights'  # 权重路径 runs/exp/weights
    os.makedirs(wdir, exist_ok=True)
    last = wdir / 'last.pt'  # 上一次训练保存的模型路径 runs/exp/weights/last.pt
    best = wdir / 'best.pt'  # 目前最优模型路径 runs/exp/weights/best.pt
    results_file = str(log_dir / 'results.txt')  # 训练过程中各种指标路径 runs/exp/results.txt
    epochs, batch_size, total_batch_size, weights, rank = \
        opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    with open(log_dir / 'hyp.yaml', 'w') as f:  # 保存当前参数
        yaml.dump(hyp, f, sort_keys=False)
    with open(log_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # 数据文件
    train_path = data_dict['train']  # 数据路径 MaskDataSet/train/images
    test_path = data_dict['val']  # MaskDataSet/vaild/images
    nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # 类别数量, 名称
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    pretrained = weights.endswith('.pt')
    if pretrained:  # True
        attempt_download(weights)  # 加载模型参数
        ckpt = torch.load(weights, map_location=device)  # 加载检查点
        if hyp.get('anchors'):
            ckpt['model'].yaml['anchors'] = round(hyp['anchors'])  # force autoanchor
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)  # create
        exclude = ['anchor'] if opt.cfg or hyp.get('anchors') else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # 只加载键值名称和参数形状都匹配的部分
        model.load_state_dict(state_dict, strict=False)  # load
    else:
        model = Model(opt.cfg, ch=3, nc=nc).to(device)

    # 不做迁移学习 所以没写

    nbs = 64  # nominal batch size 累计多少次更新一次模型，咱们的话就是64/16=4次，相当于扩大batch
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs

    pg0, pg1, pg2 = [], [], []  # 优化组：其它参数，权重衰减，偏置
    for k, v in model.named_parameters():
        v.requires_grad = True
        if '.bias' in k:
            pg2.append(v)  # 偏置
        elif '.weight' in k and '.bn' not in k:
            pg1.append(v)  # 权重衰减
        else:
            pg0.append(v)  # all else

    optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # 余弦退火

    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']  # [0.0, 0.0, 0.1, 0.9]*[精确度, 召回率, mAP@0.5, mAP@0.5:0.95]

        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])

        start_epoch = ckpt['epoch'] + 1
        if epochs < start_epoch:  # 就是你设置的epoch为100 但是现在模型已经训练了150 那就再训练100
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    gs = int(max(model.stride))  # 总的下采样比例
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # 检查数据的大小能不能整除这个比例

    # 没用多机多卡 单机多卡 不写dp、ddp

    ema = ModelEMA(model) if rank in [-1, 0] else None  # 滑动平均能让参数更新的更平滑一点不至于波动太大

    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect,
                                            rank=rank, world_size=opt.world_size, workers=opt.workers)
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class 判断类别数是否正常
    nb = len(dataloader)  # batch数量
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    ema.updates = start_epoch * nb // accumulate  # set EMA updates
    testloader = create_dataloader(test_path, imgsz_test, total_batch_size, gs, opt,
                                   hyp=hyp, augment=False, cache=opt.cache_images and not opt.notest, rect=True,
                                   rank=-1, world_size=opt.world_size, workers=opt.workers)[0]  # testloader

    hyp['cls'] *= nc / 80.
    model.nc = nc  # 将类的数量附加到模型
    model.hyp = hyp  # 将超参数附加到模型
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # 根据标签设置各类别数据初始权重
    model.names = names

    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1e3)  # 热身持续多少个epoch
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1
    scaler = amp.GradScaler(enabled=cuda)  # 混合精度训练

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # 按权重随机选数据

        mloss = torch.zeros(4, device=device)
        pbar = enumerate(dataloader)  # 创建进度条
        logger.info(('\n' + '%12s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size'))
        print(('\n' + '%12s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size'))
        # 'Epoch', 'GPU占用', '边界框损失', '置信度损失', '分类损失', '总损失', '目标数量', '图像尺寸'
        pbar = tqdm(pbar, total=nb)

        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch ---------------------------------------------
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # 归一化

            if ni <= nw:  # 热身环节 通过插值逐步调节学习率和动量，使得模型的训练在热身阶段更加平滑、稳定
                xi = [0, nw]  # 插值范围
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):  # j == 2 偏置项的学习率
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Forward
            with amp.autocast(enabled=cuda):  # 混合精度
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(device), model)

            scaler.scale(loss).backward()

            if ni % accumulate == 0:  # 相当于Backward多次才更新一次参数
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # 打印信息
            mloss = (mloss * i + loss_items) / (i + 1)  # 平均loss
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ('%12s' * 2 + '%12.4g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)
            if ni < 3:
                f = str(log_dir / ('train_batch%g.jpg' % ni))  # filename
                result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                if tb_writer and result is not None:
                    tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)

        # end batch ------------------------------------------------------------------------------------------------

        lr = [x['lr'] for x in optimizer.param_groups]  # 学习率衰减
        scheduler.step()  # 余弦退火

        if ema:
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
        final_epoch = epoch + 1 == epochs
        if not opt.notest or final_epoch:
            results, maps, times = test.test(opt.data,
                                             batch_size=total_batch_size,
                                             imgsz=imgsz_test,
                                             model=ema.ema,
                                             single_cls=opt.single_cls,
                                             dataloader=testloader,
                                             save_dir=log_dir,
                                             plots=epoch == 0 or final_epoch)

        with open(results_file, 'a') as f:
            f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        if tb_writer:
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                tb_writer.add_scalar(tag, x, epoch)

        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        if fi > best_fitness:
            best_fitness = fi

        save = (not opt.nosave) or (final_epoch and not opt.evolve)
        if save:
            with open(results_file, 'r') as f:  # create checkpoint
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': f.read(),
                        'model': ema.ema,
                        'optimizer': None if final_epoch else optimizer.state_dict()}

            torch.save(ckpt, last)  # Save last, best and delete
            if best_fitness == fi:
                torch.save(ckpt, best)
            del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    n = opt.name if opt.name.isnumeric() else ''
    fresults, flast, fbest = log_dir / f'results{n}.txt', wdir / f'last{n}.pt', wdir / f'best{n}.pt'
    for f1, f2 in zip([wdir / 'last.pt', wdir / 'best.pt', results_file], [flast, fbest, fresults]):
        if os.path.exists(f1):
            os.rename(f1, f2)  # rename
            if str(f2).endswith('.pt'):  # is *.pt
                strip_optimizer(f2)  # strip optimizer
                os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket else None  # upload
    # Finish
    if not opt.evolve:
        plot_results(save_dir=log_dir)  # save as results.png
    logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    dist.destroy_process_group() if rank not in [-1, 0] else None
    torch.cuda.empty_cache()
    return results


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='模型的网络配置')
    parser.add_argument('--data', type=str, default='data/MaskDataSet/data.yaml', help='数据路径')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='超参数路径')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='矩形训练')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='接着之前的训练')
    parser.add_argument('--nosave', action='store_true', help='只保存最终的检查点')
    parser.add_argument('--notest', action='store_true', help='只测试最终的epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='是否调整候选框')
    parser.add_argument('--evolve', action='store_true', help='超参数更新')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='缓存图片')
    parser.add_argument('--image-weights', action='store_true', help='使用加权图像选择进行训练')
    parser.add_argument('--name', default='', help='renames experiment folder exp{N} to exp{N}_{name} if supplied')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='是否多尺度训练')
    parser.add_argument('--single-cls', action='store_true', help='是否单类别')
    parser.add_argument('--adam', action='store_true', help='优化器选择')
    parser.add_argument('--sync-bn', action='store_true', help='跨GPU的BN')
    parser.add_argument('--local_rank', type=int, default=-1, help='GPU ID')
    parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
    parser.add_argument('--workers', type=int, default=0, help='数据加载器工作程序的最大数目, windows用户勿扰')
    opt = parser.parse_args()

    opt.total_batch_size = opt.batch_size
    opt.world_size = 1
    opt.global_rank = -1  # 非分布式

    if opt.resume:  # 是否继续训练
        # 传入模型的路径或者最后一次跑的模型（在runs中有last.pt）
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        log_dir = Path(ckpt).parent.parent  # runs/exp0
        assert os.path.isfile(ckpt), 'ERROR: --恢复检查点不存在'
        with open(log_dir / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True

    else:  # 加载之前配置好的参数
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        log_dir = increment_dir(Path(opt.logdir) / 'exp', opt.name)  # runs/exp1

    device = select_device(opt.device, batch_size=opt.batch_size)
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    if not opt.evolve:
        logger.info(f'启动Tensorboard "tensorboard --logdir {opt.logdir}", view at http://localhost:6006/')
        tb_writer = SummaryWriter(log_dir=log_dir)  # runs/exp0
        train(hyp, opt, device, tb_writer)
