import time
import mindspore
import mindspore.ops as ops
import mindspore.nn as nn
import random

from utils import *
from options import TrainOptions
from models import NAFNet
from losses import LossL1, LossFreqReco, LossSSIM
from datasets_test import PairedImgDataset

print('---------------------------------------- step 1/5 : parameters preparing... ----------------------------------------')
opt = TrainOptions().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
set_random_seed(opt.seed)
models_dir, log_dir, train_images_dir, val_images_dir = prepare_dir(opt.results_dir, opt.experiment, delete=(not opt.resume))
writer = SummaryWriter(log_dir=log_dir)

print('---------------------------------------- step 2/5 : data loading... ------------------------------------------------')
# print('training data loading...')
# train_dataset = PairedImgDataset(data_source=opt.data_source, mode='train', crop=[opt.cropx, opt.cropy], random_resize=None)
# train_dataloader = DataLoader(train_dataset, batch_size=opt.train_bs_per_gpu*gpu_num, shuffle=True, num_workers=opt.num_workers, pin_memory=False)
# print('successfuly loading training pairs. =====> qty:{} bs:{}'.format(len(train_dataset),opt.train_bs_per_gpu*gpu_num))

print('validating data loading...')
val_dataset = PairedImgDataset(data_source=opt.data_source, mode='val')
# val_dataloader = DataLoader(val_dataset, batch_size=opt.val_bs, shuffle=False, num_workers=opt.num_workers, pin_memory=False)
print('successfully loading validating pairs. =====> qty:{} bs:{}'.format(len(val_dataset),opt.val_bs))

print('---------------------------------------- step 3/5 : model defining... ----------------------------------------------')
model = NAFNet(width=48, num_blks=16)
print_para_num(model)

if opt.pretrained is not None:
    mindspore.load_param_into_net(mindspore.load_checkpoint(opt.pretrained))
    print('successfully loading pretrained model.')
    
print('---------------------------------------- step 4/5 : requisites defining... -----------------------------------------')
criterion_l1 = LossL1().cuda()

if opt.resume:
    state = mindspore.load_checkpoint(opt.pretrained)
    optimizer = state['optimizer']
    scheduler = state['scheduler']
else:
    optimizer = nn.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = nn.piecewise_constant_lr(optimizer, opt.scheduler, 0.3)

print('---------------------------------------- step 5/5 : training... ----------------------------------------------------')
def main():
    
    optimal = [0., 0.]
    start_epoch = 1
    if opt.resume:
#         state = mindspore.load_checkpoint(models_dir + '/latest.pth')
        mindspore.load_param_into_net(state['model'])
        start_epoch = state['epoch'] + 1
        optimal = state['optimal']
        
        print('Resume from epoch %d' % (start_epoch), optimal)
    
#     for epoch in range(start_epoch, opt.n_epochs + 1):
#         train(epoch, optimal)
        
#         if (epoch) % opt.val_gap == 0:
    print(opt.exp_name)
    val(start_epoch, optimal)
        
    writer.close()
    
def train(epoch, optimal):
    model.train()
    
    max_iter = len(train_dataloader)
        
    iter_ssim_meter = AverageMeter()
    iter_timer = Timer()
    
    for i, (imgs_l, imgs_r, gts_l, gts_r) in enumerate(train_dataloader):
        [imgs_l, imgs_r, gts_l, gts_r] = [x.cuda() for x in [imgs_l, imgs_r, gts_l, gts_r]]
        cur_batch = imgs_l.shape[0]
        
        optimizer.zero_grad()
        input = mindspore.ops.Concat(1)([imgs_l, imgs_r])
        preds_l, preds_r = model(input)

        loss = criterion_l1(preds_l, gts_l) + criterion_l1(preds_r, gts_r)
        
        loss.backward()
        optimizer.step()
        
        iter_ssim_meter.update(loss.item()*cur_batch, cur_batch)
        

        if (i+1) % opt.print_gap == 0:
            print('Training: Epoch[{:0>4}/{:0>4}] Iteration[{:0>4}/{:0>4}] Best: {:.4f}/{:.4f} loss: {:.4f} Time: {:.4f} LR: {:.8f}'.format(epoch, 
            opt.n_epochs, i + 1, max_iter, optimal[0], optimal[1], iter_ssim_meter.average(), iter_timer.timeit(), scheduler.get_last_lr()[0]))
            writer.add_scalar('Loss_cont', iter_ssim_meter.average(auto_reset=True), i+1 + (epoch - 1) * max_iter)
            
            
    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
    mindspore.save_checkpoint({'model': model.state_dict(), 'epoch': epoch, 'optimal': optimal, 'optimizer': optimizer, 'scheduler': scheduler}, models_dir + '/latest.pth')
    scheduler.step()
    
def val(epoch, optimal):
    model.eval()
    
    print(''); print('Validating...', end=' ')
    
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    timer = Timer()
    
    for i, (imgs_l, imgs_r, gts_l, gts_r) in enumerate(val_dataloader):
        [imgs_l, imgs_r, gts_l, gts_r] = [x.cuda() for x in [imgs_l, imgs_r, gts_l, gts_r]]
        h, w = gts_l.size(2), gts_l.size(3)
        [imgs_l, imgs_r] = [check_padding(x) for x in [imgs_l, imgs_r]]
        input = mindspore.ops.Concat(1)([imgs_l, imgs_r])

        with torch.no_grad():
            preds_l, preds_r = model(input)
        # [preds_l, preds_r] = [torch.clamp(x, 0, 1) for x in [preds_l, preds_r]]
        [preds_l, preds_r] = [x[:, :, :h, :w] for x in [preds_l, preds_r]]

        psnr_value, ssim_value = get_metrics(preds_l, gts_l, psnr_only=False)

        psnr_meter.update(psnr_value, imgs_l.shape[0])
        ssim_meter.update(ssim_value, imgs_l.shape[0])
        
        # psnr_meter.update(get_metrics(preds_clip, gts), imgs.shape[0])
        
        # if i == 0:
        #     if epoch == opt.val_gap:
        #         save_image(imgs, val_images_dir + '/epoch_{:0>4}_iter_{:0>4}_img.png'.format(epoch, i+1), nrow=opt.val_bs, normalize=True, scale_each=True)
        #         save_image(gts, val_images_dir + '/epoch_{:0>4}_iter_{:0>4}_gt.png'.format(epoch, i+1), nrow=opt.val_bs, normalize=True, scale_each=True)
        #     save_image(preds_clip, val_images_dir + '/epoch_{:0>4}_iter_{:0>4}_restored.png'.format(epoch, i+1), nrow=opt.val_bs, normalize=True, scale_each=True)
    
    if optimal[0] < psnr_meter.average():
        optimal[0] = psnr_meter.average()
        mindspore.save_checkpoint(model.state_dict(), models_dir + '/optimal_psnr.pth')
        # mindspore.save_checkpoint(model.state_dict(), models_dir + '/optimal_{:.2f}_epoch_{:0>4}.pth'.format(optimal[0], epoch))
    if optimal[1] < ssim_meter.average():
        optimal[1] = ssim_meter.average()
        mindspore.save_checkpoint(model.state_dict(), models_dir + '/optimal_ssim.pth')
        
    writer.add_scalar('psnr', psnr_meter.average(), epoch)
    writer.add_scalar('ssim', ssim_meter.average(), epoch)

    print('Epoch[{:0>4}/{:0>4}] PSNR/SSIM: {:.4f}/{:.4f} Best: {:.4f}/{:.4f} Time: {:.4f}'.format(epoch, opt.n_epochs, psnr_meter.average(),
     ssim_meter.average(), optimal[0], optimal[1],timer.timeit())); print('')
    
    # mindspore.save_checkpoint(model.state_dict(), models_dir + '/epoch_{:0>4}.pth'.format(epoch))
    
if __name__ == '__main__':
    main()
    