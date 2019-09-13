import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import datetime
from btp_dataset import BtpDataset
from utils import time_series_to_plot
from tensorboardX import SummaryWriter
from models.recurrent_models import LSTMGenerator, LSTMDiscriminator
from models.convolutional_models import CausalConvGenerator, CausalConvDiscriminator

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="btp", help='dataset to use (only btp for now)')
parser.add_argument('--dataset_path', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--nz', type=int, default=100, help='dimensionality of the latent vector z')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='checkpoints', help='folder to save checkpoints')
parser.add_argument('--imf', default='images', help='folder to save images')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--logdir', default='log', help='logdir for tensorboard')
parser.add_argument('--run_tag', default='', help='tags for the current run')
parser.add_argument('--checkpoint_every', default=5, help='number of epochs after which saving checkpoints') 
parser.add_argument('--tensorboard_image_every', default=5, help='interval for displaying images on tensorboard') 
parser.add_argument('--dis_type', default='cnn', choices=['cnn','lstm'], help='architecture to be used for discriminator to use')
parser.add_argument('--gen_type', default='lstm', choices=['cnn','lstm'], help='architecture to be used for generator to use')
opt = parser.parse_args()

#Create writer for tensorboard
date = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
run_name = f"{opt.run_tag}_{date}" if opt.run_tag != '' else date
log_dir_name = os.path.join(opt.logdir, run_name)
writer = SummaryWriter(log_dir_name)
writer.add_text('Options', str(opt), 0)
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass
try:
    os.makedirs(opt.imf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("You have a cuda device, so you might want to run with --cuda as option")

if opt.dataset == "btp":
    dataset = BtpDataset(opt.dataset_path)
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
nz = int(opt.nz)
#Retrieve the sequence length as first dimension of a sequence in the dataset
seq_len = dataset[0].size(0)
#An additional input is needed for the delta
in_dim = opt.nz + 1

if opt.dis_type == "lstm": 
    netD = LSTMDiscriminator(in_dim=2, hidden_dim=256).to(device)
if opt.dis_type == "cnn":
    netD = CausalConvDiscriminator(input_size=2, n_layers=8, n_channel=10, kernel_size=8, dropout=0).to(device)
if opt.gen_type == "lstm":
    netG = LSTMGenerator(in_dim=in_dim, out_dim=1, hidden_dim=256).to(device)
if opt.gen_type == "cnn":
    netG = CausalConvGenerator(noise_size=in_dim, output_size=1, n_layers=8, n_channel=10, kernel_size=8, dropout=0.2).to(device)
    
assert netG
assert netD

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))    
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))

print("|Discriminator Architecture|\n", netD)
print("|Generator Architecture|\n", netG)

criterion = nn.BCELoss().to(device)
delta_criterion = nn.MSELoss().to(device)

#Generate fixed noise to be used for visualization
fixed_noise = torch.randn(opt.batchSize, seq_len, nz, device=device)

#Sample both deltas and noise for visualization
deltas = dataset.sample_deltas(opt.batchSize).unsqueeze(2).repeat(1, seq_len, 1)
fixed_noise = torch.cat((fixed_noise, deltas), dim=2)

real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr)

for epoch in range(opt.epochs):
    for i, data in enumerate(dataloader, 0):
        niter = epoch * len(dataloader) + i
        
        #Save just first batch of real data for displaying
        if i == 0:
            real_display = data.cpu()
      
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        #Train with real data
        netD.zero_grad()
        real = data.to(device)
        batch_size, seq_len = real.size(0), real.size(1)
        label = torch.full((batch_size, seq_len, 1), real_label, device=device)
        
        deltas = (real[:, -1] - real[:, 0]).unsqueeze(2).repeat(1, seq_len, 1)
        real = torch.cat((real, deltas), dim=2)
        output = netD(real)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()
        
        #Train with fake data
        noise = torch.randn(batch_size, seq_len, nz, device=device)
        #Sample a delta for each batch and concatenate to the noise for each timestep
        deltas = dataset.sample_deltas(batch_size).unsqueeze(2).repeat(1, seq_len, 1)
        noise = torch.cat((noise, deltas), dim=2)

        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(torch.cat((fake.detach(), deltas), dim=2))
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()
        
        #Visualize discriminator gradients
        for name, param in netD.named_parameters():
            writer.add_histogram("DiscriminatorGradients/{}".format(name), param.grad, niter)

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label) 
        output = netD(torch.cat((fake, deltas), dim=2))
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        
        optimizerG.step()
        
        #Visualize generator gradients
        for name, param in netG.named_parameters():
            writer.add_histogram("GeneratorGradients/{}".format(name), param.grad, niter)
        
        ###########################
        # (3) Supervised update of G network: minimize mse of input deltas and actual deltas of generated sequences
        ###########################

        #Report metrics
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' 
              % (epoch, opt.epochs, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2), end='')
        print()
        writer.add_scalar('DiscriminatorLoss', errD.item(), niter)
        writer.add_scalar('GeneratorLoss', errG.item(), niter)
        writer.add_scalar('D of X', D_x, niter) 
        writer.add_scalar('D of G of z', D_G_z1, niter)
        
    ##### End of the epoch #####
    real_plot = time_series_to_plot(dataset.denormalize(real_display))
    if (epoch % opt.tensorboard_image_every == 0) or (epoch == (opt.epochs - 1)):
        writer.add_image("Real", real_plot, epoch)
    
    fake = netG(fixed_noise)
    fake_plot = time_series_to_plot(dataset.denormalize(fake))
    torchvision.utils.save_image(fake_plot, os.path.join(opt.imf, opt.run_tag+'_epoch'+str(epoch)+'.jpg'))
    if (epoch % opt.tensorboard_image_every == 0) or (epoch == (opt.epochs - 1)):
        writer.add_image("Fake", fake_plot, epoch)
                             
    # Checkpoint
    if (epoch % opt.checkpoint_every == 0) or (epoch == (opt.epochs - 1)):
        torch.save(netG, '%s/%s_netG_epoch_%d.pth' % (opt.outf, opt.run_tag, epoch))
        torch.save(netD, '%s/%s_netD_epoch_%d.pth' % (opt.outf, opt.run_tag, epoch))
