from model import Generator
from model import Discriminator
from dataset import data_dict
from dataset import loader_dict
import pickle
import random
import torch


device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
label_dict = None
label_name = ['Neutral','Happiness','Sadness','Surprise','Fear','Disgust','Anger','Contempt']
with open('./label_dict.pkl','rb') as f:
    label_dict = pickle.load(f)
bs = 40
for i in range(8):
    label_dict[i]=label_dict[label_name[i]]
generator = Generator(style_dim=512,n_mlp=4).to(device)
discriminator = Discriminator(256).to(device)

datasets = data_dict('./AffectNet_small_version/train_set')
loaders = loader_dict(datasets,bs)

record_directory = './results/'
def training(generator,discriminator,dataloaders,bs,result_dir,epochs=100,lr=0.0002,reg_interval=8,gp_weight=1):
    global device
    labels = [i for i in range(8)]
    gen_opt = torch.optim.Adam(params=generator.parameters(),lr=lr,betas=[0.5,0.999])
    dis_opt = torch.optim.Adam(params=discriminator.parameters(),lr=lr,betas=[0.5,0.999])
    for epoch in range(epochs):
        for i in range(len(dataloaders[3])):
            #training the discriminator
            generator.eval()
            discriminator.train()
            random.shuffle(labels)
            #이부분, 뱃치에 섞어서할지 아니면 이렇게 레이블마다 한번씩할지 실험해서 결과내야함
            for j in labels:
                reals = next(iter(dataloaders[j])).to(device)
                latents = torch.randn((512)).to(device).unsqueeze(0)
                label = torch.tensor(label_dict[j],dtype=torch.float32).to(device)
                label = label.repeat((bs,1))
                fakes = generator(label,latents,input_is_stylespace=False,input_is_latent=True)
                p_real = discriminator(reals)
                p_fake = discriminator(fakes)


                adv_loss = torch.log(p_real) - torch.log(p_fake)
                reals = next(iter(dataloaders[j])).to(device)
                outs = discriminator(reals)
                gradient_norm = torch.autograd.grad(outputs=[outs.sum()],inputs=[reals],create_graph=True, only_inputs=True)[0]

                loss = adv_loss + gradient_norm
                loss.backward()
                dis_opt.step()
                dis_opt.zero_grad()

            #training the generator
            discriminator.eval()
            generator.train()
            for j in labels:
                latents = torch.randn((bs,512)).to(device)
                labels = label_dict[j]
                fakes = generator(labels,latents,input_is_stylespace=False) # generate the fake samples using the latent code
                p = discriminator(fakes)
                loss = -torch.log(p)
                loss.backward()
                gen_opt.step()
                gen_opt.zero_grad()

training(generator,discriminator,loaders,bs,'.')