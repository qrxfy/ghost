import argparse
import torch
from torchvision import transforms
from datasetclass17 import Datasetee17
import os

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 48)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--visual', action='store_true', default=False,
                        help='For visualization')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    torch.cuda.set_device(2) 
    print('model loading...\n')
    net = torch.load('./model/checkpoint.pt')
    net = net.to(device)
    
    print('data loading...\n')
    transform1 = transforms.Compose([
                               transforms.Resize(224),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                           ])
    
    predict_loader = torch.utils.data.DataLoader(
        Datasetee17('./PascalVOC/','predict.txt',transform=transform1),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    
    if(os.path.exists('./preres.txt')):
        os.remove('./preres.txt')
    for batch_idx, (data, target, _) in enumerate(predict_loader):
        print('batch_idx',batch_idx)
        data, target = data.to(device), target.to(device)
        prediction = net(data)
        prediction = torch.clamp(prediction,0,10000000)
        for t_idx, tar in enumerate(target):
            print('第{}张图：'.format(t_idx))
            print('target:{}'.format(tar))
            print('predict:{}'.format(prediction[t_idx]))
            with open('./preres.txt','a') as f:
                f.write('第'+str(t_idx+batch_idx*args.batch_size)+'张图'+'\n')
                tar_t = tar.cpu().numpy().tolist()
                tar_t = [int(zz) for zz in tar_t]
                tar_t = [str(zz) for zz in tar_t]
                tar_t = '['+",".join(tar_t)+']'
                f.write('target:'+tar_t+'\n')
                res = prediction[t_idx].cpu().detach().numpy().tolist()
                res = [str(zz) for zz in res]
                res ='['+",".join(res)+']'
                f.write('result:'+res+'\n')
                
    
if __name__ == '__main__':
    main()
