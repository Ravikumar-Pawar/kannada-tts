import torch, os

def main():
    os.makedirs('pretrained', exist_ok=True)
    torch.save({'dummy': True}, 'pretrained/vits_kannada.pth')
    torch.save({'dummy': True}, 'pretrained/tacotron2_kannada.pth')
    print('Created demo pretrained files.')

if __name__ == '__main__':
    main()
