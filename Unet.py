import torch
import torch.nn as nn

   
########################################################################################################################
def create_upconv(in_channels, out_channels, size=None):
    return nn.Sequential(
        nn.Upsample(size=size, mode='nearest')
        , nn.Conv2d(in_channels,out_channels,3,1,1)
        , nn.BatchNorm2d(num_features=out_channels)
        , nn.LeakyReLU(inplace=True)
        , nn.Conv2d(out_channels,out_channels,3,1,1)
        , nn.BatchNorm2d(num_features=out_channels)
        , nn.LeakyReLU(inplace=True)
        )

class Unet(nn.Module):
    def __init__(self, num_class, num_input=1):
        super().__init__()
        
        n1 = 4
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.conv_l1 = nn.Sequential(
            nn.Conv2d(num_input,filters[0],3,1,1)
            , nn.BatchNorm2d(num_features=filters[0])
            , nn.LeakyReLU(inplace=True)
            , nn.Conv2d(filters[0],filters[0],3,1,1)
            , nn.BatchNorm2d(num_features=filters[0])
            , nn.LeakyReLU(inplace=True)
            )

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l2 = nn.Sequential(
            nn.Conv2d(filters[0],filters[1],3,1,1)
            , nn.BatchNorm2d(num_features=filters[1])
            , nn.LeakyReLU(inplace=True)
            , nn.Conv2d(filters[1],filters[1],3,1,1)
            , nn.BatchNorm2d(num_features=filters[1])
            , nn.LeakyReLU(inplace=True)
            )

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l3 = nn.Sequential(
            nn.Conv2d(filters[1],filters[2],3,1,1)
            , nn.BatchNorm2d(num_features=filters[2])
            , nn.LeakyReLU(inplace=True)
            , nn.Conv2d(filters[2],filters[2],3,1,1)
            , nn.BatchNorm2d(num_features=filters[2])
            , nn.LeakyReLU(inplace=True)
            )

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l4 = nn.Sequential(
            nn.Conv2d(filters[2],filters[3],3,1,1)
            , nn.BatchNorm2d(num_features=filters[3])
            , nn.LeakyReLU(inplace=True)
            , nn.Conv2d(filters[3],filters[3],3,1,1)
            , nn.BatchNorm2d(num_features=filters[3])
            , nn.LeakyReLU(inplace=True)
            )

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l5 = nn.Sequential(
            nn.Conv2d(filters[3],filters[4],3,1,1)
            , nn.BatchNorm2d(num_features=filters[4])
            , nn.LeakyReLU(inplace=True)
            , nn.Conv2d(filters[4],filters[4],3,1,1)
            , nn.BatchNorm2d(num_features=filters[4])
            , nn.LeakyReLU(inplace=True)
            )

        self.deconv_u4 = create_upconv(in_channels=filters[4], out_channels=filters[3], size=(53,30))

        self.conv_u4 = nn.Sequential(
            nn.Conv2d(filters[4],filters[3],3,1,1)
            , nn.BatchNorm2d(num_features=filters[3])
            , nn.LeakyReLU(inplace=True)
            , nn.Conv2d(filters[3],filters[3],3,1,1)
            , nn.BatchNorm2d(num_features=filters[3])
            , nn.LeakyReLU(inplace=True)
            )

        self.deconv_u3 = create_upconv(in_channels=filters[3], out_channels=filters[2], size=(106,60))

        self.conv_u3 = nn.Sequential(
            nn.Conv2d(filters[3],filters[2],3,1,1)
            , nn.BatchNorm2d(num_features=filters[2])
            , nn.LeakyReLU(inplace=True)
            , nn.Conv2d(filters[2],filters[2],3,1,1)
            , nn.BatchNorm2d(num_features=filters[2])
            , nn.LeakyReLU(inplace=True)
            )

        self.deconv_u2 = create_upconv(in_channels=filters[2], out_channels=filters[1], size=(213,120))

        self.conv_u2 = nn.Sequential(
            nn.Conv2d(filters[2],filters[1],3,1,1)
            , nn.BatchNorm2d(num_features=filters[1])
            , nn.LeakyReLU(inplace=True)
            , nn.Conv2d(filters[1],filters[1],3,1,1)
            , nn.BatchNorm2d(num_features=filters[1])
            , nn.LeakyReLU(inplace=True)
            )

        self.deconv_u1 = create_upconv(in_channels=filters[1], out_channels=filters[0], size=(426,240))

        self.conv_u1 = nn.Sequential(
            nn.Conv2d(filters[1],filters[0],3,1,1)
            , nn.BatchNorm2d(num_features=filters[0])
            , nn.LeakyReLU(inplace=True)
            , nn.Conv2d(filters[0],filters[0],3,1,1)
            , nn.BatchNorm2d(num_features=filters[0])
            , nn.LeakyReLU(inplace=True)
            )
        
        self.conv1x1_out = nn.Conv2d(n1, num_class, 1, 1, 0, bias=True)
        
    def forward(self, x):

        output1 = self.conv_l1(x)
        input2 = self.maxpool1(output1)
        
        output2 = self.conv_l2(input2)
        input3 = self.maxpool2(output2)
        
        output3 = self.conv_l3(input3)
        input4 = self.maxpool3(output3)
        
        output4 = self.conv_l4(input4)
        input5 = self.maxpool4(output4)
        
        output5 = self.conv_l5(input5)
        input6 = self.deconv_u4(output5)
        
        output6 = self.conv_u4(torch.cat((input6, output4), dim=1))
        input7 = self.deconv_u3(output6)
        
        output7 = self.conv_u3(torch.cat((input7, output3), dim=1))
        input8 = self.deconv_u2(output7)
        
        output8 = self.conv_u2(torch.cat((input8, output2), dim=1))
        input9 = self.deconv_u1(output8)
        
        output9 = self.conv_u1(torch.cat((input9, output1), dim=1))
        
        out = self.conv1x1_out(output9)
        
        return out

class Unet_part(nn.Module):
    def __init__(self):
        super().__init__()
        
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.conv_l1 = nn.Sequential(
            nn.Conv2d(3,filters[0],3,1,1)
            , nn.BatchNorm2d(num_features=filters[0])
            , nn.LeakyReLU(inplace=True)
            , nn.Conv2d(filters[0],filters[0],3,1,1)
            , nn.BatchNorm2d(num_features=filters[0])
            , nn.LeakyReLU(inplace=True)
            )

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l2 = nn.Sequential(
            nn.Conv2d(filters[0],filters[1],3,1,1)
            , nn.BatchNorm2d(num_features=filters[1])
            , nn.LeakyReLU(inplace=True)
            , nn.Conv2d(filters[1],filters[1],3,1,1)
            , nn.BatchNorm2d(num_features=filters[1])
            , nn.LeakyReLU(inplace=True)
            )

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l3 = nn.Sequential(
            nn.Conv2d(filters[1],filters[2],3,1,1)
            , nn.BatchNorm2d(num_features=filters[2])
            , nn.LeakyReLU(inplace=True)
            , nn.Conv2d(filters[2],filters[2],3,1,1)
            , nn.BatchNorm2d(num_features=filters[2])
            , nn.LeakyReLU(inplace=True)
            )

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l4 = nn.Sequential(
            nn.Conv2d(filters[2],filters[3],3,1,1)
            , nn.BatchNorm2d(num_features=filters[3])
            , nn.LeakyReLU(inplace=True)
            , nn.Conv2d(filters[3],filters[3],3,1,1)
            , nn.BatchNorm2d(num_features=filters[3])
            , nn.LeakyReLU(inplace=True)
            )

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l5 = nn.Sequential(
            nn.Conv2d(filters[3],filters[4],3,1,1)
            , nn.BatchNorm2d(num_features=filters[4])
            , nn.LeakyReLU(inplace=True)
            , nn.Conv2d(filters[4],filters[4],3,1,1)
            , nn.BatchNorm2d(num_features=filters[4])
            , nn.LeakyReLU(inplace=True)
            )

        self.deconv_u4 = create_upconv(in_channels=filters[4], out_channels=filters[3], size=(53,30))

        self.conv_u4 = nn.Sequential(
            nn.Conv2d(filters[4],filters[3],3,1,1)
            , nn.BatchNorm2d(num_features=filters[3])
            , nn.LeakyReLU(inplace=True)
            , nn.Conv2d(filters[3],filters[3],3,1,1)
            , nn.BatchNorm2d(num_features=filters[3])
            , nn.LeakyReLU(inplace=True)
            )

        self.deconv_u3 = create_upconv(in_channels=filters[3], out_channels=filters[2], size=(106,60))

        self.conv_u3 = nn.Sequential(
            nn.Conv2d(filters[3],filters[2],3,1,1)
            , nn.BatchNorm2d(num_features=filters[2])
            , nn.LeakyReLU(inplace=True)
            , nn.Conv2d(filters[2],filters[2],3,1,1)
            , nn.BatchNorm2d(num_features=filters[2])
            , nn.LeakyReLU(inplace=True)
            )

        self.deconv_u2 = create_upconv(in_channels=filters[2], out_channels=filters[1], size=(213,120))

        self.conv_u2 = nn.Sequential(
            nn.Conv2d(filters[2],filters[1],3,1,1)
            , nn.BatchNorm2d(num_features=filters[1])
            , nn.LeakyReLU(inplace=True)
            , nn.Conv2d(filters[1],filters[1],3,1,1)
            , nn.BatchNorm2d(num_features=filters[1])
            , nn.LeakyReLU(inplace=True)
            )

        self.deconv_u1 = create_upconv(in_channels=filters[1], out_channels=filters[0], size=(426,240))

        self.conv_u1 = nn.Sequential(
            nn.Conv2d(filters[1],filters[0],3,1,1)
            , nn.BatchNorm2d(num_features=filters[0])
            , nn.LeakyReLU(inplace=True)
            , nn.Conv2d(filters[0],filters[0],3,1,1)
            , nn.BatchNorm2d(num_features=filters[0])
            , nn.LeakyReLU(inplace=True)
            )
        
    def forward(self, x):

        output1 = self.conv_l1(x)
        input2 = self.maxpool1(output1)
        
        output2 = self.conv_l2(input2)
        input3 = self.maxpool2(output2)
        
        output3 = self.conv_l3(input3)
        input4 = self.maxpool3(output3)
        
        output4 = self.conv_l4(input4)
        input5 = self.maxpool4(output4)
        
        output5 = self.conv_l5(input5)
        input6 = self.deconv_u4(output5)
        
        output6 = self.conv_u4(torch.cat((input6, output4), dim=1))
        input7 = self.deconv_u3(output6)
        
        output7 = self.conv_u3(torch.cat((input7, output3), dim=1))
        input8 = self.deconv_u2(output7)
        
        output8 = self.conv_u2(torch.cat((input8, output2), dim=1))
        input9 = self.deconv_u1(output8)
        
        output9 = self.conv_u1(torch.cat((input9, output1), dim=1))
        
        return output9

class Unet_head(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 32)
            )
        
    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

class Unet_last(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        
        self.conv1x1_out = nn.Conv2d(64, num_class, 1, 1, 0, bias=True)
        #self.softmax = nn.Softmax2d()
    
    def forward(self, x):
        x = self.conv1x1_out(x)
        #x = self.softmax(x)
        return x

class Unet_softmax(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.softmax = nn.Softmax2d()
    
    def forward(self, x):
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #a = torch.randn((1,3,426,240)).to(device)
    oper = Unet(1).to(device)
    #b = oper(a)
    #print(b.shape)
    
    parameter = list(oper.parameters())

    cnt = 0
    for i in range(len(parameter)):
        cnt += parameter[i].reshape(-1).shape[0]
    
    print(cnt)
    
    #oper2 = Unet_head().to(device)
    #c = oper2(b)
    #print(c.shape)
    
    