import sys

sys.path.append("../ai-optim/")

import models
from models import *

# Model
print("==> Building model..")
net = VGG("VGG16")
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # return {"Total": total_num, "Trainable": trainable_num}
    return {"Total": total_num}
    # return {"Trainable": trainable_num}


N = get_parameter_number(net)


print("Number of parameter: %s\n" % N)
# print("Number of parameter: %s" % M)
