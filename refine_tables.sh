#!/bin/sh
sed -i 's/alexnet/AlexNet/g' table*.tex
sed -i 's/googlenet/GoogLeNet/g' table*.tex
sed -i 's/mobilenet/MobileNet/g' table*.tex
sed -i 's/resnet/ResNet/g' table*.tex
sed -i 's/resnext/ResNeXt/g' table*.tex
sed -i 's/shufflenet/ShuffleNet/g' table*.tex
sed -i 's/squeezenet/SqueezeNet/g' table*.tex
sed -i 's/vgg16/VGG16/g' table*.tex
sed -i 's/wide/Wide/g' table*.tex
sed -i 's/wilson/UOC/g' table*.tex
sed -i 's/mae/MAE/g' table*.tex
sed -i 's/acc/Accuracy/g' table*.tex
sed -i 's/tau/Kendall'"'"'s $\\tau$/g' table*.tex
sed -i 's/roc\\_auc/ROC AUC/g' table*.tex
sed -i 's/gini/Gini/g' table*.tex
