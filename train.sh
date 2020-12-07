#!/bin/sh
METHODS="Base Beckham OrdinalEncoder UnimodalCE HO2 CO2 CO"
KS="7 4"
FOLDS=`seq 0 9`
#ARCHITECTURES="shufflenet_v2_x1_0 squeezenet1_0 resnet18 googlenet mnasnet1_0 vgg16"
#ARCHITECTURES="alexnet densenet161 mobilenet_v2"
ARCHITECTURES="inception_v3 resnext50_32x4d wide_resnet50_2"

for A in $ARCHITECTURES; do
    for K in $KS; do
        for M in $METHODS; do
            for F in $FOLDS; do
                python3 -u train.py $A $M $K $F
            done
        done
    done
done
