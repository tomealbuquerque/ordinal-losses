KS="4 7"
METRICS="wilson mae acc tau roc_auc"
ARCHITECTURES="alexnet googlenet mobilenet_v2 resnet18 resnext50_32x4d shufflenet_v2_x1_0 squeezenet1_0 vgg16 wide_resnet50_2"
METHODS="Base UnimodalCE Beckham OrdinalEncoder CO CO2 HO2"
TOCLASSES="mode mode     mean    mode         mode mode mode"
HEADERS="CE BU PU OE CO CO2 HO2"

echo "Results tables..."
for METRIC in $METRICS; do
    for K in $KS; do
        echo $METRIC $K
        python3 evaluate.py $METRIC $K mode outputs-k$K/* \
            --architectures $ARCHITECTURES \
            --methods $METHODS \
            --toclasses $TOCLASSES \
            --headers $HEADERS \
            > results-$METRIC-$K.tex
    done
done

echo "Aggregate toclass tables..."
METHODS="Base UnimodalCE Beckham OrdinalEncoder CO CO2 HO2"
METRICS="wilson mae acc tau roc_auc gini"
HEADERS="CE BU PU OE CO CO2 HO2"
for K in $KS; do
    echo $K
    python3 evaluate_toclass.py $K outputs-k$K/* \
        --metrics $METRICS \
        --architectures $ARCHITECTURES \
        --methods $METHODS \
        --headers $HEADERS \
        > results-toclass-$K.tex
done

echo "Confusion matrices..."
METHODS="Base HO2"
ARCHITECTURE="wide_resnet50_2"
for METHOD in $METHODS; do
    echo $METHOD
    python3 evaluate_confmatrix.py 7 $ARCHITECTURE $METHOD mode outputs-k7/* > confmat-$METHOD-k7.tex
done

echo "Distribution plots..."
METHODS="Base OrdinalEncoder Beckham HO2"
for TRUEK in 1 3; do
    for METHOD in $METHODS; do
        python3 evaluate_plot.py $TRUEK 7 wide_resnet50_2 $METHOD outputs-k7/*
    done
done
