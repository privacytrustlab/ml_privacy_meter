# End-to-end test 
This folder provides settings for running population attacks and reference attacks on cifar10, following the same setting as Experimental Setup I in [Enhanced Membership Inference Attacks against Machine Learning Models](https://arxiv.org/abs/2111.09679).

## Reproducing the population attack results 
```
cd ../../
python main.py --cf tests/end_to_end_tests/config_blackbox_test_population.yaml
```
Expected (Table 4 of the appendix)
```
train acc: 96.2±0.046
test acc: 40.9±0.029
auc of the population attack: 0.857±0.023
```

Test Outcome:
```
train acc: 0.992
test acc: 0.4316
auc of the population attack: 0.870
```

## Reproducing the reference attack results 
```
cd ../../
python main.py --cf tests/end_to_end_tests/config_blackbox_test_reference.yaml
```
Expected (Table 4 of the appendix)
```
train acc: 96.2±0.046
test acc: 40.9±0.029
auc of the reference attack: 0.874±0.018
```

Test Outcome:
```
train acc: 0.992
test acc: 0.4316
auc of the reference attack: 0.879
```
