import numpy as np 
impoty json
import matplotlib.pyplot as plt

def compare_models():
    with open('logs/attack/results', 'w+') as json_file:
        data = json.load(json_file)['result']
        n_groups = len(data)
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.35
        opacity = 0.8
        model_names = []
        target_acc = []
        attack_acc = []
        for result in data:
            for key, value in data.items():
                model_names.append(key)
                target_acc.append(value['target_acc'])
                attack_acc.append(value['attack_acc'])
            
        rects1 = plt.bar(index, target_acc, bar_width, alpha=opacity, label='Target model Acc')
        rects2 = plt.bar(index, attack_acc, bar_width, alpha=opacity, label='Inference model Acc')

        plt.xlabel('Accuracy')
        plt.ylabel('Attack/Model setup')

        plt.xticks(index + bar_width, model_names)
        plt.legend()
        plt.tight_layout()

        plt.savefig('logs/attack/comparison.png')