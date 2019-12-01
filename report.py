import os
import numpy as np
import matplotlib.pyplot as plt

class ModelData:
    def __init__(self, model_name, path, tloss, tacc, vloss, vacc):
        self.model_name = model_name
        self.model_base = model_name.split("_")[0]
        self.path = path
        self.tloss = tloss
        self.tacc = tacc
        self.vloss = vloss
        self.vacc = vacc

    def __repr__(self):
        d = ["model_name: {}".format(self.model_name),
             "path: {}".format(self.path),
             "tloss: {}".format(self.tloss),
             "tacc: {}".format(self.tacc),
             "vloss: {}".format(self.vloss),
             "vacc: {}".format(self.vacc)]

        return " ".join(d)


class Report:
    def __init__(self):
        pass

    def get_model_cv_avg(self, path_dir, k=10):
        with open(os.path.join(path_dir, 'cv.txt'), 'r') as f:
            lines = f.readlines()

            results = lines[k].split()
            results = [float(results[i]) for i in range(1, len(results), 2)]
            
            return tuple(results)

    def get_models_cv_avgs(self, models_folder, k=10):
        files = os.listdir(models_folder)

        models_data = []
        for f in files:
            path = os.path.join(models_folder, f)
            if os.path.isdir(path):
                data = self.get_model_cv_avg(path, k=k)

                model_data = ModelData(
                    f, path, data[0], data[1], data[2], data[3])

                models_data.append(model_data)

        return models_data

    def plot_cv_data(self, models_folder, stat='acc', k=10):
        results = self.get_models_cv_avgs(models_folder, k=k)
        width = 0.20

        labels = ["LeNet5", "Suggested Net", "Linear"]
        x = np.arange(len(labels) - 1)
        x_pure = [-0.3, 0.7, 2]
        x_bn = [-0.1, 0.9]
        x_do = [0.1, 1.1]
        x_bn_do = [0.3, 1.3]

        if stat == 'acc':
            pure_means = []
            bn_means = []
            do_means = []
            bn_do_means = []

            for m in results:
                if 'bn' in m.model_name and 'dropout' in m.model_name:
                    bn_do_means.append(m.vacc)
                elif 'bn' in m.model_name:
                    bn_means.append(m.vacc)
                elif 'dropout' in m.model_name:
                    do_means.append(m.vacc)
                else:
                    pure_means.append(m.vacc)

        fig, ax = plt.subplots()
        rects1 = ax.bar(x_pure, pure_means, width, label='Pure')
        rects2 = ax.bar(x_bn, bn_means, width, label='BN')
        rects3 = ax.bar(x_do, do_means, width, label='Dropout')
        rects4 = ax.bar(x_bn_do, bn_do_means, width, label='BN + Dropout')
        
        # rects2 = ax.bar(x + width/2, women_means, width, label='Women')
        ax.set_ylabel('Accuracy')
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(labels)
        ax.set_ylim([0.7, 1])
        ax.legend()

        fig.tight_layout()
        plt.show()


def test_cv_avg():
    path_dir = ("/home/vaitses/Desktop/tmp/"
                "tu-kl-mlp-neural-networks/"
                "logs/model-select/models/lenet5")

    report = Report()

    tloss, tacc, vloss, vacc = report.get_model_cv_avg(path_dir)
    print(tloss, tacc, vloss, vacc)


def test_models_data():
    path_dir = ("/home/vaitses/Desktop/tmp/"
                "tu-kl-mlp-neural-networks/"
                "logs/model-select/models")

    report = Report()
    report.get_models_cv_avgs(path_dir)


def comparison_models_acc(models_folder):
    report = Report()
    report.plot_cv_data(models_folder, stat='acc')


if __name__ == "__main__":
    path = ("/home/vaitses/Desktop/tmp/"
            "tu-kl-mlp-neural-networks/"
            "logs/model-select/models")

    comparison_models_acc(path)




