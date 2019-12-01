import os
import numpy as np
import matplotlib.pyplot as plt

class ModelData:
    def __init__(self, model_name, path, tloss, tacc, vloss, vacc, 
                 epochs_for_convergence=None):
        self.model_name = model_name
        self.model_base = model_name.split("_")[0]
        self.path = path
        self.tloss = tloss
        self.tacc = tacc
        self.vloss = vloss
        self.vacc = vacc
        self.epochs = epochs_for_convergence

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

    def get_avg_epochs_for_convergence(self, path_dir):
        files = os.listdir(path_dir)

        epochs = 0
        count = 0
        for f in files:
            if 'checkpoint' in f:
                epochs += int(f[-5:-3])
                count += 1

        return epochs/count

    def get_models_cv_avgs(self, models_folder, k=10):
        files = os.listdir(models_folder)

        models_data = []
        for f in files:
            path = os.path.join(models_folder, f)
            if os.path.isdir(path):
                data = self.get_model_cv_avg(path, k=k)
                epochs = self.get_avg_epochs_for_convergence(path)

                model_data = ModelData(
                    f, path, data[0], data[1], data[2], data[3],
                    epochs_for_convergence=epochs)

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

        pure_means = [0,0,0]
        bn_means = [0,0]
        do_means = [0,0]
        bn_do_means = [0,0]

        if stat == 'acc':
            ylabel = 'Accuracy'
            ylim = [0.7, 1.0]
            for m in results:
                if 'lenet5' in m.model_name:
                    idx = 0
                elif 'net' in m.model_name:
                    idx = 1
                else:
                    idx = 2

                if 'bn' in m.model_name and 'dropout' in m.model_name:
                    bn_do_means[idx] = (m.vacc)
                elif 'bn' in m.model_name:
                    bn_means[idx] = m.vacc
                elif 'dropout' in m.model_name:
                    do_means[idx] = m.vacc
                else:
                    pure_means[idx] = m.vacc

        elif stat == 'loss':
            ylabel = 'Loss'
            ylim = [0.0, 1.75]
            for m in results:
                if 'lenet5' in m.model_name:
                    idx = 0
                elif 'net' in m.model_name:
                    idx = 1
                else:
                    idx = 2

                if 'bn' in m.model_name and 'dropout' in m.model_name:
                    bn_do_means[idx] = (m.vloss)
                elif 'bn' in m.model_name:
                    bn_means[idx] = m.vloss
                elif 'dropout' in m.model_name:
                    do_means[idx] = m.vloss
                else:
                    pure_means[idx] = m.vloss

        fig, ax = plt.subplots()
        rects1 = ax.bar(x_pure, pure_means, width, label='Pure')
        rects2 = ax.bar(x_bn, bn_means, width, label='BN')
        rects3 = ax.bar(x_do, do_means, width, label='Dropout')
        rects4 = ax.bar(x_bn_do, bn_do_means, width, label='BN + Dropout')
        
        ax.set_ylabel(ylabel)
        ax.set_xticks([0,1,2])
        ax.set_xticklabels(labels)
        ax.set_ylim(ylim)
        ax.legend()

        fig.tight_layout()
        plt.show()


def comparison_models_acc(models_folder):
    report = Report()
    report.plot_cv_data(models_folder, stat='acc', k=5)


def comparison_models_loss(models_folder):
    report = Report()
    report.plot_cv_data(models_folder, stat='loss', k=5)


if __name__ == "__main__":
    # python3 model-select.py should be run first to create the folder
    path = "./logs/model-select/models_k5"

    comparison_models_acc(path)
    comparison_models_loss(path)




