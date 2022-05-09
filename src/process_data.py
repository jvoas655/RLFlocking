import matplotlib.pyplot as plt
import numpy as np
import os

cmap = plt.get_cmap("tab20")

def make_plot(data, start, skip, title, labels, filepath = None):
    fig, axes = plt.subplots(1, 1)
    #fig.tight_layout()
    axes.set_title(title)
    stats_c = []
    stats_d = []
    for line in range(data.shape[0]):
        axes.plot(range(0, data.shape[1] * skip - start, skip), data[line, start // skip:], label = labels[line])
        if ("Eval" in title and "50" in title and ("Collisions" in title)):
            stats_c.append((labels[line], np.median(np.sort(data[line, start // skip:])[:4])))
        if ("Eval" in title and "50" in title and ("Density" in title and not "Density2" in title)):
            stats_d.append((labels[line], np.median(np.sort(data[line, start // skip:])[:4])))
    axes.set_xlabel("Episode")
    axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    fig.tight_layout()
    if (filepath):
        plt.savefig(filepath + ".png")
    else:
        plt.show()
    plt.close(fig)
    return stats_c, stats_d
tests = ["Variance", "Gamma", "Batch", "Hidden", "Neighbor", "Center", "Observe", ("Observe", "observe_no_npos_norm", "Filtered Observe"), "Algo", "Baseline", "Agents", "Maneuverability"]
all_stats_c = []
all_stats_d = []
result_dir = "..\\results\\"
eval_titles = ["Collisions", "Density", "Density2"]
training_titles = ["Rewards", "Collisions", "Density", "Density2", "Critic", "Actor"]
for test in tests:
    eval_data = []
    training_data = []
    eval_labels = []
    training_labels = []
    for path in os.listdir(result_dir):
        if (isinstance(test, str)):
            if not (test.lower() in path or "baseline" in path):
                continue
        else:
            if test[1] in path or not (test[0].lower() in path or "baseline" in path):
                continue
        full_path = os.path.join(result_dir, path)
        if (os.path.isfile(full_path)):
            continue
        label_set_e = False
        label_set_t = False
        for result_file in os.listdir(full_path):
            result_path = os.path.join(full_path, result_file)
            result_parts = result_file.split(".")
            if (len(result_parts) > 1 and result_parts[1] == "npy"):
                if ("training" in result_file):
                    target = training_data
                    if (not label_set_t):
                        training_labels.append(path)
                        label_set_t = True
                elif ("eval" in result_file):
                    target = eval_data
                    if (not label_set_e):
                        eval_labels.append(path)
                        label_set_e = True
                data = np.load(result_path)
                for ind in range(data.shape[0]):
                    while (len(target) <= ind):
                        target.append([])
                    target[ind].append(data[ind, :])

    print(training_labels)
    if (isinstance(test, str)):
        ftitle = test
    else:
        ftitle = test[2]
    for i in [0, 50]:
        for j in range(len(training_data)):
            stats_c, stats_d = make_plot(np.array(training_data[j]), i, 1, ftitle + " Train " + str(i) + ": " + training_titles[j], training_labels, result_dir + "_".join(ftitle.lower().split()) + "_train_" + str(i) + "_" + training_titles[j].lower())
            all_stats_c += stats_c
            all_stats_d += stats_d
    for i in [0, 50]:
        for j in range(len(eval_data)):
            stats_c, stats_d = make_plot(np.array(eval_data[j]), i, 10, ftitle + " Eval " + str(i) + ": " + eval_titles[j], eval_labels, result_dir + "_".join(ftitle.lower().split()) + "_eval_" + str(i) + "_" + eval_titles[j].lower())
            all_stats_c += stats_c
            all_stats_d += stats_d
found = []
filtered_stats_c = []
for i in all_stats_c:
    if not (i[0] in found):
        filtered_stats_c.append(i)
        found.append(i[0])
found = []
filtered_stats_d = []
for i in all_stats_d:
    if not (i[0] in found):
        filtered_stats_d.append(i)
        found.append(i[0])

colors = ["Variance", "Gamma", "Batch", "Hidden", "Neighbor", "Center", "Observe", "Algo", "Baseline", "Agents", "Maneuverability"]

set_to_c = {}
for ind in range(len(colors)):
    set_to_c[colors[ind]] = cmap(ind)


c_data = list(map(lambda l: l[1], filtered_stats_c))
c_labels = list(map(lambda l: " ".join(l[0].split("_")), filtered_stats_c))
c_colors = {}
c_color_list = []
for lab in c_labels:
    for sub_ind in colors:
        if (sub_ind.lower() in lab):
            c_colors[lab] = set_to_c[sub_ind]
            c_color_list.append(set_to_c[sub_ind])
fig, axes = plt.subplots(1, 1, figsize=(9,5))
axes.set_title("Collisions Ablation")
axes.set_ylabel("Average Collisions per Agent")
axes.set_xlabel("Ablation")
axes.bar(range(len(c_data)), c_data, color=c_color_list)
axes.set_xticks(range(len(c_data)), fontsize=8)
handles = [plt.Rectangle((0,0),0.5, 0.5, color=c_colors[label]) for label in c_labels]
plt.legend(handles, [str(label_ind) + " = " + c_labels[label_ind] for label_ind in range(len(c_labels))], fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
fig.tight_layout()
plt.savefig(result_dir + "collision_ablation.png")
plt.close(fig)

d_data = list(map(lambda l: l[1], filtered_stats_d))
d_labels = list(map(lambda l: " ".join(l[0].split("_")), filtered_stats_d))
d_colors = {}
d_color_list = []
for lab in d_labels:
    for sub_ind in colors:
        if (sub_ind.lower() in lab):
            d_colors[lab] = set_to_c[sub_ind]
            d_color_list.append(set_to_c[sub_ind])
fig, axes = plt.subplots(1, 1, figsize=(9,5))
axes.set_title("Density Ablation")
axes.set_xlabel("Ablation")
axes.set_ylabel("Average Density per Agent")
axes.bar(range(len(d_data)), d_data, color=d_color_list)
axes.set_xticks(range(len(d_data)), fontsize=8)
handles = [plt.Rectangle((0,0),0.5,0.5, color=d_colors[label]) for label in d_labels]
plt.legend(handles, [str(label_ind) + " = " + d_labels[label_ind] for label_ind in range(len(d_labels))], fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
fig.tight_layout()
plt.savefig(result_dir + "density_ablation.png")
plt.close(fig)
