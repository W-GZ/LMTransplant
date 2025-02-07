import os
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import json_jsonl_read, json_jsonl_write


def figure_draw(em_scores, keys_to_plot, figure_name):
    if len(keys_to_plot) == 0:
        return
    
    sns.set_theme(style="ticks")  # 设置Seaborn样式  # darkgrid, whitegrid, dark, white, ticks

    fig, ax = plt.subplots(figsize=(10, 5))

    for key, values in em_scores.items():
        if key in keys_to_plot:
            # ax.plot(range(1, len(values) + 1), values, label=key, linewidth=1.5, alpha=1.0)

            values_ = list(values)
            values_.insert(0, 0.1187214611872146)
            ax.plot(values_, label=key, linewidth=1.5, alpha=1.0)

    ax.legend(loc='lower right')

    ax.set_title('EM Scores', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('EM Score', fontsize=12)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # 添加网格线
    ax.grid(True)

    # # 添加水平线和垂直线，例如在y=0.5和x=5的位置
    # ax.axhline(y=0.5, color='r', linestyle='--')  # 水平线
    # ax.axvline(x=5, color='g', linestyle='--')  # 垂直线


    plt.savefig(figure_name, dpi=300)  # 保存图表时设置高分辨率

    # plt.show()
    plt.close(fig)


if __name__ == "__main__":
    base_path = "./save_models_qwen1.5_0.5B/"

    dataset_name = "MLQA_"
    # augmentation_num = 3
    # exp_ids = [0, 1, 2, 3]

    # augmentation_num = 9
    # exp_ids = [0]

    augmentation_num = 3
    exp_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    augmenters = ["none", "MoreData", "Copy", "EDA", "BackTranslation", "GPT3Mix", "AugGPT", "llmda"]
    seednums = [50] # , 100, 400]

    em_scores_average = {}
    for exp_id in exp_ids:
        em_scores = {}
        for augmenter in augmenters:
            for seednum in seednums:

                # if exp_id in [4, 5, 6, 7, 8, 9] and seednum in [100, 400]:
                #     continue

                subsample_name = f'train_subsample_{int(seednum):03}'
                save_path = base_path + f"{dataset_name}/augmentation_num_{augmentation_num:02}/exp_{exp_id:02}/" + f"{augmenter}/{subsample_name}_{augmenter}/"

                record_path = os.path.join(save_path, 'epochs_performace/record.jsonl')
                if os.path.exists(record_path):
                    epochs_performace_record = json_jsonl_read(record_path)
                    em_score = [item["em_score"] for item in epochs_performace_record]

                    if len(em_score) < 50:
                        continue
                    else:
                        # em_score = em_score[:50]
                        em_scores[f"{augmenter}_{seednum}"] = em_score

                        em_scores_average[f"{augmenter}_{seednum}_{exp_id}"] = em_score

        # print(list(em_scores.keys()))
        
        keys_to_plot_all = []
        for seednum in seednums:
            keys_to_plot = []
            for augmenter in augmenters:

                # if exp_id in [4, 5, 6, 7, 8, 9] and seednum in [100, 400]:# and augmenter != "llmda":
                #     continue
        
                subsample_name = f'train_subsample_{int(seednum):03}'
                save_path = base_path + f"{dataset_name}/augmentation_num_{augmentation_num:02}/exp_{exp_id:02}/" + f"{augmenter}/{subsample_name}_{augmenter}/"
                record_path = os.path.join(save_path, 'epochs_performace/record.jsonl')
                if os.path.exists(record_path):

                    name_ = f"{augmenter}_{seednum}"
                    keys_to_plot_all.append(name_)
                    keys_to_plot.append(name_)

            figure_draw(em_scores, keys_to_plot, base_path + f"{dataset_name}/augmentation_num_{augmentation_num:02}/exp_{exp_id:02}/" + f"exp_{exp_id:02}_{seednum}.png")

        # # figure_draw(em_scores, keys_to_plot_all, base_path + f"{dataset_name}/augmentation_num_{augmentation_num:02}/exp_{exp_id:02}/" + f"exp_{exp_id:02}_ALL.png")



    em_scores_average_ = {}
    em_scores_average__ = {}
    for augmenter in augmenters:
        for seednum in seednums:
            count_ = 0
            temp_list = []
            for exp_id in exp_ids:
                if f"{augmenter}_{seednum}_{exp_id}" in em_scores_average:
                    temp_list.append(em_scores_average[f"{augmenter}_{seednum}_{exp_id}"])
                    count_ += 1

            if count_ > 0:
                em_scores_average_[f"{augmenter}_{seednum}"] = [sum(values) / count_ for values in zip(*temp_list)]

                temp = em_scores_average_[f"{augmenter}_{seednum}"][10:20]

                mean_value = sum(temp) / len(temp)
                variance = sum((x - mean_value) ** 2 for x in temp) / len(temp)
                std_deviation = variance ** 0.5


                em_scores_average__[f"{augmenter}_{seednum}"] = {
                    "em": mean_value, 
                    "em_std": std_deviation,
                }



    print(em_scores_average__)

    formatted_data = {key: [round(value["em"] * 100, 2), round(value["em_std"] * 100, 2)] for key, value in em_scores_average__.items()}
    # 打印格式化后的数据
    for key, value in formatted_data.items():
        print(f"{key}: {value}")



    for seednum in seednums:
        keys_to_plot = []
        for augmenter in augmenters:
            name_ = f"{augmenter}_{seednum}"
            keys_to_plot.append(name_)

        figure_draw(em_scores_average_, keys_to_plot, base_path + f"{dataset_name}/augmentation_num_{augmentation_num:02}/" + f"average_{seednum}.png")

