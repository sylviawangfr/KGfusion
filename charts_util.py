import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

OUT_FILE_PREFIX = "../outputs/figs/"
plt.rcParams.update({
    "text.usetex": True
})


def draw_bar(in_file, sheet_name):
    df_sheet = pd.read_excel(in_file, sheet_name=sheet_name, header=0)
    models = df_sheet['Models'].dropna().to_list()
    legends = ["Head", "Tail", "Both"]
    umls_mrr = df_sheet["Unnamed: 4"][1:].to_list()
    umls_mrr = np.array(umls_mrr).reshape(-1, 3)
    wn_mrr = df_sheet["Unnamed: 8"][1:].to_list()
    wn_mrr = np.array(wn_mrr).reshape(-1, 3)
    umls_data = {legends[i]: umls_mrr[:,i] for i in range(len(legends))}
    wn_data = {legends[i]: wn_mrr[:,i] for i in range(len(legends))}

    def bar1(data, out_file, y1, y2):
        plt.rcParams.update({
            "text.usetex": True
        })
        fig, ax = plt.subplots()
        width=0.20
        multiplier = 0
        x = np.arange(len(models))
        for attr, measurement in data.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attr)
            # ax.bar_label(rects, padding=3)
            multiplier += 1
        ax.set_ylabel('Mean Reciprocal Rank')
        ax.set_title('KG fusion result')
        ax.set_xticks(x + width, np.arange(1, 10))
        ax.legend(loc='upper left')
        ax.set_ylim(y1, y2)
        if len(out_file):
            plt.savefig(out_file, dpi=600)
        plt.show()

    # bar1(umls_data, '', 0.5, 1)
    bar1(wn_data, '', 0, 0.7)
    return


if __name__ == "__main__":
    draw_bar("kgFusion_resutls.xlsx", "Sheet1")
