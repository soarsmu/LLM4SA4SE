import matplotlib.pyplot as plt

def draw_line_figures(y1, y2, title):
    # Data for the x-axis
    x = ["Gerrit", "GitHub", "GooglePlay", "Jira", "StackOverflow"]

    # Data for the first and second lines (y1 and y2)
    plt.rcParams.update({'font.size': 14, 'legend.fontsize': 14, 'axes.labelsize': 14, 'axes.titlesize': 14})

    plt.plot(x, y2, label='Best Few-shot', linestyle='-', marker='o', color='b')

    # Create the second line plot
    plt.plot(x, y1, label='Best Zero-shot', linestyle='--', marker='x', color='r')

    # Add a legend
    plt.legend()

    plt.xlabel('Dataset', fontsize=14)  # Set the x-axis label font size
    plt.ylabel(title, fontsize=14)  # Set the y-axis label font size
    plt.savefig('figures/zsl-fsl-{}.png'.format(title.lower()), dpi=600, bbox_inches="tight")

    # Display the plot
    plt.show()

def draw_venn_diagrams_zero():
    from matplotlib_venn import venn3
    import matplotlib.pyplot as plt

    # Define the sizes of the sets
    set1 = 456  # vicuna
    set2 = 431  # wizardlm
    set3 = 424  # llama2
    overlap12 = 33  # Size of the overlap between sets 1 and 2
    overlap13 = 22   # Size of the overlap between sets 1 and 3
    overlap23 = 14   # Size of the overlap between sets 2 and 3
    overlap123 = 370 # Size of the overlap between all three sets

    # Create the Venn diagram
    venn = venn3(subsets=(set1, set2, overlap12, set3, overlap13, overlap23, overlap123),
                set_labels=('Vicuna', 'WizardLM', 'LLaMA 2-Chat'))

    # Customize the Venn diagram (optional)
    venn.get_label_by_id('100').set_text('31')
    venn.get_label_by_id('010').set_text('14')
    venn.get_label_by_id('001').set_text('18')

    # Display the diagram
    plt.savefig('figures/venn-zero-shot.png', dpi=300, bbox_inches="tight")
    plt.show()
    
def draw_venn_diagrams_few():
    from matplotlib_venn import venn3
    import matplotlib.pyplot as plt

    # Define the sizes of the sets
    set1 = 456  # vicuna
    set2 = 454  # wizardlm
    set3 = 406  # llama2
    overlap12 = 66  # Size of the overlap between sets 1 and 2
    overlap13 = 14   # Size of the overlap between sets 1 and 3
    overlap23 = 21   # Size of the overlap between sets 2 and 3
    overlap123 = 350 # Size of the overlap between all three sets

    # Create the Venn diagram
    venn = venn3(subsets=(set1, set2, overlap12, set3, overlap13, overlap23, overlap123),
                set_labels=('Vicuna', 'WizardLM', 'LLaMA 2-Chat'))

    # Customize the Venn diagram (optional)
    venn.get_label_by_id('100').set_text('26')
    venn.get_label_by_id('010').set_text('17')
    venn.get_label_by_id('001').set_text('21')

    # Display the diagram
    plt.savefig("figures/venn-few-shot.png", dpi=300, bbox_inches="tight")
    plt.show()

def draw_heatmap():
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Sample data (replace with your actual data)
    data = {
        '0-shot': [0.69, 0.71,	0.8,	0.81,	0.41],
        '1-shot': [0.76,	0.68,	0.89,	0.78,	0.54],
        '3-shot': [0.75,	0.72,	0.87,	0.9,	0.59],
        '5-shot': [0.75,	0.71,	0.82,	0.91,	0.54],
        # Similar data for LLM2 and LLM3 variants
    }
    datasets = ["Gerrit", "GitHub", "GooglePlay", "Jira", "StackOverflow"]

    df = pd.DataFrame(data, index=datasets)

    # Create the extended heatmap
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.4)  # Set the font size to 14px (1.4 * 10)
    heatmap = sns.heatmap(df, annot=True, cmap="YlGnBu", linewidths=.5, annot_kws={"size": 14})
    # plt.title("Performance of LLM Variants on Different Datasets", fontsize=14)  # Set title font size
    # plt.xlabel("LLM Variants", fontsize=14)  # Set x-axis label font size
    # plt.ylabel("Datasets", fontsize=14)  # Set y-axis label font size

    # Set color bar font size
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    plt.savefig("figures/wizardlm-heatmap.png", dpi=600, bbox_inches="tight")
    plt.show()
    
def draw_boxplot(data_dict, title="WizardLM"):
    import matplotlib.pyplot as plt
    import numpy as np

    # Sample data (replace with your actual data)

    # Create a list of LLM variants
    llm_variants = list(data_dict.keys())
    # llm_variants[0] = 'Vicuna\n(0-shot)'
    # Create a boxplot
    plt.figure(figsize=(6, 6))
    plt.boxplot([data_dict[variant] for variant in llm_variants], labels=llm_variants)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(title, fontsize=14)
    # plt.xlabel("LLM Variants")
    # plt.ylabel("Values")
    plt.savefig("figures/{}_boxplot.png".format(title.lower()), dpi=600, bbox_inches="tight")
    # Show the plot
    plt.show()

def draw_boxplot_rq2():
    import matplotlib.pyplot as plt
    import numpy as np

    # Sample data (replace with your actual data)

    # Create a list of LLM variants
    macro_dict = {
        "Gerrit": [0.74, 0.76, 0.75, 0.73, 0.75, 0.81, 0.74, 0.77],
        "GitHub": [0.72, 0.72, 0.68, 0.9, 0.92, 0.92, 0.94, 0.91],
        "GooglePlay": [0.98, 0.89, 1, 0.56, 0.49, 0.57, 0.42, 0.39],
        "Jira": [0.89, 0.91, 0.89, 0.97, 0.95, 0.95, 0.95, 0.94],
        "StackOverflow": [0.65, 0.59, 0.51, 0.64, 0.57, 0.6, 0.68, 0.67]
    }
    
    micro_dict = {
        "Gerrit": [0.82, 0.84, 0.83, 0.81, 0.8, 0.86, 0.81, 0.83],
        "GitHub": [0.72, 0.72, 0.68, 0.9, 0.92, 0.92, 0.94, 0.91],
        "GooglePlay": [0.97, 0.94, 1, 0.8, 0.71, 0.83, 0.63, 0.57],
        "Jira": [0.89, 0.92, 0.89, 0.97, 0.96, 0.96, 0.95, 0.95],
        "StackOverflow": [0.83, 0.74, 0.72, 0.84, 0.84, 0.84, 0.86, 0.89]
    }
    
    llm_variants = list(macro_dict.keys())
    plt.boxplot([macro_dict[variant] for variant in llm_variants], labels=llm_variants)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Macro-F1', fontsize=14)
    plt.savefig("figures/Macro_F1_boxplot.png", dpi=600, bbox_inches="tight")
    # Show the plot
    plt.show()
    plt.clf()
    llm_variants = list(micro_dict.keys())
    plt.boxplot([micro_dict[variant] for variant in llm_variants], labels=llm_variants)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Micro-F1', fontsize=14)
    plt.savefig("figures/Micro_F1_boxplot.png", dpi=600, bbox_inches="tight")
    
def draw_bar_graph(data1, data2, score_type="Macro-F1"):
    import matplotlib.pyplot as plt
    import numpy as np

    # Sample data for five datasets
    categories = ["Gerrit", "GitHub", "GooglePlay", "Jira", "StackOverflow"]
    
    # Set the width of the bars
    bar_width = 0.35

    # Generate x-positions for the bars
    x = np.arange(len(categories))

    # Create the bar plots
    plt.bar(x - bar_width/2, data1, bar_width, label='Best Zero-shot')
    plt.bar(x + bar_width/2, data2, bar_width, label='Best Few-shot')

    # Set the labels, title, and legend
    plt.xlabel('Dataset')
    plt.ylabel(score_type)
    # plt.title('Bar Graph with Two Bars and Five Datasets')
    plt.xticks(x, categories)
    plt.legend()
    plt.savefig("figures/{}-bar-graph.png".format(score_type.lower()), dpi=600, bbox_inches="tight")
    # Display the graph
    plt.show()

    
if __name__ == "__main__":
    # draw_line_figures()
    # plt.clf()
    # draw_venn_diagrams_zero()
    # plt.clf()
    # draw_venn_diagrams_few()
    # draw_heatmap()
    vicuna_data = {
        "Gerrit": [0.73, 0.73, 0.7, 0.74, 0.73, 0.71],
        "GitHub": [0.72, 0.65, 0.67, 0.68, 0.72, 0.72],
        "GooglePlay": [0.98, 0.74, 0.82, 0.74, 0.82, 0.77],
        "Jira": [0.85, 0.69, 0.75, 0.77, 0.86, 0.89],
        "StackOverflow": [0.59, 0.56, 0.53, 0.56, 0.65, 0.64]
    }
    
    wizard_data = {
        "Gerrit": [0.69, 0.69, 0.68, 0.76, 0.75, 0.75],
        "GitHub": [0.71, 0.7, 0.7, 0.68, 0.72, 0.71],
        "GooglePlay": [0.8, 0.82, 0.79, 0.89, 0.87, 0.82],
        "Jira": [0.81, 0.82, 0.77, 0.78, 0.9, 0.91],
        "StackOverflow": [0.41, 0.59, 0.52, 0.54, 0.59, 0.54]
    }
    
    llama2_data = {
        "Gerrit": [0.73, 0.71, 0.75, 0.69, 0.69, 0.68],
        "GitHub": [0.68, 0.64, 0.68, 0.54, 0.6, 0.61],
        "GooglePlay": [0.89, 0.89, 0.89, 0.89, 0.87, 1],
        "Jira": [0.83, 0.71, 0.78, 0.82, 0.84, 0.89],
        "StackOverflow": [0.45, 0.5, 0.51, 0.42, 0.46, 0.47]     
    }
    # draw_bar_graph()
    # draw_boxplot(vicuna_data, title="Vicuna")
    # plt.clf()
    # draw_boxplot(wizard_data, title="WizardLM")
    # plt.clf()
    # draw_boxplot(llama2_data, title="LLaMA 2-Chat")
    # plt.clf()
    
    macro_y1 = [0.75,	0.72,	0.98,	0.85,	0.59]
    macro_y2 = [0.76,	0.72,	1,	0.91,	0.65]
    
    ## micro-f1
    micro_y1 = [0.83,	0.72,	0.97,	0.86,	0.82]
    micro_y2 = [0.84,	0.72,	1,	0.92,	0.83]
    
    # draw_bar_graph(micro_y1, micro_y2, score_type="Micro-F1")
    # plt.clf()
    # draw_bar_graph(macro_y1, macro_y2, score_type="Macro-F1")
    # draw_line_figures(micro_y1, micro_y2, "Micro-F1")
    # plt.clf()
    # draw_line_figures(macro_y1, macro_y2, "Macro-F1")
    draw_boxplot_rq2()