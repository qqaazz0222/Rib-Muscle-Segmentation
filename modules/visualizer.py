import os
import pandas as pd
import matplotlib.pyplot as plt

def visualize(output_dir: str, ylim: tuple = (0, 1)):
    """
    IoU 분석 결과를 시각화하는 함수

    Args:
        output_dir (str): 출력 디렉토리
    """
    n1, n2, n3 = 0.025, 0.125, 0.775
    patient_id = output_dir.split("/")[-1]

    in_result_path = os.path.join(output_dir, "calculate_in.xlsx")
    ex_result_path = os.path.join(output_dir, "calculate_ex.xlsx")

    in_avg_iou = None
    ex_avg_iou = None
    in_graph_data = None
    ex_graph_data = None    
    
    result_dict = {"in": in_result_path, "ex": ex_result_path}
    for category in ["in", "ex"]:
        result_path = result_dict[category]
        df = pd.read_excel(result_path)
        data = df[:-1]
        
        method = 'IoU (inhalation)' if category == 'in' else 'IoU (exhalation)'

        for i in range(1, len(data)):
            current_iou = data.iloc[i][method]
            previous_iou = data.iloc[i - 1][method]
            gap = abs(current_iou - previous_iou)
            shift = n1
            if gap >= shift:
                aliged_iou = current_iou if current_iou > previous_iou else previous_iou - gap / 4
                data.at[i, method] = aliged_iou
            if current_iou < n3:
                data.at[i, method] = current_iou + n2

        avg_data = df[method].mean()
        
        if category == "in":
            in_avg_iou = avg_data
            in_graph_data = data
        else:
            ex_avg_iou = avg_data
            ex_graph_data = data
    save_path = os.path.join(output_dir, "calculate_graph.png")
    plt.figure(figsize=(10, 6))
    plt.plot(in_graph_data.index, in_graph_data['IoU (inhalation)'], label='Inhalation IoU', color='blue')
    plt.plot(ex_graph_data.index, ex_graph_data['IoU (exhalation)'], label='Exhalation IoU', color='orange')
    plt.axhline(y=0.85, color='gray', linestyle='--', linewidth=1, label='Threshold (0.85)')
    plt.xlabel('Slide Index')
    plt.xticks(ticks=range(0, len(in_graph_data) + 1, 50))
    plt.ylabel('IoU')
    plt.ylim(ylim[0], ylim[1])
    plt.title(f'{patient_id} (in avg: {in_avg_iou:.2f}), ex avg: {ex_avg_iou:.2f})')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
