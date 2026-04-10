# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
# # 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 表3数据：微调前后对比
# table3_data = {
#     'Model': ['Qwen3-0.6B-v1', 'Qwen3-0.6B-v2'],
#     'Accuracy': [7.6, 40.8],
#     'Precision': [37.8, 64.2],
#     'Recall': [35.5, 61.8],
#     'F1': [34.9, 61.8]
# }
#
# df3 = pd.DataFrame(table3_data)
#
# # 创建两个子图
# fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
#
# # 指标和颜色设置
# metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
# x = np.arange(len(metrics))
#
# # 设置更细的柱子宽度
# width = 0.4
#
# # 更柔和的冷色调
# light_cool_colors = ['#B3E0FF', '#A6D8E7', '#B5EAD7', '#C7CEEA']
#
# # 第一个子图：微调前(v1)
# ax1 = axes[0]
#
# bars1 = []
# for i, metric in enumerate(metrics):
#     bar = ax1.bar(i, df3.iloc[0, i+1], width, label=metric, color=light_cool_colors[i],
#                   alpha=0.85, edgecolor='black', linewidth=0.8)
#     bars1.append(bar[0])
#
# # 设置第一个子图的标题和标签
# ax1.set_xlabel('Qwen3-0.6B-v1', fontsize=16, labelpad=10)  # 字体从12调大到16
# ax1.set_ylabel('Percentage (%)', fontsize=16, labelpad=10)  # 字体从12调大到16
# ax1.set_xticks(x)
# ax1.set_xticklabels(metrics, fontsize=15, rotation=0)  # 字体从11调大到15
# ax1.set_ylim(0, 70)
# ax1.tick_params(axis='y', labelsize=15)  # 字体从11调大到15
# ax1.grid(True, alpha=0.2, linestyle='--', axis='y')
#
# # 添加数值标签
# for i, bar in enumerate(bars1):
#     height = bar.get_height()
#     ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
#              f'{height:.1f}%', ha='center', va='bottom',
#              fontweight='bold', fontsize=13, color='black')  # 字体从10调大到13
#
# # 第二个子图：微调后(v2)
# ax2 = axes[1]
#
# bars2 = []
# for i, metric in enumerate(metrics):
#     bar = ax2.bar(i, df3.iloc[1, i+1], width, label=metric, color=light_cool_colors[i],
#                   alpha=0.85, edgecolor='black', linewidth=0.8)
#     bars2.append(bar[0])
#
# # 设置第二个子图的标题和标签
# ax2.set_xlabel('Qwen3-0.6B-v2', fontsize=16, labelpad=10)  # 字体从12调大到16
# ax2.set_xticks(x)
# ax2.set_xticklabels(metrics, fontsize=15, rotation=0)  # 字体从11调大到15
# ax2.set_ylim(0, 70)
# ax2.tick_params(axis='y', labelsize=15)  # 字体从11调大到15
# ax2.grid(True, alpha=0.2, linestyle='--', axis='y')
#
# # 添加数值标签
# for i, bar in enumerate(bars2):
#     height = bar.get_height()
#     ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
#              f'{height:.1f}%', ha='center', va='bottom',
#              fontweight='bold', fontsize=13, color='black')  # 字体从10调大到13
#
# # 添加图例（只添加一次，避免重复）
# handles = []
# for i, (metric, color) in enumerate(zip(metrics, light_cool_colors)):
#     handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=metric, alpha=0.85))
#
# # 调整子图间距
# plt.tight_layout(rect=[0, 0, 0.85, 1])
#
# # 导出图片
# output_path = '微调效果对比_英文.pdf'
# plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
# print(f"图片已保存到: {output_path}")
#
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
# # 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 表4数据
# table4_data = {
#     'Model': ['Qwen3-0.6B-n1', 'Qwen3-0.6B-n1', 'Qwen3-0.6B-n2', 'Qwen3-0.6B-n2'],
#     'Validation_Set': ['annotated', 'No annotations', 'annotated', 'No annotations'],
#     'Accuracy': [75.8, 31.9, 77.8, 66.6],
#     'Precision': [90.1, 74.2, 90.2, 87.0],
#     'Recall': [90.4, 68.2, 89.0, 84.9],
#     'F1': [90.4, 68.2, 89.0, 84.9]
# }
#
# df4 = pd.DataFrame(table4_data)
#
# # 创建分组柱状图
# fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# fig.suptitle('', fontsize=16, fontweight='bold')
#
# # 更柔和的冷色调
# light_cool_colors = ['#B3E0FF', '#A6D8E7', '#B5EAD7', '#C7CEEA']
#
# # 每种情况的颜色分配
# color_mapping = {
#     'n1-验证集一': light_cool_colors[0],
#     'n1-验证集二': light_cool_colors[1],
#     'n2-验证集一': light_cool_colors[2],
#     'n2-验证集二': light_cool_colors[3]
# }
#
# metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
# metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
# labels = ['n1-Validation Set 1', 'n1-Validation Set 2', 'n2-Validation Set 1', 'n2-Validation Set 2']
# x = np.arange(len(labels))
# width = 0.4
#
# # 绘制四个指标的分组图
# for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
#     ax = axes[idx // 2, idx % 2]
#     values = df4[metric].values
#
#     # 每种情况使用相同的颜色
#     bar_colors = [light_cool_colors[0], light_cool_colors[1], light_cool_colors[2], light_cool_colors[3]]
#
#     bars = ax.bar(x, values, width, color=bar_colors, edgecolor='white', linewidth=1.5)
#     ax.set_title(f'{metric_name} Comparison', fontsize=18, fontweight='bold')  # 从12调大到18
#     ax.set_ylabel('Score (%)', fontsize=16)  # 从10调大到16
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=13)  # 从9调大到13
#     ax.set_ylim(0, 100)
#     ax.grid(True, alpha=0.2, linestyle='--', axis='y')
#
#     # 设置背景色
#     ax.set_facecolor('#f8f9fa')
#
#     # 在柱子上添加数值
#     for bar, value in zip(bars, values):
#         height = bar.get_height()
#         # 调整文字颜色，确保在浅色柱子上可见
#         text_color = '#2C3E50'  # 深灰色文字
#         ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
#                 f'{value:.1f}', ha='center', va='bottom',
#                 fontsize=13, fontweight='bold', color=text_color)  # 从9调大到13
#
#     # 只在左上角子图添加图例
#     # if idx == 0:
#     #     from matplotlib.patches import Patch
#     #
#     #     legend_elements = [
#     #         Patch(facecolor=light_cool_colors[0], edgecolor='white', label='n1-有注释'),
#     #         Patch(facecolor=light_cool_colors[1], edgecolor='white', label='n1-无注释'),
#     #         Patch(facecolor=light_cool_colors[2], edgecolor='white', label='n2-有注释'),
#     #         Patch(facecolor=light_cool_colors[3], edgecolor='white', label='n2-无注释')
#     #     ]
#     #     ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
#
# plt.tight_layout()
# plt.savefig('注释信息影响_英文.pdf', dpi=300, bbox_inches='tight',
#             facecolor='white', edgecolor='none')
# plt.show()



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 表5数据：轻量级vs大模型（包含所有5个模型）
table5_data = {
    'Model': ['Qwen3-0.6B-v1', 'Qwen3-0.6B-v2', 'DeepSeek-chat', 'Qwen3-max', 'GLM4.7','llama-3.3-70b-instruct'],
    'Accuracy': [11.7, 43.9, 44.2, 39.0, 53.3, 30.4],
    'Precision': [42.6, 65.4, 66.8, 62.4, 72.7, 56.7],
    'Recall': [41.4, 64.9, 69.4, 71.1, 71.7, 66.3],
    'F1': [40.2, 64.1, 66.7, 64.7, 71.2, 59.4]
}

df5 = pd.DataFrame(table5_data)

# 创建雷达图
fig, ax = plt.subplots(figsize=(12, 9), subplot_kw=dict(projection='polar'))

# 准备雷达图数据
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1']

# 准备角度数据
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # 闭合雷达图

# 五个模型的数据
models_data = {
    'Qwen3-0.6B-v1': df5[df5['Model'] == 'Qwen3-0.6B-v1'].iloc[0, 1:].tolist(),
    'Qwen3-0.6B-v2': df5[df5['Model'] == 'Qwen3-0.6B-v2'].iloc[0, 1:].tolist(),
    'DeepSeek-chat': df5[df5['Model'] == 'DeepSeek-chat'].iloc[0, 1:].tolist(),
    'Qwen3-max': df5[df5['Model'] == 'Qwen3-max'].iloc[0, 1:].tolist(),
    'GLM4.7': df5[df5['Model'] == 'GLM4.7'].iloc[0, 1:].tolist(),
    'llama-3.3-70b-instruct': df5[df5['Model'] == 'llama-3.3-70b-instruct'].iloc[0, 1:].tolist()
}

# 颜色配置
colors = ['#95a5a6', '#3498db', '#e74c3c', '#9b59b6', '#2ecc71','#ffd518']
labels = list(models_data.keys())

# 绘制雷达图线
for i, (model_name, values) in enumerate(models_data.items()):
    values += values[:1]  # 闭合数据
    ax.plot(angles, values, 'o-', linewidth=2.5, label=labels[i],
            color=colors[i], markersize=6, markeredgewidth=1.5)
    ax.fill(angles, values, alpha=0.15, color=colors[i])

# 设置雷达图属性
ax.set_xticks(angles[:-1])
ax.set_ylim(0, 80)
ax.set_yticks([0, 20, 40, 60, 80])
ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%'], fontsize=16)  # 从10调大到16

# 设置X轴标签位置
label_radius = 85
for i, (angle, label) in enumerate(zip(angles[:-1], metric_labels)):
    if angle < np.pi / 4 or angle > 7 * np.pi / 4:
        ha = 'left'
    elif np.pi / 4 <= angle <= 3 * np.pi / 4:
        ha = 'center'
    elif 3 * np.pi / 4 < angle < 5 * np.pi / 4:
        ha = 'right'
    else:
        ha = 'center'

    ax.text(angle, label_radius, label,
            ha=ha, va='center', fontsize=18, fontweight='bold',  # 从12调大到18
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9,
                      edgecolor='none'))

# 设置X轴刻度标签为空
ax.set_xticklabels([''] * len(metrics))

# 增强网格线
ax.grid(True, alpha=0.6, linewidth=1.2)

# 添加同心圆环
for r in [20, 40, 60, 80]:
    ax.plot(angles, [r] * len(angles), color='#999999', alpha=0.3,
            linewidth=0.8, linestyle='--')

# 设置图例位置（调整以适应更多模型）
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=15,  # 从11调大到15
          frameon=True, framealpha=0.9, edgecolor='black')

# 调整布局
plt.tight_layout()

# 导出图片
plt.savefig('RQ3_雷达图_英文.pdf', dpi=300, bbox_inches='tight', facecolor='white')

plt.show()