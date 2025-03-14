import matplotlib.pyplot as plt

def plot(data):

    # 创建 x 轴数据（索引）
    x = range(len(data))

    # 绘制折线图
    plt.plot(x, data, marker='o', linestyle='-', color='b', label='Data Points')

    # 添加标题和标签
    plt.title('List Data Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # 添加网格
    plt.grid(True)

    # 显示图例
    plt.legend()

    # 保存图形为图片文件
    plt.savefig('list_plot.png', dpi=300, bbox_inches='tight')  # 保存为 PNG 文件
    print("图形已保存为 'list_plot.png'")