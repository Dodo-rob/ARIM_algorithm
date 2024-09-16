import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 设置中文字体（例如：SimHei 字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def analyze_data(file_path, sheet_name, columns):
    """
       读取Excel文件中的指定列，计算这些列之间的Pearson相关系数，分析可替代性和互补性，
       并绘制相关性矩阵的热力图。

        :param  file_path: Excel文件的路径。
        :param sheet_name: 要读取的Excel工作表名称。
        :param columns: 需要分析的列名列表。
       """
    # 1. 读取Excel文件
    data = pd.read_excel(file_path, sheet_name=sheet_name)

    # 2. 选择需要的列
    df = data[columns]

    # 3. 计算相关性矩阵 (Pearson 相关系数)
    correlation_matrix = df.corr(method='pearson')

    # 4. 定义函数分析可替代性和互补性
    def analyze_complementarity(corr_matrix):
        print("可替代性和互补性分析:")
        for i in range(corr_matrix.shape[0]):
            for j in range(i + 1, corr_matrix.shape[1]):
                corr_value = corr_matrix.iloc[i, j]
                if corr_value > 0.8:
                    print(
                        f"变量 {corr_matrix.columns[i]} 和 变量 {corr_matrix.columns[j]} 具有高度可替代性，相关系数: {corr_value:.2f}")
                elif corr_value < -0.8:
                    print(
                        f"变量 {corr_matrix.columns[i]} 和 变量 {corr_matrix.columns[j]} 具有高度互补性，相关系数: {corr_value:.2f}")

    # 5. 调用分析函数
    analyze_complementarity(correlation_matrix)

    # 6. 绘制相关性矩阵的热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Pearson 相关性分析")
    plt.show()


def main_1():
    file_path = r'C:\Users\DoDO\Desktop\综合数据.xlsx'
    sheet_name = 'Sheet2'
    columns = ['蔬菜产量', '稻谷产量', '署类产量', '小麦产量', '豆类产量', '玉米产量']
    analyze_data(file_path, sheet_name, columns)


def main_2():
    file_path = r'C:\Users\DoDO\Desktop\综合数据.xlsx'
    sheet_name = 'Sheet3'
    columns = [
        '马铃薯单产', '薯类单产', '红小豆单产', '绿豆单产',
        '大豆单产', '高粱单产', '谷子单产', '稻谷单产',
        '谷物单产', '秋粮单产', '夏收粮食单产'
    ]
    analyze_data(file_path, sheet_name, columns)


if __name__ == "__main__":
    main_2()
    main_1()