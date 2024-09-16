import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 设置中文字体（例如：SimHei 字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


class ScikitLearnmol:
    def __init__(self, file_path, n_nian_fen, s_lie_ming, y_yuce_shuliang, sheet_na):
        """
         ScikitLearnmol学习模型 初始化类，定义必要的变量。

         :param file_path: Excel 文件路径
         :param n_nian_fen: 包含年份的列名
         :param s_lie_ming: 需要预测的列名列表
         :param y_yuce_shuliang: 预测的未来年份数量
         :param sheet_na: 预测结果保存的表名
         """
        self.file_path = file_path
        self.n_nian_fen = n_nian_fen
        self.s_lie_ming = s_lie_ming
        self.y_yuce_shuliang = y_yuce_shuliang
        self.sheet_na = sheet_na
        self.df = None  # 用于存储加载的Excel数据
        self.yuce_result = {}  # 用于存储每列的预测结果
        self.s_souyilv_list = []  # 用于存储各列的收益率
        self.yearly_return_dict = {}  # 用于存储每年收益率

    def load_data(self):
        """
        加载 Excel 文件并进行预处理。设置年份为索引，以方便后续操作。
        """

        self.df = pd.read_excel(self.file_path)
        self.df['Year'] = pd.to_datetime(self.df[self.n_nian_fen], format='%Y')
        self.df = self.df.set_index('Year')

    def scikit_mol(self, x, y):
        """
        使用 Scikit-learn 的线性回归模型对数据进行拟合。
        :param x: 输入特征（年份）
        :param y: 输出特征（销售额或其他预测值）
        :return: 拟合好的模型
        """
        model = LinearRegression()  # 创建线性回归模型
        model.fit(x, y)  # 拟合模型
        return model

    def predict_sales(self):
        """
        对每列数据进行线性回归预测，并计算未来若干年的收益率。
        """

        for column in self.s_lie_ming:
            # 提取该列的数据，并转换为二维数组以用于训练
            tiqu_data = self.df[column].values.reshape(-1, 1)
            # 提取年份信息作为特征数据x值
            years = np.array([year.year for year in self.df.index]).reshape(-1, 1)
            # 训练模型
            model = self.scikit_mol(years, tiqu_data)
            # 预测未来年份对应的y值
            yuce_weilai = np.arange(self.df.index[-1].year + 1,
                                    self.df.index[-1].year + self.y_yuce_shuliang + 1).reshape(-1, 1)
            forecast = model.predict(yuce_weilai)
            # 存储预测结果
            self.yuce_result[column] = forecast.flatten()
            # 计算预测的年收益率并存储
            yearly_return = self.calculate_yearly_return(forecast.flatten())
            self.yearly_return_dict[column] = yearly_return
            # 计算平均年收益率
            avg_shouyilv = np.mean(yearly_return)
            self.s_souyilv_list.append(avg_shouyilv)
            # 打印每列的预测结果和平均年收益率
            print(f"列 {column} 的预测结果: {self.yuce_result[column]}")
            print(f"列 {column} 的平均年收益率: {avg_shouyilv:.4f}")

    # 计算每年收益率
    def calculate_yearly_return(self, forecast):
        """
        根据预测结果计算每年的收益率。

        :param forecast: 预测的未来几年数值
        :return: 每年收益率列表
        """

        return [(forecast[i + 1] - forecast[i]) / forecast[i] for i in range(len(forecast) - 1)]

    # 输出每个作物每年的收益率
    def output_yearly_return(self):
        """
        打印每列数据每年的收益率。
        """

        print("\n每个作物每年的收益率:")
        for column, yearly_return in self.yearly_return_dict.items():
            print(f"\n{column} 每年的收益率:")
            for i, return_rate in enumerate(yearly_return, start=1):
                print(f"第 {i} 年: {return_rate:.4f}")

    def save_forecast(self):
        """
        将预测结果保存到 Excel 文件的新表中。
        """

        yuce_weilai = pd.date_range(start=self.df.index[-1] + pd.DateOffset(years=1), periods=self.y_yuce_shuliang,
                                    freq='YE')
        forecast_df = pd.DataFrame(index=yuce_weilai)  # 创建存储预测结果的DataFrame

        for column in self.s_lie_ming:
            forecast_df[column] = self.yuce_result[column]

        with pd.ExcelWriter(self.file_path, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
            forecast_df.to_excel(writer, sheet_name=self.sheet_na)

    def visualize_forecast(self):
        """
        将历史数据和预测数据可视化。
        """

        plt.figure(figsize=(10, 6))  # 设置图形大小
        for column in self.s_lie_ming:
            plt.plot(self.df.index.year, self.df[column], label=f'{column} 历史')
            future_year_range = np.arange(self.df.index.year[-1] + 1, self.df.index.year[-1] + self.y_yuce_shuliang + 1)
            plt.plot(future_year_range, self.yuce_result[column], label=f'{column} 预测', linestyle='--')

        plt.title('未来预测')
        plt.xlabel('年份')
        plt.ylabel(self.sheet_na)
        plt.legend()
        plt.grid(True)
        plt.show()

    def calculate_avg_shouyilv(self):
        """
        计算所有列的平均年收益率。
        :return: 所有列的平均收益率
        """

        avg_shouyilv = np.mean(self.s_souyilv_list)
        print(f"\n所有列的平均收益率: {avg_shouyilv:.2%}")
        return avg_shouyilv

    def run_forecast(self):
        """
        运行整个预测流程，包括数据加载、预测、保存、可视化和输出收益率。
        """

        self.load_data()
        self.predict_sales()
        self.calculate_avg_shouyilv()
        self.save_forecast()
        self.visualize_forecast()
        self.output_yearly_return()  # 输出每年收益率


# 预测十年销量主函数
def main_1():
    # 设置文件路径和相关参数
    file_path = r'C:\Users\DoDO\Desktop\F.xlsx'
    n_nian_fen = '年份'  # 日期列名
    s_lie_ming = ['总销量', '小麦销售量', '稻谷销售量', '玉米销售量', '大豆销售量', '其他粮食销售量']  # 销售数据列
    y_yuce_shuliang = 13  # 预测的年份数
    # 创建预测类的实例并运行预测流程
    forecaster = ScikitLearnmol(file_path, n_nian_fen, s_lie_ming, y_yuce_shuliang, sheet_na='销量')
    forecaster.run_forecast()


# 预测十年成本主函数
def main_2():
    # 设置文件路径和相关参数
    file_path = r'C:\Users\DoDO\Desktop\H.xlsx'
    n_nian_fen = '年份'  # 日期列名
    s_lie_ming = ['稻谷成本', '小麦成本', '玉米成本']  # 成本数据列名
    y_yuce_shuliang = 13  # 预测的年份数
    # 创建预测类的实例并运行预测流程
    forecaster = ScikitLearnmol(file_path, n_nian_fen, s_lie_ming, y_yuce_shuliang, sheet_na='成本')
    forecaster.run_forecast()


# 预测十年单产主函数
def main_3():
    # 设置文件路径和相关参数
    file_path = r'C:\Users\DoDO\Desktop\G.xlsx'
    n_nian_fen = '年份'  # 日期列名
    s_lie_ming = ['稻谷单位产量', '小麦单位产量', '玉米单位产量', '大豆单位产量']  # 单产数据列名
    y_yuce_shuliang = 13  # 预测的年份数
    # 创建预测类的实例并运行预测流程
    forecaster = ScikitLearnmol(file_path, n_nian_fen, s_lie_ming, y_yuce_shuliang, sheet_na='单位产量')
    forecaster.run_forecast()


if __name__ == "__main__":
    # main_1()
    # main_2()
    main_3()

