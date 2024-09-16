import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


class Arima_mol:
    def __init__(self, file_path, sheet_na, y_data, x_data):
        """
        :param file_path: Excel 文件路径
        :param sheet_na: 读取数据的工作表名称
        :param y_data: y轴(作为自变量)
        :param x_data: x轴(作为自变量)
         ARIMA 模型的参数 (p, d, q)
        """
        self.file_path = file_path
        self.sheet_na = sheet_na
        self.y_data = y_data
        self.x_data = x_data
        self.data = pd.read_excel(file_path, sheet_name=sheet_na)
        self.years = self.data[y_data].values
        self.mushroom_price = self.data[x_data].values
        self.scaler = MinMaxScaler()

    # 平滑和归一化处理
    def preprocess_data(self):
        # 将因变量换为 pandas Series(是fillna要求的类型)
        y = pd.Series(self.mushroom_price)
        # 对因变量进行平滑处理
        smoothed_price = y.rolling(window=1).mean().fillna(y)
        # 归一化处理
        self.guiyi = self.scaler.fit_transform(smoothed_price.values.reshape(-1, 1)).flatten()

    # 训练 ARIMA 模型。
    def train_model(self, order=(6, 1, 2)):
        self.model = ARIMA(self.guiyi, order=order)
        self.arima_result = self.model.fit()
        self.fitted_values_scaled = self.arima_result.fittedvalues
        self.fitted_values = self.scaler.inverse_transform(self.fitted_values_scaled.reshape(-1, 1)).flatten()

    # ARIMA 模型进行未来预测
    def forecast(self, n_years=7):
        """
        :param n_years: 预测的未来年份数
         return: 预测的未来值
        """
        forecast_scaled = self.arima_result.forecast(steps=n_years)
        self.forecast_original_scale = self.scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
        return self.forecast_original_scale

    # 评估模型性能，计算 MSE, RMSE 和 MAE
    def evaluate_model(self):
        mse = mean_squared_error(self.mushroom_price, self.fitted_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.mushroom_price, self.fitted_values)
        return mse, rmse, mae

    # 计算预测值的平均年收益率
    def calculate_annual_returns(self, yuce_zhi):
        """
        :param yuce_zhi: 预测值
        :return: 平均年收益率
        """
        annual_returns = [(yuce_zhi[i + 1] - yuce_zhi[i]) / yuce_zhi[i] for i in range(len(yuce_zhi) - 1)]
        return np.mean(annual_returns)

    # 绘制原始数据、平滑数据和拟合值的图表
    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.years, self.mushroom_price, label='原始蘑菇价格', marker='o', linestyle='-', color='blue')
        plt.plot(self.years, self.fitted_values, label='拟合值', linestyle='-', color='red')
        plt.xlabel('年份')
        plt.ylabel('蘑菇价格')
        plt.title('蘑菇价格的原始数据与 ARIMA 模型拟合结果')
        plt.legend()
        plt.grid(True)
        plt.show()

    # 保存到 Excel 文件的新的工作表
    def save_forecast(self, forecast, n_years=7):
        """
        :param forecast: 预测值
        :param n_years: 预测的未来年份数
        """
        future_years = np.arange(2024, 2024 + n_years)
        forecast_df = pd.DataFrame({
            '年份': future_years,
            '预测蘑菇价格': forecast
        })

        with pd.ExcelWriter(self.file_path, mode='a', if_sheet_exists='replace') as writer:
            forecast_df.to_excel(writer, sheet_name='预测结果', index=False)

    # 运行模型程序
    def run(self, n_years=7):
        # 归一
        self.preprocess_data()
        # 训练 ARIMA 模型
        self.train_model()
        # 进行未来预测
        forecast = self.forecast(n_years=n_years)
        # 模型评估
        mse, rmse, mae = self.evaluate_model()
        print(f"模型评估: MSE = {mse:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
        # 计算年收益率
        average_return = self.calculate_annual_returns(forecast)
        print(f"预测结果的平均年收益率为: {average_return:.2%}")
        # 绘制拟合结果图表
        self.plot_results()
        # 保存预测结果到 Excel 文件
        self.save_forecast(forecast, n_years)


def main():
    junlei_forecast = Arima_mol(file_path='C:/Users/DoDO/Desktop/N.xlsx', sheet_na='Sheet1', y_data='年份',
                                x_data='菌类平均价格')
    junlei_forecast.run(n_years=7)


if __name__ == "__main__":
    main()