code intriduction: <br> 
1.ARIM主要适用于时序预测，预测蘑菇价格

************ARIM_algorithm.py************ <br>

class Arimamol: 主要的运行文件 <br> 
def __init__:<br>
        :param file_path: Excel 文件路径<br>
        :param sheet_na: 读取数据的工作表名称<br>
        :param y_data: y轴(作为自变量)<br>
        :param x_data: x轴(作为自变量)<br>
         ARIMA 模型的参数 (p, d, q)<br>
def preprocess_data: 平滑和归一化处理<br>

def train_model: 训练 ARIMA 模型<br>

def forecast: ARIMA 模型进行未来预测<br>

def evaluate_model: 评估模型性能，计算 MSE, RMSE 和 MAE<br>

def calculate_annual_returns: 计算预测值的平均年收益率<br>
        :param yuce_zhi: 预测值<br>
        :return: 平均年收益率<br>
        
ddef plot_results: 制原始数据、平滑数据和拟合值的图表<br>

def save_forecast: 保存到 Excel 文件的新的工作表<br>

def run : 运行模型程序<br>

********************N.xlsx********************<br>
    试运行文件
