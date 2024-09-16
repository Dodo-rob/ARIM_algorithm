import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm

import Excel_deal as excel


class GeneticAlgorithm:
    def __init__(self, costs, prices, yields, area,
                 area_mode, crop_type, year, population_size,
                 number_iterations, probability_variation,
                 choose_min = 1,choose_max = 3):
        """
        :param costs: 作物种植成本的列表
        :param prices: 各年作物价格的列表
        :param yields: 各年作物亩产量的列表
        :param area: 各地块面积的列表
        :param area_mode: 地块数量
        :param crop_type: 作物数量
        :param year: 年份数量
        :param population_size: 种群大小
        :param number_iterations: 迭代次数
        :param probability_variation: 变异概率
        :param choose_min: 选择最少作物
        :param choose_max: 选择最多作物
        """
        self.costs = costs  # 种植成本
        self.prices = prices  # 不同年份的价格
        self.yields = yields  # 不同年份的亩产量
        self.area = area  # 地块面积
        self.area_mode = area_mode  # 地块数量
        self.crop_type = crop_type  # 作物数量
        self.year = year  # 年份数 7年 (2024-2030)
        self.population_size = population_size  # 种群大小
        self.number_iterations = number_iterations  # 迭代次数
        self.probability_variation = probability_variation  # 变异概率
        self.choose_min = choose_min  # 选择最少作物
        self.choose_max = choose_max  # 选择最多作物

    def initialize_population(self, pop_size):
        """
        初始化种群结构

        :param pop_size: 种群大小
        :return: 初始化后的种群
        """

        population = []
        for _ in range(pop_size):
            individual = np.zeros((self.area_mode, self.year, self.crop_type), dtype=int)
            for i in range(self.area_mode):
                for k in range(self.year):
                    crop_type_chosen = random.randint(self.choose_min, self.choose_max)  # 每年随机选择min到max# 种作物
                    crops_chosen = random.sample(range(self.crop_type), crop_type_chosen)
                    for crop in crops_chosen:
                        individual[i, k, crop] = 1  # 确保至少有一种作物
            population.append(individual)
        return population

    def calculate_penalty(self, individual, front_plots , middle_plots, end_plots):
        """
        计算正态分布惩罚函数

        :param individual: 个体解决方案
        :param front_plots: 分块1
        :param middle_plots: 分块2
        :param end_plots: 分块3 
        :return: penalty 惩罚值
        """
        penalty = 0
        expected_distribution = np.concatenate([
            norm.pdf(np.linspace(-2, -1, front_plots)),
            norm.pdf(np.linspace(-1, 1, middle_plots)),
            norm.pdf(np.linspace(1, 2, end_plots))
        ])
        expected_distribution /= np.sum(expected_distribution)  # 标准化

        # 计算每个作物的分布惩罚
        for crop in range(self.crop_type):
            crop_distribution = np.sum(individual[:, :, crop], axis=1)
            error = np.sum((crop_distribution - expected_distribution) ** 2)
            penalty += error

        return penalty

    def at_least_one_crop(self, individual):
        """
        每块地至少种一种作物的惩罚函数

        :param individual: 个体解决方案
        :return: penalty 惩罚值
        """
        penalty = 0
        for i in range(self.area_mode):
            for k in range(self.year):
                if np.sum(individual[i, k, :]) == 0:  # 如果没有种作物
                    penalty += 500  # 惩罚数值
        return penalty

    def up_to_n_crops(self, individual, up_to_n_crops):
        """
        每块地每年最多种三种作物的惩罚函数

        :param individual: 个体解决方案
        :param up_to_n_crops : 最多多少作物
        :return: penalty 惩罚值
        """
        penalty = 0
        for i in range(self.area_mode):
            for k in range(self.year):
                if np.sum(individual[i, k, :]) > up_to_n_crops:  # 如果种植作物数量超过三种
                    penalty += 500
        return penalty

    def fitness(self, individual, a, b):
        """
        计算个体的适应度：总利润减去惩罚

        :param individual: 个体解决方案
        :param a: 销量占产量系数初始化
        :param b: 增收因子
        :return: 适应度值
        """
        total_profit = 0
        penalties = 0
        # 遍历所有地块和年份计算利润
        for i in range(self.area_mode):
            for k in range(self.year):
                crops_chosen = np.where(individual[i, k, :] == 1)[0]
                if len(crops_chosen) > 0:
                    mianji = self.area[i]
                    mianji_per_crop = mianji / len(crops_chosen)
                    for crop in crops_chosen:
                        price = self.prices[k][crop]
                        yield_per_mu = self.yields[k][crop]
                        cost = self.costs[crop]
                        profit = (a * mianji_per_crop * yield_per_mu * price) - (mianji_per_crop * cost)
                        total_profit += profit
            a = min(1, a + b)  # 增加收益因子，最高为1

        # 计算惩罚
        penalties += self.calculate_penalty(individual, front_plots = 6, middle_plots = 14 , end_plots = 6)
        penalties += self.at_least_one_crop(individual)
        penalties += self.up_to_n_crops(individual, up_to_n_crops = 3)

        return total_profit - penalties

    def selection(self, population):
        """
        选择适应度最高的个体

        :param population: 种群
        :return: 选择后的种群
        """
        fitness_values = [self.fitness(individual, a = 0.9, b = 0.0375) for individual in population]
        selected_indices = np.argsort(fitness_values)[-self.population_size:]  # 选择适应度最高的个体
        return [population[i] for i in selected_indices]

    def crossover(self, parent1, parent2):
        """
        交叉操作，生成两个子代个体

        :param parent1: 父代1
        :param parent2: 父代2
        :return: 子代1, 子代2
        """
        point = random.randint(0, self.area_mode - 1)  # 随机选择交叉点
        child1 = np.copy(parent1)
        child2 = np.copy(parent2)
        child1[point:] = parent2[point:]  # 交换交叉点之后的部分
        child2[point:] = parent1[point:]
        return child1, child2

    def mutate(self, individual):
        """
        变异操作：随机改变个体

        :param individual: 个体解决方案
        :return: 变异后的个体
        """
        if random.random() < self.probability_variation:  # 根据变异概率决定是否变异
            i = random.randint(0, self.area_mode - 1)
            k = random.randint(0, self.year - 1)
            crop_type_chosen = random.randint(1, 3)  # 随机选择1到3种作物
            individual[i, k, :] = 0
            crops_chosen = random.sample(range(self.crop_type), crop_type_chosen)
            for crop in crops_chosen:
                individual[i, k, crop] = 1
        return individual

    def run_genetic_algorithm(self):
        """
        运行遗传算法并返回最佳个体

        :return: 最佳个体
        """
        population = self.initialize_population(self.population_size)  # 初始化种群
        for generation in range(self.number_iterations):  # 迭代次数
            selected_population = self.selection(population)  # 选择适应度最高的个体
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected_population, 2)  # 随机选择两个父代
                child1, child2 = self.crossover(parent1, parent2)  # 交叉生成子代
                new_population.append(self.mutate(child1))  # 变异操作
                new_population.append(self.mutate(child2))
            population = new_population  # 更新种群
        best_individual = max(population, key=lambda individual: self.fitness(individual, a=0.9, b=0.0375))
        return best_individual

def heatmap(solution, year_index, area_mode, crop_type):
    """
        绘制某一年作物分布的热力图。

        :param solution: 三维数组，表示每个地块在每年种植的作物分布情况。
        维度为 (地块数量, 年份数量, 作物类型数量)，
        数组中的值为 0 或 1，表示该地块在某年是否种植某种作物。
        :param year_index: 指定需要查看哪一年的作物分布，年索引从 0 开始。
        :param area_mode: 地块数量，即 solution 的第一个维度的大小。
        :param crop_type: 作物种类数量，即 solution 的第三个维度的大小。
    """
    crop_distribution = np.zeros((area_mode, crop_type))
    # 填充 crop_distribution 数据
    for plot in range(area_mode):
        for crop in range(crop_type):
            if np.any(solution[plot, year_index, crop] == 1):
                crop_distribution[plot, crop] = 1
    # 绘制热图
    plt.figure(figsize=(12, 8))
    sns.heatmap(crop_distribution, annot=True, cmap="viridis", cbar=False,
                xticklabels=[f'Crop {i + 1}' for i in range(crop_type)],
                yticklabels=[f'Plot {i + 1}' for i in range(area_mode)])
    plt.title(f'Crop Distribution in Year {2024 + year_index}')
    plt.xlabel('Crops')
    plt.ylabel('Plots')
    plt.show()

def calculate_std_distribution(optimized_individual, area_mode, crop_type):
    """
    计算优化后个体中每个作物在不同区域（前6块、中间14块、后6块）的种植分布占比，
    并返回这些区域的标准差和占比结果。

    :param optimized_individual: 三维数组，表示每个地块的作物分布情况。
                                维度为 (地块数量, 年份数量, 作物类型数量)，
                                数组中的值为 0 或 1，表示该地块是否种植某种作物。
    :param area_mode : 地块数量。
    :param crop_type : 作物种类数量。
    :return std_results : 每个作物在三个区域（前6块、中间14块、后6块）的占比和标准差。
    :return front_6_zhanbi : 每个作物在前6块地的占比。
    :return middle_14_zhanbi: 每个作物在中间14块地的占比。
    :return last_6_zhanbi: 每个作物在后6块地的占比。
    """
    # 保存每个作物的标准差结果
    std_results = []
    front_6_zhanbi = np.zeros(16)
    middle_14_zhanbi = np.zeros(16)
    last_6_zhanbi = np.zeros(16)
    for crop in range(crop_type):
        # 提取当前作物的种植分布
        crop_distribution = np.zeros_like(optimized_individual[:, 0, crop])
        # 将 optimized_individual[:, 0, crop] 中的非零值变为 1，并赋值给 crop_distribution
        crop_distribution[optimized_individual[:, 0, crop] != 0] = 1
        # 前6块，中间14块，最后6块
        front_6 = crop_distribution[:6]
        middle_14 = crop_distribution[6:20]
        last_6 = crop_distribution[20:]
        # 计算占比
        total_sum = front_6.sum() + middle_14.sum() + last_6.sum()
        if total_sum != 0:
            front_6_zhanbi[crop] = front_6.sum() / total_sum
            middle_14_zhanbi[crop] = middle_14.sum() / total_sum
            last_6_zhanbi[crop] = last_6.sum() / total_sum
        else:
            front_6_zhanbi[crop] = 0
            middle_14_zhanbi[crop] = 0
            last_6_zhanbi[crop] = 0
        # 保存标准差结果
        std_results.append({
            "作物": crop,
            "前6块占比": front_6_zhanbi,
            "中间14块占比": middle_14_zhanbi,
            "后6块占比": last_6_zhanbi,
        })

    return std_results, front_6_zhanbi, middle_14_zhanbi, last_6_zhanbi

if __name__ == "__main__":
    # Excel文件中读取并存储
    # Excel 文件路径
    file_path = 'C:/Users/DoDO/Desktop/J.xlsx'
    # 指定需要读取的列
    usecols = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    # 读取数据的起始行和结束行
    start_row = 0
    end_row = 44
    # 调用函数读取数据
    costs, prices, yields = excel.read_and_extract_data(file_path, usecols, start_row, end_row)

    # 验证
    """
        print(f"costs: {costs[:5]}")
    for i, price_col in enumerate(prices):
        print(f"price_col_{i + 1}: {price_col[:5]}")
    for i, yield_col in enumerate(yields):
        print(f"yield_col_{i + 1}: {yield_col[:5]}")
    """
    # 地块面积
    area = np.array(
        [80, 55, 35, 72, 68, 55, 60, 46,
         40, 28, 25, 86, 55, 44, 50, 25,
         60, 45, 35, 20, 15, 13, 15, 18,
         27,20])

    # 创建遗传算法实例并运行
    ga_specific_parameters = GeneticAlgorithm(
        costs = costs,
        prices = prices,
        yields = yields,
        area = area,
        area_mode = 26,  # 地块数量
        crop_type = 15,  # 作物数量
        year = 7,  # 年份数量
        population_size = 50,  # 种群大小
        number_iterations = 200,  # 迭代次数
        probability_variation = 0.2  # 变异概率
    )
    # 运行遗传算法并输出结果
    best_solution = ga_specific_parameters.run_genetic_algorithm()
    best_fitness = ga_specific_parameters.fitness(best_solution, a = 0.9, b = 0.0375)

    #参数设置
    area_modes = 26  # 地块数量
    crop_types = 15  # 作物数量
    years = 7  # 年份数量

    #保存数据到EXCDEL文件之中
    print("Best solution:", best_solution)
    print("Best solution fitness:", best_fitness)
    excel.save_all_data_to_excel_by_year(file_path, best_solution, area, area_modes,
                                         years, crop_types, prices, yields, costs)

    #可视化热力图
    # 显示图标
    heatmap(best_solution, year_index = 1, area_mode = area_modes, crop_type = crop_types)



