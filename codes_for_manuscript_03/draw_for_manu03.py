# coding: utf-8
# 开发团队：重大化工LC214 AI小分队
# 开发人员：Tristan
# 开发时间：2024/4/9—23:49
# 文件名称：draw_for_manu03 py
# 开发工具：PyCharm

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import numpy as np

from calc_logP_QED_sa import mol_properties
from fragments2mol import *
from test_fp_similarity import *
import time
from littlecode.tools.chouyang import chouyang
from littlecode.tools.mkdir import mkdir
from codes_for_manuscript_02.my_som_cluster import SOM_learner

data_root = r'F:\manuscript3\rundata_for_manuscript03\generation3\result'
save_root = r'F:\manuscript3\英文期刊\图表'


def kernel_plots(prop='DRD2'):
    '''
    绘制核密度估计曲线for各个指标 in all_final_new_mols.csv
    '''

    # 数据
    all_final_new_mols = pd.read_csv(data_root+'\\all_final_new_mols.csv')
    all_final_new_mols.gen_id = 'Gen ' + all_final_new_mols.gen_id.astype(str)
    all_final_new_mols = all_final_new_mols.rename(columns={'new_mertric':'IDP'})

    # 设置图形风格
    # 设置画布大小
    plt.figure(figsize=(8, 6))  # 设置画布大小为 8x6
    sns.set(style="whitegrid", font_scale=1.4, font='times new roman' )

    lims = {'IDP':[(0.1, 0.5),(0, 15)],
            'DRD2':[(0.5, 1.1),(0, 7)],
            'QED':[(0.4, 1.1),(0, 7)],
            'SAscore':[(1, 4),(0, 1.4)]}
    plt.xlim(*lims[prop][0])  # 设置 x 轴刻度范围
    plt.ylim(*lims[prop][1])  # 设置 y 轴刻度范围


# # 绘制核密度曲线
    # sns.kdeplot(data=all_final_new_mols[prop], shade=True, label='DRD2')

    # 按照组别绘制核密度曲线
    for gen_id in all_final_new_mols.gen_id.unique():
        if gen_id == 'Gen 6':  # 第六代数量太少，不足以统计
            continue
        sns.kdeplot(data=all_final_new_mols[all_final_new_mols.gen_id == gen_id][prop], shade=True, label=gen_id)

    # 添加标题和标签

    plt.xlabel(prop)
    plt.ylabel('Density')

    # 显示图例
    plt.legend()
    # 保存结果
    plt.savefig(save_root+'\\show_section2\\Density for '+prop+' in all_final_new_mols.png', dpi = 1200)

    # 显示图形
    plt.show()

def kernel_plots2(prop='DRD2'):
    '''
    绘制核密度估计曲线for各个指标 in promising mols with cid.csv
    '''

    # 数据
    promising_mols_with_cid = pd.read_csv(r'F:\manuscript3\rundata_for_manuscript03\generation3\result\search_pubchem\promising mols with cid.csv',sep='\t')
    print(promising_mols_with_cid)
    # 设置图形风格
    # 设置画布大小
    plt.figure(figsize=(8, 6))  # 设置画布大小为 8x6
    sns.set(style="whitegrid", font_scale=1.4, font='times new roman' )

    lims = {'IDP':[(0.1, 0.5),(0, 10)],
            'DRD2':[(0.5, 1.1),(0, 6)],
            'QED':[(0.5, 1.1),(0, 7)],
            'SAscore':[(1, 4),(0, 1.2)]}
    plt.xlim(*lims[prop][0])  # 设置 x 轴刻度范围
    plt.ylim(*lims[prop][1])  # 设置 y 轴刻度范围


    # # 绘制核密度曲线
    # sns.kdeplot(data=all_final_new_mols[prop], shade=True, label='DRD2')

    # 按照组别绘制核密度曲线
    for activaty in promising_mols_with_cid.activaty.unique():

        if activaty == 'Not recorded':  # 不统计untested
            continue
        sns.kdeplot(data=promising_mols_with_cid[promising_mols_with_cid.activaty == activaty][prop], shade=True, label=activaty)

    # 添加标题和标签

    plt.xlabel(prop)
    plt.ylabel('Density')

    # 显示图例
    plt.legend()
    # 保存结果
    plt.savefig(save_root+'\\show_section4\\Density for '+prop+' in promising_mols_with_cid.png', dpi = 1200)

    # 显示图形
    plt.show()

def kde_2d_plots(prop1, prop2):
    '''
    绘制2D核密度估计曲线 in all_final_new_mols.csv
    后期加上
    '''

    # 数据
    all_final_new_mols = pd.read_csv(data_root+'\\all_final_new_mols.csv')
    all_final_new_mols.gen_id = 'Gen ' + all_final_new_mols.gen_id.astype(str)
    all_final_new_mols = all_final_new_mols.rename(columns={'new_mertric':'IDP'})

    # 设置图形风格
    # 设置画布大小
    plt.figure(figsize=(8, 6))  # 设置画布大小为 8x6
    sns.set(style="whitegrid", font_scale=1.4, font='times new roman' )

    lims = {'IDP':[(0.1, 0.5),(0, 15)],
            'DRD2':[(0.5, 1.1),(0, 7)],
            'QED':[(0.4, 1.1),(0, 7)],
            'SAscore':[(1, 4),(0, 1.4)]}
    plt.xlim(*lims[prop1][0])  # 设置 x 轴刻度范围
    plt.ylim(*lims[prop2][0])  # 设置 y 轴刻度范围

    # # 绘制核密度曲线
    # sns.kdeplot(data=all_final_new_mols[prop], shade=True, label='DRD2')

    # 绘制2D核密度曲线

    sns.jointplot(x=all_final_new_mols[prop1], y=all_final_new_mols[prop2], kind='kde', color="skyblue")

    # plt.xlabel(prop1)
    # plt.ylabel(prop2)

    # 保存结果
    plt.savefig(save_root+'\\show_section2\\2D Density for '+prop1+' vs '+prop2+' in all_final_new_mols.png', dpi = 1200)

    # 显示图形
    plt.show()

def scatters(prop1, prop2):
    '''
    绘制scatters in all_final_new_mols.csv
    后期加上
    '''

    # 数据
    all_final_new_mols = pd.read_csv(data_root+'\\all_final_new_mols.csv')
    all_final_new_mols.gen_id = 'Gen ' + all_final_new_mols.gen_id.astype(str)
    all_final_new_mols = all_final_new_mols.rename(columns={'new_mertric':'IDP'})

    # 设置图形风格
    # 设置画布大小
    plt.figure(figsize=(8, 6))  # 设置画布大小为 8x6
    sns.set(style="whitegrid", font_scale=1.4, font='times new roman' )

    lims = {'IDP':[(0.1, 0.5),(0, 15)],
            'DRD2':[(0.5, 1.1),(0, 7)],
            'QED':[(0.4, 1.1),(0, 7)],
            'SAscore':[(1, 4),(0, 1.4)]}
    plt.xlim(*lims[prop1][0])  # 设置 x 轴刻度范围
    plt.ylim(*lims[prop2][0])  # 设置 y 轴刻度范围

    # # 绘制核密度曲线
    # sns.kdeplot(data=all_final_new_mols[prop], shade=True, label='DRD2')

    # 绘制2D核密度曲线


    sns.jointplot(x=all_final_new_mols[prop1], y=all_final_new_mols[prop2], kind='scatter', color="skyblue")

    # plt.xlabel(prop1)
    # plt.ylabel(prop2)

    # 保存结果
    plt.savefig(save_root+'\\show_section2\\scatters for '+prop1+' vs '+prop2+' in all_final_new_mols.png', dpi = 1200)

    # 显示图形
    plt.show()

def generation_records_count():
    '''
    统计每代的生成结构数、碎片数
    '''
    root = r'F:\manuscript3\rundata_for_manuscript03\generation3'

    generation_records = pd.DataFrame()

    for i in range(7):
        if i == 0:
            len_all_new_mols = 181595
            final_new_mols = pd.read_csv(root+'\\gen'+str(i)+'\\selected_mols.csv')

        else:
            all_new_mols = pd.read_csv(root+'\\gen'+str(i)+'\\all_new_mols.csv')
            len_all_new_mols = all_new_mols.shape[0]

            final_new_mols = pd.read_csv(root+'\\gen'+str(i)+'\\final_new_mols.csv')

        len_final_new_mols = final_new_mols.shape[0]


        gen = i
        file1 = root + "generation3\\gen" + str(gen) + "\\init_scaffords_groups.csv"
        file2 = root + "generation3\\gen" + str(gen) + "\\init_decorations_groups.csv"
        file3 = root + "generation3\\gen" + str(gen) + "\\similar_scaffords_groups.csv"
        file4 = root + "generation3\\gen" + str(gen) + "\\similar_decorations_groups.csv"

        if (gen % 2) != 0:
            init_scaffords = pd.read_csv(file1,index_col=0)
            init_decorations = pd.read_csv(file2,index_col=0)
            len_decorations, len_scaffolds = (init_decorations.shape[1]-1)*5, (init_scaffords.shape[1]-1)*5
        else:
            try:
                init_scaffords = pd.read_csv(file3,index_col=0)
                init_decorations = pd.read_csv(file4,index_col=0)

                len_decorations, len_scaffolds = init_decorations.shape[1]-1, init_scaffords.shape[1]-1
            except:
                init_scaffords = pd.read_csv(file1,index_col=0)
                init_decorations = pd.read_csv(file2,index_col=0)
                len_decorations, len_scaffolds = (init_decorations.shape[1]-1)*5, (init_scaffords.shape[1]-1)*5

        record = {'Gen': i, 'Scaffolds': len_scaffolds,
                  'Decorations': len_decorations, 'all_new_mols': len_all_new_mols,
                  'final_new_mols': len_final_new_mols}
        generation_records = generation_records.append(record, ignore_index=True)

    generation_records.to_csv(root+'\\result\\N_generation_records.csv')

def yichangchuli(smi_list):
    result = []
    for i in smi_list:
        try:
            i = eval(i)
            i = i[0]
        except:
            i = i
        result.append(i)

    return result

def frags_cluster_get(scaffords,decorations):
    '''
    碎片化学空间确定
    :param scaffords: list
    :param decorations: list
    :return: 化学空间信息
    '''
    root = "F:\\manuscript3\\rundata_for_manuscript03\\"
    # 骨架
    if scaffords != []:
        # 补全分子
        scaffords_addH= []
        for item in scaffords:
            item = item.replace("Fr","H")
            scaffords_addH.append(item)
        # 计算fp
        scaffords_fp_cal = cal_fingerprint(scaffords_addH)
        scaffords_fp_cal.RDKitTopological(1024)
        scaffords_fp = scaffords_fp_cal.fp_TopoFingerprint.values
        # 读取SOM化学空间模型
        scaffords_model_path = root + "\\som化学空间\\Scaffold\\size20_sigma3_random_seed2023_model.p"

        scaffords_model = SOM_learner(dataset=scaffords_fp,trained_som=scaffords_model_path,size=20)
        scaffords_model.cluster_results()

        # 获取对应的cluster_index
        scaffords_cluster_results_table = scaffords_model.cluster_results_table
        scaffords_cluster_results_table['SMILES'] = scaffords
        scaffords_sample_count = scaffords_model.sample_count

    else:
        scaffords_sample_count = pd.DataFrame()
        scaffords_cluster_results_table = pd.DataFrame()

    # 修饰物
    if decorations != []:
        # 补全分子
        decorations_addH = []
        for item in decorations:
            item = item.replace("Cs","H")
            decorations_addH.append(item)
        # 计算fp
        decorations_fp_cal = cal_fingerprint(decorations_addH)
        decorations_fp_cal.RDKitTopological(1024)
        decorations_fp = decorations_fp_cal.fp_TopoFingerprint.values
        # 读取SOM化学空间模型
        decorations_model_path = root + "\\som化学空间\\decorations\\size10_sigma3_random_seed2023_model.p"

        decorations_model = SOM_learner(dataset=decorations_fp,trained_som=decorations_model_path,size=10)
        decorations_model.cluster_results()

        # 获取对应的cluster_index
        decorations_cluster_results_table = decorations_model.cluster_results_table
        decorations_cluster_results_table['SMILES'] = decorations
        decorations_sample_count = decorations_model.sample_count

    else:
        decorations_sample_count = pd.DataFrame()
        decorations_cluster_results_table = pd.DataFrame()

    return scaffords_cluster_results_table,scaffords_sample_count,\
           decorations_cluster_results_table,decorations_sample_count

def frags_count(save_path = r'F:\manuscript3\英文期刊\图表\show_section3'):
    '''
    统计每代的碎片化学空间信息
    :param save_path:
    :return:
    '''
    start_gen,end_gen = 0, 7
    root = "F:\\manuscript3\\rundata_for_manuscript03\\"
    for gen in range(start_gen,end_gen):
        file1 = root + "generation3\\gen" + str(gen) + "\\init_scaffords_groups.csv"
        file2 = root + "generation3\\gen" + str(gen) + "\\init_decorations_groups.csv"
        file3 = root + "generation3\\gen" + str(gen) + "\\similar_scaffords_groups.csv"
        file4 = root + "generation3\\gen" + str(gen) + "\\similar_decorations_groups.csv"
        print(gen)

        if (gen % 2) != 0:
            init_scaffords = pd.read_csv(file1,index_col=0)
            init_decorations = pd.read_csv(file2,index_col=0)
        else:
            try:
                init_scaffords = pd.read_csv(file3,index_col=0)
                init_decorations = pd.read_csv(file4,index_col=0)
            except:
                init_scaffords = pd.read_csv(file1,index_col=0)
                init_decorations = pd.read_csv(file2,index_col=0)

        len_decorations = init_decorations.shape[1]
        len_scaffords = init_scaffords.shape[1]

        # 扩充骨架
        all_scaffords = set()
        for i in range(len_scaffords):
            scaffords01 = init_scaffords.iloc[0,i]
            scaffords1 = eval(scaffords01)
            all_scaffords.update(set(scaffords1))

        # 扩充修饰物
        all_decorations = set()
        for j in range(len_decorations):
            decorations01 = init_decorations.iloc[0,j]
            decorations1 = eval(decorations01)
            decorations1 = yichangchuli(decorations1)
            all_decorations.update(set(decorations1))


        scaffords_cluster_results_table, scaffords_sample_count, decorations_cluster_results_table,decorations_sample_count = frags_cluster_get(scaffords=list(all_scaffords),decorations=list(all_decorations))

        scaffords_cluster_results_table.to_csv(save_path+'\\scaffolds\\gen'+str(gen)+'_scaffords_cluster_results_table.csv')
        decorations_cluster_results_table.to_csv(save_path+'\\decorations\\gen'+str(gen)+'_decorations_cluster_results_table.csv')

        np.savetxt(X=scaffords_sample_count,
                   fname=save_path+'\\scaffolds\\gen'+str(gen)+'_scaffords_sample_count.csv',
                   delimiter=",")
        np.savetxt(X=decorations_sample_count,
                   fname=save_path+'\\decorations\\gen'+str(gen)+'_scaffords_sample_count.csv',
                   delimiter=",")

def frags_count2(save_path = r'F:\manuscript3\英文期刊\图表\show_section3'):
    '''
    统计每代的碎片化学空间信息
    :param save_path:
    :return:
    '''
    root = r'F:\manuscript3\rundata_for_manuscript03\generation3\result'
    frag_type = ['decorations','scaffords']
    decorations_table, Scaffold_table = (pd.read_csv(root+'\\all_'+x+'_table.csv') for x in frag_type)
    i = 0
    for frags_table in [decorations_table,Scaffold_table]:
        if i == 0:
            N = 10 # decorations的SOM为10*10
        else:
            N = 20

    

def cluster_frags_prop():
    '''
    统计SOM碎片结构库中的每个cluster碎片的性质max
    :return: 统计表格矩阵形式N*N
    '''
    root = r'F:\manuscript3\rundata_for_manuscript03\generation3\result'
    frag_type = ['decorations','scaffords']
    decorations_table, Scaffold_table = (pd.read_csv(root+'\\all_'+x+'_table.csv') for x in frag_type)

    # 计算DRD2
    i = 0
    for frags_table in [decorations_table,Scaffold_table]:
        if i == 0:
            N = 10 # decorations的SOM为10*10
            frags_table['addH_SMILES'] = frags_table['smiles'].str.replace('Cs', 'H')
        else:
            N = 20
            frags_table['addH_SMILES'] = frags_table['smiles'].str.replace('Fr', 'H')

        # 计算map_x, map_y
        # 使用 divmod() 函数进行整除，并获取结果和余数
        frags_table['map_x'], frags_table['map_y'] = divmod(frags_table[frag_type[i]+'_location'], N)

        frags_list = frags_table['addH_SMILES'].tolist()
        a = mol_properties(frags_list)

        frags_table['DRD2'] = a.cal_DRD2()
        frags_table['SA'] = a.cal_sa()
        frags_table['QED'] = a.cal_qed()
        frags_table['SLOGP'] = a.cal_SLOGP()
        frags_table.to_csv(root+'\\max_heatmap\\'+frag_type[i]+'_space_with props.csv') #保存

        # 统计
        # 初始化 n×n 的矩阵，用于统计样本量
        count_matrix = np.zeros((N, N), dtype=int)

        # 统计样本量
        for index, row in frags_table.iterrows():
            x, y = row['map_x'], row['map_y']
            count_matrix[x][y] += 1

        # 将矩阵转换为 DataFrame
        count_matrix_df = pd.DataFrame(count_matrix, columns=[f'y_{i}' for i in range(N)], index=[f'x_{i}' for i in range(N)])

        # 保存 DataFrame 到 CSV 文件
        count_matrix_df.to_csv(root+'\\max_heatmap\\'+frag_type[i]+'_count.csv')



        # 按照 x 和 y 列进行分组，并计算每个组的最大值
        for prop in ['DRD2','SA','QED','SLOGP']:
            max_prop = frags_table.groupby(['map_x', 'map_y'])[prop].max()

            # 创建一个 N*N 的矩阵，填充为 NaN

            matrix = np.full((N, N), np.nan)

            # 将max值填充到矩阵中对应的位置
            for (x, y), value in max_prop.items():
                matrix[x][y] = value

            # 将矩阵转换为 DataFrame
            matrix_df = pd.DataFrame(matrix, columns=[f'y_{i}' for i in range(N)], index=[f'x_{i}' for i in range(N)])

            # 保存 DataFrame 到 CSV 文件
            matrix_df.to_csv(root+'\\max_heatmap\\'+frag_type[i]+'_max_of '+prop+'.csv')

        i += 1



if __name__ == '__main__':
    # for item in ['DRD2','QED','SAscore','IDP']:
    #     kernel_plots(item)

    # from itertools import combinations
    #
    # # 示例列表
    # lst = ['DRD2','QED','SAscore','IDP']
    # # 生成两两不同元素的组合
    # comb = list(combinations(lst, 2))
    #
    # for c in comb:
    #     kde_2d_plots(*c)

    # generation_records_count()

    # frags_count()

    # cluster_frags_prop()

    # lst = ['DRD2','QED','SAscore','IDP']
    # for prop in lst:
    #     kernel_plots2(prop)

    lst = ['DRD2','QED','SAscore','IDP']

    scatters('QED', 'DRD2')