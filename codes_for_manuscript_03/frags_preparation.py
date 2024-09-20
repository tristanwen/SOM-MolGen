# coding: utf-8
# 开发团队：重大化工LC214 AI小分队
# 开发人员：Tristan
# 开发时间：2023/4/20—9:53
# 文件名称：frags_preparation py
# 开发工具：PyCharm

'''
骨架-碎片准备工作

骨架碎片库：生成，清洗，归类，编码
初始骨架碎片集：筛选性能高的若干分子，拆分
变异父母集：通过化学空间+相似性寻找变异碎片和骨架。变异的碎片直接加入原始碎片母集，变异的骨架另作新的父集
子集：生成的新分子通过构效关系评估性能，筛除性能没有提升的个体

'''

import os
os.chdir("E:\\Pycharm projects")
from codes_for_manuscript_03.fragments2mol import *
import pandas as pd
from littlecode.tools.mkdir import mkdir

'''
步骤：
1.先清洗smiles,clean_smi
2.读取所有清洗后的smiles,read_smi_csv
3.碎片化split_mol
4.归纳碎片集，filter_frags
'''


def clean_smi(infile="F:\\WORK\\和外合作\\天宇媛媛毕设\\清洗后分子总样本id+cas+smiles.csv",
              outfile="F:\\WORK\\和外合作\\天宇媛媛毕设\\清洗后分子总样本smiles.csv"):
    '''
    判断分子是否正确，并筛除、另存

    '''
    dataset = pd.read_csv(infile)
    len_before_clean = len(dataset)
    sample_smi = read_smi_csv(file=infile)
    dataset["mol"] = read_mol(sample_smi)
    dataset = dataset.dropna(axis=0, subset=["mol"])
    len_after_clean = len(dataset)
    dataset[['index', 'CAS', 'SMILES']].to_csv(outfile, index=False)

    return len_before_clean - len_after_clean


def read_smi_csv(file="F:\\WORK\\和外合作\\天宇媛媛毕设\\清洗后分子总样本smiles.csv",
                 num_smiles=-1):
    '''
    读取smiles文件
    file：绝对位置
    num_smiles：需要提取前多少条，默认提取所有
    '''

    dataset = pd.read_csv(file)
    if num_smiles == -1:
        smiles = list(dataset["SMILES"])
    else:
        smiles = list(dataset["SMILES"])[:num_smiles]

    return smiles


def filter_frags(frags_smi, limit=8, keep_single_node=0, save_path=0):
    '''
    对于碎片长短进行筛选，依据C原子数。在此范围内的归纳为'修饰碎片',更大的归纳为'骨架'
    修饰的碎片默认限制在2~8，骨架限制在8以上
    frags_smi: 拆分分子后的碎片 smiles list
    keep_single_node: 修饰碎片是否保持一个位点0 1 ，其它位点用H占。建议只保持一个连接位点
    save_path: 保存位置，不保存0,。 "F:\\WORK\\和外合作\\天宇媛媛毕设"
    '''
    frags_smi_with_H = frag_smi_clean(frags_smi)  # 补上H

    # decorations 修饰碎片
    result_decorations_smi = []  # 保存结果
    result_decorations_smi_with_H = []  # 保存结果with_H
    # Scaffold 骨架
    result_Scaffold_smi = []  # 保存结果
    result_Scaffold_smi_with_H = []  # 保存结果with_H

    for i in range(len(frags_smi)):

        mol_i_with_H = frags_smi_with_H[i]
        mol_i = read_mol([mol_i_with_H])[0]  # 读取为mol
        atom_num = mol_i.GetNumAtoms()  # 原子个数

        if atom_num < limit:  # 修饰碎片
            result_decorations_smi.append(frags_smi[i])
            result_decorations_smi_with_H.append(mol_i_with_H)
        else:  # 骨架
            result_Scaffold_smi.append(frags_smi[i])
            result_Scaffold_smi_with_H.append(mol_i_with_H)

    # 标识为Fr, Cs
    result_decorations_smi = frag_smi_clean(result_decorations_smi, replace_atom="[Cs]")
    result_Scaffold_smi = frag_smi_clean(result_Scaffold_smi, replace_atom="[Fr]")

    # 修饰碎片保留一个位点
    if keep_single_node:
        for i in range(len(result_decorations_smi)):
            a = result_decorations_smi[i]
            if a.count("Cs") > 1:
                a = a.replace("Cs", "H", a.count("Cs") - 1)  # 只保留一个Cs位点，其余全为H
                a = read_mol(read_mol([a]), smiles2mol=0)  # 规范smiles
                result_decorations_smi[i] = a
            else:
                continue

    # 打包
    table_decorations = pd.DataFrame(
        {"frag_SMILES": result_decorations_smi, "addH_SMILES": result_decorations_smi_with_H})
    table_Scaffold = pd.DataFrame({"frag_SMILES": result_Scaffold_smi, "addH_SMILES": result_Scaffold_smi_with_H})

    # 保存文件
    if save_path != 0:
        mkdir(save_path, False)  # 创建文件夹
        table_decorations.to_csv(save_path + "\\decorations.csv", index=False)
        table_Scaffold.to_csv(save_path + "\\Scaffold.csv", index=False)

    return table_decorations, table_Scaffold

if __name__ == '__main__':
    save_path = "F:\\WORK\\和外合作\\天宇媛媛毕设\\初始数据"
    all_smiles = read_smi_csv()
    all_smiles1 = all_smiles

    all_frags_smi = set() # 所有碎片
    # 分批运行
    n = 30000
    for i in range(0, len(all_smiles1), n):
        print(i)
        part_smiles = all_smiles1[i:i + n]
        _, smi_all_frags, _, _ = split_mol(part_smiles, mols_type="smiles")
        all_frags_smi.update(smi_all_frags)

    all_frags_smi = list(all_frags_smi)

    table_decorations, table_Scaffold = filter_frags(all_frags_smi, keep_single_node=1, save_path=save_path)
    print(len(table_decorations), len(table_Scaffold))
    print(table_decorations.head())
    print(table_Scaffold.head())
