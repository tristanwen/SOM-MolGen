# coding: utf-8
# 开发团队：重大化工LC214 AI小分队
# 开发人员：Tristan
# 开发时间：2023/5/12—16:27
# 文件名称：calc_logP_QED_sa py
# 开发工具：PyCharm

from rdkit.Chem import QED
import sys
sys.path.append(r"E:\Pycharm projects\codes_for_manuscript_03")
from fragments2mol import read_mol
from SAscorer import *
sys.path.append(r"E:\Pycharm projects\IL_Generator")
from CO2_solubility_prediction.test import all_predictors
import numpy as np
import sys
sys.path.append(r"E:\Pycharm projects\MCMG-master\MCMG-master\MCMG_utils")
from properties import drd2_model2,gsk3_model2,jnk3_model2

class mol_properties:
    def __init__(self, mol_list=[]):
        self.mol_list0 = mol_list
        if mol_list == []: # 示例数据
            self.mol_list = read_mol(["O=C1[C@@H](CC)N(CC2=CC(C)=CS2)C3=C(N1C)C=NC(NC4=C(OCC)C=C(C5CCN(C)CC5)C=C4)=N3",
                                  "S(SC1=Nc2ccccc2S1)C3=Nc4ccccc4S3",
                                  "O=C1c2ccccc2C(=O)c3ccc(N)cc13",
                                  "C(=O)(O)c1ccccc1C2=C3C=CC(=O)C(Br)=C3Oc4c(Br)c(O)ccc24",
                                  "O=C1c2ccccc2C(=O)C(Cl)=C1N(C)C"])

        if type(mol_list[0]) == str: # smiles格式转为mols
            self.mol_list = read_mol(mol_list)
        else:
            self.mol_list = mol_list

    def cal_qed(self):
        QED_list = [QED.qed(mol) for mol in self.mol_list]
        return QED_list

    def cal_SLOGP(self):
        SLOGP_list = [QED.properties(mol).ALOGP for mol in self.mol_list]
        return SLOGP_list

    def cal_sa(self):
        sa_list = [calculateScore(mol) for mol in self.mol_list]
        return sa_list

    def cal_DRD2(self):
        drd2_list = drd2_model2(self.mol_list0).drd2
        return drd2_list.tolist()

    def cal_gsk3(self):
        gsk3_list = gsk3_model2(self.mol_list0).gsk3
        return gsk3_list.tolist()

    def cal_jnk3(self):
        jnk3_list = jnk3_model2(self.mol_list0).jnk3
        return jnk3_list.tolist()

    def cal_co2_solubility(self,temperature_input=298.15,pressure_input=10):
        '''
        :param temperature_input: 温度k
        :param pressure_input: 压强bar
        :return: 5个最优模型的预测结果的均值，list
        temperature_input = 298.15  # 温度，单位为K
        pressure_input = 10  # 压力，单位为bar

        '''

        result = []
        for smi in self.mol_list0:
            try:
                co2_solubility = all_predictors(smi,temperature_input,pressure_input)
            except:
                print('Cannot calc co2_solubility for: %s \n'%smi)
                co2_solubility = 0
            result.append(np.mean(co2_solubility))
        return result

    def calc_num_atoms(self):
        '''
        计算结构的非氢原子数量
        :param
        :return
        '''
        non_h_count_list = []
        for mol in self.mol_list:
            try:
                non_h_atoms = [atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]
            except:
                non_h_atoms = []
            non_h_count = len(non_h_atoms)
            non_h_count_list.append(non_h_count)

        return non_h_count_list

def IDP(QED,SAscore,DRD2):
    return np.sqrt(((QED-1)/1)**2+((SAscore-1)/7)**2+((DRD2-1)/1)**2)


if __name__ == '__main__':

    # mol_list0 = ['OCCCN1CCN(CC1)c2c(Cl)c(Cl)ccc2',
    #              'Cc1nc(NCCN2CCN(CC2)c3c(Cl)c(Cl)ccc3)sn1',
    #              'OCCCN1CCCN(CC1)c2c(Cl)c(Cl)ccc2',
    #              'O=c1c(CCCN2CCN(CC2)c3c(Cl)c(Cl)ccc3)c1',
    #              'SCCN1CCN(CC1)c2c(Cl)c(Cl)ccc2',
    #              'CC1CCC1CN2CCN(CC2)c3c(Cl)c(Cl)ccc3']

    import pandas as pd
    table = pd.read_csv(r'F:\manuscript3\rundata_for_manuscript03\初始数据\清洗后分子总样本smiles_with drd2.csv')
    mol_list0 = table['SMILES'].tolist()
    a = mol_properties(mol_list0)
    QED_list, SAscore_list, DRD2_list = a.cal_qed(),a.cal_sa(),a.cal_DRD2()
    IDP_list = IDP(np.array(QED_list), np.array(SAscore_list), np.array(DRD2_list))
    # for QED, SAscore, DRD2 in zip(QED_list, SAscore_list, DRD2_list):
    #     print(DRD2, IDP(QED, SAscore, DRD2))

    new_table = pd.DataFrame({'SMILES': mol_list0, 'QED':QED_list,'SA':SAscore_list,'DRD2':DRD2_list,'IDP':list(IDP_list)})
    selected_table = new_table[new_table['IDP']<0.44]
    new_table.to_csv(r'F:\manuscript3\rundata_for_manuscript03\generation3\result\initial 18w+mols info.csv')
    selected_table.to_csv(r'F:\manuscript3\rundata_for_manuscript03\generation3\result\Gen0_parents.csv')




