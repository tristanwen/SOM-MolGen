#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from rdkit import rdBase, Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.Draw import SimilarityMaps
from pubchempy import get_compounds, Compound, get_properties
from rdkit.Chem import BRICS
from rdkit.Chem import rdFMCS


# In[3]:


# 检索相似分子，返回isomeric_smiles。第一个是原分子
def similarity_mol_search(mol_source='', list_max=10, source_type='smiles',search_by='similarity', dimension='2d'):
    # mol_source source_type  输入单个分子或者分子片段的 'name', 'cid',  'name', 'smiles', 'smarts'等
    # list_max 展示结果的数量限制。默认是前十条
    # dimension 2d or 3d, 后者费时间
    # search_by 相似物搜索similarity，否则identity
    
    if type(mol_source) == str:
        mol_source = [mol_source]
    
    if dimension == '2d':
        results = get_compounds(mol_source,namespace=source_type, searchtype = search_by, listkey_count=list_max)
        all_mols = [mol.isomeric_smiles for mol in results]
        all_mols_id = [mol.cid for mol in results]
    else:
        results = get_compounds(mol_source,namespace=source_type, searchtype = search_by, listkey_count=list_max, record_type='3d')
        all_mols_id = [mol.cid for mol in results]
        all_mols = [Compound.from_cid(cid) for cid in all_mols_id]
        all_mols = [mol.isomeric_smiles for mol in all_mols]

    # 返回smiles和对应id
    return all_mols, all_mols_id


# In[4]:
# 获取分子信息，返回为列表
def search_property(mol_id=[],identifier_type="smiles"):
    # 输入是上述的分子对应的 id

    properties = ['MolecularFormula',
                  'MolecularWeight',
                  'CanonicalSMILES',
                  'IsomericSMILES',
                  'InChI',
                  'InChIKey',
                  'IUPACName',
                  'XLogP',
                  'ExactMass',
                  'MonoisotopicMass',
                  'TPSA',
                  'Complexity',
                  'Charge',
                  'HBondDonorCount',
                  'HBondAcceptorCount',
                  'RotatableBondCount',
                  'HeavyAtomCount',
                  'IsotopeAtomCount',
                  'AtomStereoCount',
                  'DefinedAtomStereoCount',
                  'UndefinedAtomStereoCount',
                  'BondStereoCount',
                  'DefinedBondStereoCount',
                  'UndefinedBondStereoCount',
                  'CovalentUnitCount',
                  'Volume3D',
                  'XStericQuadrupole3D',
                  'YStericQuadrupole3D',
                  'ZStericQuadrupole3D',
                  'FeatureCount3D',
                  'FeatureAcceptorCount3D',
                  'FeatureDonorCount3D',
                  'FeatureAnionCount3D',
                  'FeatureCationCount3D',
                  'FeatureRingCount3D',
                  'FeatureHydrophobeCount3D',
                  'ConformerModelRMSD3D',
                  'EffectiveRotorCount3D',
                  'ConformerCount3D']

    results = pd.DataFrame()
    for id in mol_id:
        a = get_properties(properties, id, identifier_type, as_dataframe=True)
        results = pd.concat([results,a])

    return results


# In[5]:


# 分子切碎.根据键是否能够合成来进行拆解
def mol_split(mol='', node_info = False):
    # mol smiles字符串或者rdkit 的mol格式
    # node_info 是否需要标识节点所在序数，如下
    '''
    ['[16*]c1c(O)cc(CCCCC)cc1O',
    '[15*][C@@H]1C=C(C)CC[C@H]1C(=C)C',
    '[16*]c1cc(O)c([C@@H]2C=C(C)CC[C@H]2C(=C)C)c(O)c1',
    '[8*]CCCCC']
    '''
    
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)

    results = BRICS.BRICSDecompose(mol,singlePass=True)
    results = sorted(list(results))

    results = results[1:]

    results1 = [a.split('*]')[1] for a in results]
    if node_info:
        return results
    else:
        return results1
    
    


# In[6]:


# 读取为rdkit的分子对象
def read_mol(mol_source=[], source_type='smiles'):
    # mol_source 可以是sdf文件,mol文件，也可以是包含 'name', 'smiles', 'smarts'等的list
    # 或者是单个分子时候，直接读取 str
    
    # source_type 注明mol_source的类型
    # draw 是否画图
    
    if type(mol_source) == str:
        mol_source = [mol_source]
    
    if source_type == 'smiles':
        all_mols = [Chem.MolFromSmiles(mol_i) for mol_i in mol_source]
    elif source_type == 'smarts':
        all_mols = [Chem.MolFromSmarts(mol_i) for mol_i in mol_source]
    else: # 。。。。其他类型
        pass
    
    if len(all_mols) == 1:
        all_mols = all_mols[0]
        
        
    return all_mols


# In[7]:

# 画图
def draw_mol(mols, save_path=''):
    # mols mol格式的所有分子
    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(200,200), legends=['' for x in mols])
    if save_path != '':
        img.save(save_path)
    
    return img

# 画聚类的分子图
# all_mols = pd.read_excel("F:\\manuscript2\\制图\\enthalpy of formation of ideal gas\\reset版本2.xlsx", sheet_name="smiles")
def show_cluster_mols(all_mols, cluster_index = 0, select = 'all'):
    
    if select != 'all':
        mols_i = all_mols[all_mols["cluster_index"] == cluster_index]["SMILES"].tolist()
        mols_i = mols_i[select-1:select+1]
        print(mols_i)
        img = draw_mol(read_mol(mols_i))
    else:
        mols_i = all_mols[all_mols["cluster_index"] == cluster_index]["SMILES"]
        img = draw_mol(read_mol(mols_i))
    return img

if __name__ == '__main__':
    step=1
    root = "F:\\manuscript3\\rundata_for_manuscript03\\generation\\result3\\"
    if step == 1: # 数据搜集

        dataset = pd.read_csv(root+"pubchem_search\\all_final_new_mols_pubchem_缺失smiles.csv") # 第一次(root+"all_final_new_mols.csv")
        smi_list = dataset["SMILES"].tolist()

        results = search_property([smi_list[0]])


        # 空白值替换
        df = results.iloc[0]
        # 使用 empty() 方法清空 DataFrame 的值
        df.iloc[0, :] = None

        # 指定 CSV 文件路径和名称
        csv_file = root+"pubchem_search\\all_final_new_mols_pubchem.csv"
        # 在 DataFrame 中添加新列作为第一列
        results.insert(loc=0, column='SMILES', value=smi_list[0])
        results.to_csv(csv_file)

        # 逐行将数据写入 CSV 文件
        i = 1
        for smi in smi_list[1:]:
            try:
                for j in range(10):
                    result = search_property([smi])
                    # 在 DataFrame 中添加新列作为第一列
                    result.insert(loc=0, column='SMILES', value=smi)

                    if result["AtomStereoCount"] != 0 or str(0):
                        result.to_csv(csv_file,mode='a', header=True if i == 0 else False, index=True)
                        break


            except:
                pass

            i += 1


    elif step == 2: # 清除空白和重复
        import pandas as pd
        # df 是您的 DataFrame
        df = pd.read_csv(root+"pubchem_search\\all_final_new_mols_pubchem.csv")

        # 删除第一列为零的行
        df = df[df.iloc[:, 0] != 0]

        # 删除重复行
        df = df.drop_duplicates()

        # 保存更新后的 DataFrame
        df.to_csv(root+"pubchem_search\\all_final_new_mols_pubchem_清除后.csv")

    elif step == 3: # 比对smiles，提取未能采集的smiles,继续采集
        df1 = pd.read_csv(root+"all_final_new_mols.csv")
        smi_list1 = df1["SMILES"].tolist()

        df2 = pd.read_csv(root+"pubchem_search\\all_final_new_mols_pubchem_清除后.csv")
        smi_list2 = df2["SMILES"].tolist()

        droped_smiles = list(set(smi_list1)-set(smi_list2))
        droped_smiles = pd.DataFrame({"SMILES":droped_smiles})
        droped_smiles.to_csv(root+"pubchem_search\\all_final_new_mols_pubchem_缺失smiles.csv")



