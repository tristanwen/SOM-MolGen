# coding: utf-8
# 开发团队：重大化工LC214 AI小分队
# 开发人员：Tristan
# 开发时间：2023/4/11—10:34
# 文件名称：fragments2mol py
# 开发工具：PyCharm

from rdkit.Chem import BRICS
import rdkit.Chem as Chem
from rdkit.Chem import Draw
import random
import re
import itertools

'''
分子拆分和再组装的主要函数

其他工具：frag_smi_clean对于split_mol结果的碎片，可以用H修饰为正常分子
        draw_multi_mol 画分子
        read_mol 分子转格式
        
拆分：split_mol基于BRICS，需要控制min_Size

组装：frag2mol基于BRICS，但是自由度不大，会缺少很多组合；
    combine2frags基于标识的分子组合，一次性组装一对骨架-碎片，且只一次性只组装一个位点
    combine2_single_frag改进，对于多位点的骨架，指定第几个位点进行碎片组装，其余位置用H占
    combine2_multi_frags改进，对于多位点的骨架，对骨架所有位点按碎片列表顺序进行组装

综合：frag2mol_2 基于combine2frags，combine2_single_frag，combine2_multi_frags，对组装进行遍历，有两个模式：
        一，一次性只尝试一个位点，共组合M*N；二，尝试所有位点的所有组合M**N,M碎片个数，N位点个数

'''


single_atoms = [
    '[Cs]C', '[Cs]H', '[Cs]O', '[Cs]S', '[Cs]N', '[Cs]F', '[Cs]cl', '[Cs]br',
    '[Cs]I'
]


def draw_multi_mol(mols, mols_type="smiles", row_num=4, Size=200):
    '''
    分子画图器
    mols: 分子列表
    mols_type：分子格式 smiles or mols
    row_num：每行几个分子 4
    '''

    if mols_type == "mols":
        mols = mols
    elif mols_type == "smiles":
        mols = [Chem.MolFromSmiles(mol) for mol in mols]

    img = Chem.Draw.MolsToGridImage(mols,
                                    molsPerRow=row_num,
                                    subImgSize=(Size, Size))
    return img


def read_mol(mol_list=[], smiles2mol=1):
    '''
    分子格式转化smiles 2 mol
    smiles2mol: smiles 2 mol =1; mol 2 smiles =0

    '''
    mol_list2 = []

    for mol in mol_list:

        if smiles2mol:
            mol_list2.append(Chem.MolFromSmiles(mol))

        else:
            mol_list2.append(Chem.MolToSmiles(mol))

    return mol_list2


def split_mol(mols, mols_type="mol", min_Size=2, max_Size=10000, mode=""):
    '''
    分子切割器
    mols : 待切割的分子列表
    mols_type : mols的格式
    min_Size : 切割的参数minFragmentSize，生成骨架的时候用较大的minFragmentSize，如8
    max_Size: 原子数不超过max_Size, 默认10000不限制
    mode: scafford用Fr decoration用Cs
    '''
    if mols_type != "mol":
        smiles = mols  # 获得mols对应的smiles
        mols = read_mol(mol_list=mols, smiles2mol=1)
    else:
        smiles = read_mol(mol_list=mols, smiles2mol=0)

    smi_result = {}  # 保存碎片 以smiles 字典对应
    mol_result = {}  # 保存碎片 以mol
    smi_all_frags = set()  # 所有碎片 以smiles 不重复

    for i in range(len(mols)):  # 对每个分子切割
        try:
            smi_frags = list(
                BRICS.BRICSDecompose(mols[i], minFragmentSize=min_Size))

            if mode != "":
                smi_frags = frag_smi_clean(smi_frags,mode)
        except:
            print("wrong in: ", smiles[i])
            smi_frags = []

        if len(smi_frags) < 2:  # 没有拆分空间的直接跳过
            continue


        mol_frags = []
        for fsmi in smi_frags:

            fmol = Chem.MolFromSmiles(fsmi)
            if fmol.GetNumAtoms() > max_Size:
                smi_frags.remove(fsmi)
                continue

            smi_all_frags.update([fsmi])  # 传进字典，以保证不重复碎片
            mol_frags.append(fmol)  # 转为mol格式保存

        smi_result[smiles[i]] = smi_frags

        mol_result[smiles[i]] = mol_frags

    mol_all_frags = [Chem.MolFromSmiles(i)
                     for i in smi_all_frags]  # 转为smiles格式保存

    return mol_all_frags, smi_all_frags, smi_result, mol_result


# mol_all_frags, smi_all_frags, smi_result, mol_result = split_mol(mols)
# img = draw_multi_mol(mol_all_frags,mols_type='mols')
# img


# 第一种分子拼接方式
def frag2mol(frags_simles_list,
             randseed=100,
             size_limit=100,
             return_type='smiles'):
    '''
    分子合成器BRICS版本
    frags_simles_list：碎片的smiles
    randseed: 随机种子，保证可重复性
    size_limit: 当碎片很多样时候，可组合的分子有很多种，用以限制最大分子数，或者用一个大的阈值保证产生全部的组合
    return_type: 产生的分子格式 smiles 或 mols

    '''

    random.seed(randseed)  # 固定随机种子
    fragms = [Chem.MolFromSmiles(x)
              for x in sorted(frags_simles_list)]  # 读取碎片为mols
    ms = BRICS.BRICSBuild(fragms, scrambleReagents=False)  # 创建组合器

    prods = []  # 保存分子
    for i in range(size_limit):
        try:
            prods.append(next(ms))
        except:
            pass

    # [prod.UpdatePropertyCache(strict=True) for prod in prods] # 判断分子合理性，因为是BRICS切割和拼接，一般没问题
    if return_type == 'smiles':
        prods = [Chem.MolToSmiles(prod) for prod in prods]
    else:
        pass

    return prods


# mol_all_frags, smi_all_frags, smi_result, mol_result = split_mol(mols)
# gen_mols = frag2mol(smi_all_frags,return_type='mols')
# img = draw_multi_mol(gen_mols,mols_type='mols')
# img

# 第二种分子拼接方式


def get_neiid_bysymbol(mol, marker):
    # 获取marker邻居原子的index, 注意marker只能是一个单键连接核上的原子，否则邻居会多于一个
    try:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == marker:
                neighbors = atom.GetNeighbors()
                if len(neighbors) > 1:
                    print(
                        'Cannot process more than one neighbor, will only return one of them'
                    )
                atom_nb = neighbors[0]
                return atom_nb.GetIdx()
    except Exception as e:
        print(e)
        return None


def get_id_bysymbol(mol, marker):
    # 获取marker原子的index
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == marker:
            return atom.GetIdx()


def combine2frags(mol_a, mol_b, maker_b='Cs', maker_a='Fr'):
    # 将两个待连接分子置于同一个对象中

    merged_mol = Chem.CombineMols(mol_a, mol_b)
    bind_pos_a = get_neiid_bysymbol(merged_mol, maker_a)
    bind_pos_b = get_neiid_bysymbol(merged_mol, maker_b)
    # 转换成可编辑分子，在两个待连接位点之间加入单键连接，特殊情形需要其他键类型的情况较少，需要时再修改
    ed_merged_mol = Chem.EditableMol(merged_mol)
    ed_merged_mol.AddBond(bind_pos_a,
                          bind_pos_b,
                          order=Chem.rdchem.BondType.SINGLE)
    # 将图中多余的marker原子逐个移除，先移除marker a
    marker_a_idx = get_id_bysymbol(merged_mol, maker_a)
    ed_merged_mol.RemoveAtom(marker_a_idx)
    # marker a移除后原子序号变化了，所以又转换为普通分子后再次编辑，移除marker b
    temp_mol = ed_merged_mol.GetMol()
    marker_b_idx = get_id_bysymbol(temp_mol, maker_b)
    ed_merged_mol = Chem.EditableMol(temp_mol)
    ed_merged_mol.RemoveAtom(marker_b_idx)
    final_mol = ed_merged_mol.GetMol()
    return final_mol


def combine2_single_frag(mol_a,
                         mol_b,
                         maker_b='Cs',
                         maker_a='Fr',
                         decorated_location=0,
                         padding='[H]',
                         input_smiles=1):
    '''
    对于多位点的骨架mol_a，选择第几个位点插入碎片，其余位置用H代替
    padding：代替物，默认H
    decorated_location: 要插入的第几个位点，0，1，2...
    input_smiles: 是否为smiles格式的输入
    '''

    # 转换格式
    if input_smiles:
        smi_a, smi_b = mol_a, mol_b
        mol_a, mol_b = tuple(read_mol([mol_a, mol_b], smiles2mol=1))
    else:
        smi_a, smi_b = tuple(read_mol([mol_a, mol_b], smiles2mol=0))

    c = smi_a.split('[' + maker_a + ']')

    d = ''
    for i in range(len(c)):

        if i == decorated_location:
            decoration = '[' + maker_a + ']'  # 保留指定位点的'[Fr]'
        else:
            decoration = padding  # 其他位点用H代替

        if i < len(c) - 1:
            d = d + c[i] + decoration
        else:
            d = d + c[i]
            break

    mol_d = combine2frags(read_mol([d])[0],
                          mol_b,
                          maker_b=maker_b,
                          maker_a=maker_a)

    return mol_d


def combine2_multi_frags(smi_a,
                         smi_bs,
                         maker_b='Cs',
                         maker_a='Fr'):
    '''
    对于多位点的骨架mol_a，将所有碎片按  顺序  放置在 对应位点
    smi_a,smi_bs: smiles格式的输入, 'a',['a','b']

    # A = combine2_multi_frags('[Fr]C(CC)C(=O)C(C)C(O)C([Fr])C', ['[Cs]CCCO', '[Cs]CCCN'])
    # print(A)

    '''

    new_mol = smi_a
    for smi_b in smi_bs:
        try:
            new_mol = combine2frags(Chem.MolFromSmiles(new_mol), Chem.MolFromSmiles(smi_b), maker_b=maker_b,maker_a=maker_a)
            new_mol = Chem.MolToSmiles(new_mol)
        except:
            continue

    return new_mol


def frag_smi_clean(smi_frags, replace_atom="[H]"):
    '''
    将碎片smiles的[数字*]替换为其他
    smi_frags：smiles列表
    replace_atom：替换的原子，[H]，或者[Fr]，[Cs]
    '''
    results = []
    for s in smi_frags:
        pattern = r"\[\d+\*\]"
        replacement = replace_atom
        res = re.sub(pattern, replacement, s)  # 替换字符串
        results.append(res)

    if replace_atom == "[H]":  # 进一步校正，如果是H的话，保证规范
        results = read_mol(results, smiles2mol=1)
        results = read_mol(results, smiles2mol=0)

    return results


def frag2mol_2(Scaffold_simles_list,
               frags_simles_list,
               fully_decorated=0):
    '''
    分子合成器 自己改良版本，因为BRICS有的碎片不能组合在一起
    通过将碎片种植在骨架上，生成新的分子

    Scaffold_simles_list: 骨架的smiles, 结点标记为[Fr]，如果没有标记出来自动转换
    frags_simles_list：碎片的smiles, 结点标记为[Cs]，如果没有标记出来自动转换
    return_type: 产生的分子格式 smiles 或 mols
    fully_decorated: 1, 每个位点都插入碎片，新分子数量=碎片种数**位点数；0, 一次性只插入一个碎片到一个位点，其他位点用'[Cs]H'，新分子数量=碎片种数*位点数
                    当位点只有一个的时候，0
                    ****遍历fully_decorated的时候，注意每次计算的时候Scaffold_simles_list，frags_simles_list的个数不要太多，否则会组合爆炸。
                    这是在确认哪几个碎片好用的时候才进行遍历fully_decorated

    碎片补全：令frags_simles_list = ['[Cs]H']，方便计算相似性和化学空间位置

    # m1 = 'P(c1ccccc1)(c2ccccc2)c3ccccc3'
    # m2 = 'C(=O)(O)c1ccccc1C2=C3C=CC(=O)C(Br)=C3Oc4c(Br)c(O)ccc24'
    # m3 = 'C(C#N)(C(=NO)Cc1ccccc1)c2ccccc2'
    # m4 = 'C1(O)C(C)C(O)(OC(C1CC)C(C)C(O)C(C)C(=O)C(CC)C2OC(C)(CC2C)C3(O)OC(CC)(CC3C)C(O)CCC)C(CC)C(=O)O'
    # m5 = 'C1CN(CCN1C/C=C/C2=CC=CC=C2)C(C3=CC=CC=C3)C4=CC=CC=C4'
    # m6 = 'N(=N(=O)c1ccc(cc1OC)C(=O)c2ccccc2)c3ccc(cc3OC)C(=O)c4ccccc4'
    # mols = read_mol([m1,m2,m3,m4,m5,m6])
    #
    # mol_all_frags, smi_all_frags1, smi_result, mol_result = split_mol(mols,min_Size=10)
    # mol_all_frags, smi_all_frags2, smi_result, mol_result = split_mol(mols,min_Size=2)
    # m1 = list(smi_all_frags1)
    # m2 = list(smi_all_frags2)[2:4]
    # results1, results2 = frag2mol_2(m1, m2, fully_decorated=1)
    # print(results1)

    '''
    # 转为list格式
    if type(Scaffold_simles_list) == type(set()):
        Scaffold_simles_list = list(Scaffold_simles_list)
        frags_simles_list = list(frags_simles_list)

    # 如果没有标记出来自动转换
    if '[Fr]' not in Scaffold_simles_list[0]:
        Scaffold_simles_list = frag_smi_clean(Scaffold_simles_list,
                                              replace_atom='[Fr]')

    if '[Cs]' not in frags_simles_list[0]:
        frags_simles_list = frag_smi_clean(frags_simles_list,
                                           replace_atom='[Cs]')

    all_new_mols = {}  # 以字典保存smiles格式
    list_all = set()  # 笼统保存所有分子
    # 遍历所有骨架，为其装饰
    for Scaffold in Scaffold_simles_list:
        Connection_count = Scaffold.count('[Fr]')  # 可拼接点的数量

        # 每个骨架一次生成中只要一种碎片
        if fully_decorated == 0 or Connection_count == 1:
            for frag in frags_simles_list:
                all_new_mols[Scaffold + '-' + frag] = set()  # 初始化（'骨架-碎片'：新分子）
                for node in range(Connection_count):
                    try:
                        new_mol = combine2_single_frag(
                            Scaffold, frag, decorated_location=node)  # 生成新分子
                        new_mol = Chem.MolToSmiles(new_mol)

                        new_mol = new_mol.replace("[Cs]", "[H]") # ****多余未结合的Cs填充为H

                        all_new_mols[Scaffold + '-' + frag].update(
                            [new_mol])  # 保存为smiles格式
                        list_all.update([new_mol])
                    except:
                        continue

        elif fully_decorated == 1 and Connection_count > 1:

            all_new_mols[Scaffold] = set() # 初始化（'骨架'：新分子）

            # 生成迭代器perms, 碎片在位点上的排列组合情况
            perms = tuple([frags_simles_list for i in range(Connection_count)])
            perms = itertools.product(*perms)

            for perm in perms:
                new_mol = combine2_multi_frags(Scaffold, perm)  # 生成新分子
                new_mol = new_mol.replace("[Cs]", "[H]") # ****多余未结合的Cs填充为H
                all_new_mols[Scaffold].update([new_mol])  # 保存为smiles格式
                list_all.update([new_mol])
        else:
            pass

    return all_new_mols, list_all
