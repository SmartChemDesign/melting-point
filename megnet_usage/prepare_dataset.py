from rdkit import Chem
from rdkit.Chem import AllChem
from pymatgen.core import Structure, Lattice, Molecule
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os

def smiles_to_pymatgen_structure(smiles, vacuum=10.0, seed=42):
    """SMILES → pymatgen Structure с вакуумом"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    mol = Chem.AddHs(mol)
    
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.enforceChirality = True
    
    try:
        AllChem.EmbedMolecule(mol, params=params)
        AllChem.UFFOptimizeMolecule(mol)
    except:
        return None
    
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords = mol.GetConformer().GetPositions().tolist()
    
    pmg_mol = Molecule(atoms, coords)
    
    coords_np = np.array(coords)
    min_coords = coords_np.min(axis=0)
    max_coords = coords_np.max(axis=0)
    mol_size = max_coords - min_coords
    cell_size = max(mol_size) + 2 * vacuum
    
    lattice = Lattice.cubic(cell_size)
    center = (min_coords + max_coords) / 2
    shifted_coords = coords_np - center + cell_size / 2
    
    structure = Structure(lattice, atoms, shifted_coords, coords_are_cartesian=True)
    
    return structure


def prepare_dataset_for_m3gnet(csv_path,
                                smiles_col='smiles',
                                target_col='melting_point',
                                output_pkl='m3gnet_dataset.pkl',
                                vacuum=10.0,
                                resume=True):
    """
    Готовит датасет для M3GNet БЕЗ создания CIF файлов
    Сохраняет Structures в pickle для быстрого восстановления
    """
    df = pd.read_csv(csv_path)
    
    # Проверка предыдущего прогресса
    structures = []
    start_idx = 0
    
    if resume and os.path.exists(output_pkl):
        with open(output_pkl, 'rb') as f:
            cached = pickle.load(f)
        start_idx = len(cached)
        structures = cached
        print(f"📂 Найдено {start_idx} закэшированных структур. Продолжаем...")
    
    pbar = tqdm(total=len(df), initial=start_idx, desc="Генерация структур")
    
    for idx in range(start_idx, len(df)):
        row = df.iloc[idx]
        smiles = row[smiles_col]
        target = row[target_col]
        
        structure = smiles_to_pymatgen_structure(smiles, vacuum=vacuum)
        
        if structure:
            structures.append({
                'structure': structure,
                'melting_point': target,
                'mol_id': f"mol_{idx:06d}",
                'n_atoms': structure.num_sites
            })
            pbar.set_postfix({'status': 'OK', 'atoms': structure.num_sites})
        else:
            pbar.set_postfix({'status': 'FAIL', 'atoms': 0})
        
        pbar.update(1)
        
        # Чекпоинт каждые 5000 молекул
        if (idx + 1) % 5000 == 0:
            with open(output_pkl, 'wb') as f:
                pickle.dump(structures, f)
            print(f"\n💾 Чекпоинт: {len(structures)} структур")
    
    pbar.close()
    
    # Финальное сохранение
    with open(output_pkl, 'wb') as f:
        pickle.dump(structures, f)
    
    # Создание DataFrame для обучения
    data_df = pd.DataFrame([
        {'mol_id': s['mol_id'], 
         'melting_point': s['melting_point'],
         'n_atoms': s['n_atoms']}
        for s in structures
    ])
    data_df.to_csv("m3gnet_metadata.csv", index=False)
    
    print(f"\n🎉 Готово!")
    print(f"   Структур: {len(structures)}")
    print(f"   Размер pickle: {os.path.getsize(output_pkl) / 1e9:.2f} ГБ")
    
    return structures


if __name__ == "__main__":
    structures = prepare_dataset_for_m3gnet(
        csv_path="/home/dubonos/MELTING-POINT-1/main_mp_df.csv",
        smiles_col="SMILES",
        target_col="MP",
        output_pkl="m3gnet_dataset.pkl",
        vacuum=0.0,
        resume=True
    )