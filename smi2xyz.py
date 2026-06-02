from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolfiles
import os
import pandas as pd
import numpy as np
from typing import List, Optional
from tqdm import tqdm 
import json
from datetime import datetime

class SmilesTo3DConverter:
    def __init__(self, 
                 n_conformers: int = 10,
                 n_lowest_energy: int = 3,
                 max_atoms: int = 100,
                 random_seed: int = 42):
        self.n_conformers = n_conformers
        self.n_lowest_energy = n_lowest_energy
        self.max_atoms = max_atoms
        self.random_seed = random_seed
    
    def smiles_to_3d(self, 
                     smiles: str, 
                     mol_id: str, 
                     output_dir: str = "structures") -> Optional[List[str]]:
        os.makedirs(output_dir, exist_ok=True)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mol = Chem.AddHs(mol)
        
        if mol.GetNumAtoms() > self.max_atoms:
            return None
        
        params = AllChem.ETKDGv3()
        params.randomSeed = self.random_seed
        params.numThreads = 0
        params.maxAttempts = 100
        params.enforceChirality = True
        
        conformer_ids = AllChem.EmbedMultipleConfs(mol, 
                                                    numConfs=self.n_conformers, 
                                                    params=params)
        
        if len(conformer_ids) == 0:
            return None
        
        energies = []
        for conf_id in conformer_ids:
            try:
                energy = AllChem.UFFOptimizeMolecule(mol, confId=conf_id)
                energies.append((conf_id, energy))
            except:
                continue
        
        if len(energies) == 0:
            return None
        
        energies.sort(key=lambda x: x[1])
        best_conformers = energies[:self.n_lowest_energy]
        
        saved_files = []
        for i, (conf_id, energy) in enumerate(best_conformers):
            xyz_content = rdmolfiles.MolToXYZBlock(mol, confId=conf_id)
            xyz_path = os.path.join(output_dir, f"{mol_id}_conf{i}.xyz")
            with open(xyz_path, 'w') as f:
                f.write(xyz_content)
            saved_files.append(xyz_path)
        
        return saved_files
    
    def process_dataset(self, 
                        csv_path: str, 
                        smiles_col: str = 'smiles', 
                        target_col: str = 'MP',
                        id_col: str = None,
                        output_dir: str = "structures",
                        resume: bool = True) -> pd.DataFrame:

        df = pd.read_csv(csv_path)
        
        # 6-значная нумерация для 118 000+ молекул
        if id_col is None:
            df['mol_id'] = [f"mol_{i:06d}" for i in range(len(df))]
        else:
            df['mol_id'] = df[id_col].astype(str)
        
        log_path = "conversion_log.csv"
        start_idx = 0
        results = []
        
        if resume and os.path.exists(log_path):
            existing_log = pd.read_csv(log_path)
            if len(existing_log) > 0:
                start_idx = len(existing_log)
                results = existing_log.to_dict('records')
                print(f"found {start_idx} already processed molecules. continue...")
        
        pbar = tqdm(total=len(df), initial=start_idx, desc="convertation")
        
        for idx in range(start_idx, len(df)):
            row = df.iloc[idx]
            mol_id = row['mol_id']
            smiles = row[smiles_col]
            target = row[target_col]
            
            xyz_files = self.smiles_to_3d(smiles, mol_id, output_dir)
            
            if xyz_files:
                results.append({
                    'structure_file': f"{mol_id}_conf0.xyz",
                    'MP': target,
                    'mol_id': mol_id,
                    'n_conformers_generated': len(xyz_files)
                })
                pbar.set_postfix({'status': 'OK', 'conf': len(xyz_files)})
            else:
                pbar.set_postfix({'status': 'FAIL', 'conf': 0})
            
            pbar.update(1)
            
            if (idx + 1) % 1000 == 0:
                self._save_checkpoint(results, log_path)
                print(f"\checkpoint is saved on molecule {idx + 1}")
        
        pbar.close()
        
        self._save_checkpoint(results, log_path)
        self._create_id_prop(results)
        
        print(f"\n🎉 Завершено!")
        print(f"   Успешно: {len(results)}")
        print(f"   Пропущено: {len(df) - len(results)}")
        
        return pd.DataFrame(results)
    
    def _save_checkpoint(self, results, log_path):
        df_log = pd.DataFrame(results)
        df_log.to_csv(log_path, index=False)
    
    def _create_id_prop(self, results):
        df = pd.DataFrame(results)
        # ALIGNN требует CSV без заголовка: filename,target
        df[['structure_file', 'MP']].to_csv("id_prop.csv", 
                                                        index=False, 
                                                        header=False)
        print("   file id_prop.csv created.")

if __name__ == "__main__":
    converter = SmilesTo3DConverter(
        n_conformers=10,
        n_lowest_energy=1,
        random_seed=42
    )
    
    results = converter.process_dataset(
        csv_path="/home/dubonos/MELTING-POINT-1/main_mp_df.csv",
        smiles_col="SMILES",
        target_col="MP",
        id_col=None,
        output_dir="structures",
        resume=True  
    )