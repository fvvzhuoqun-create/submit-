import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from rdkit import Chem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit is not installed, will use simplified molecular representation")


class DrugCellDataProcessor:
    def __init__(self, drug_data_file, drug_target_file, cell_line_file, target_features_file='target_features.csv'):
        logger.info("Initializing Data Processor (Enhanced)...")

        self.drug_smiles_map, self.drug_physchem = self._load_drug_data(drug_data_file)
        self.drug_targets = self._load_and_process_targets(drug_target_file, target_features_file)
        self.cell_line_expr = self._load_cell_features(cell_line_file)

        self.graph_cache = {}
        self.atom_feature_dim = 64

        self.physchem_dim = self.drug_physchem.shape[1] if self.drug_physchem is not None else 0

        # Determine target dimension dynamically, default to 1024
        if self.drug_targets:
            self.target_dim = len(next(iter(self.drug_targets.values())))
        else:
            self.target_dim = 1024

        self.cell_dim = self.cell_line_expr.shape[1]

        logger.info(f"Number of drugs: {len(self.drug_smiles_map)}")
        logger.info(f"Physicochemical feature dimension: {self.physchem_dim}")
        logger.info(f"Target feature dimension: {self.target_dim}")
        logger.info(f"Cell line feature dimension: {self.cell_dim}")

    def _load_drug_data(self, file_path):
        try:
            df = pd.read_csv(file_path)
            if 'drugName' not in df.columns:
                raise ValueError("Drug data file is missing 'drugName' column")

            df['drugName'] = df['drugName'].astype(str).str.strip()
            df = df.drop_duplicates(subset=['drugName'])
            df.set_index('drugName', inplace=True)

            smiles_map = {}
            if 'SMILES' in df.columns:
                for idx, row in df.iterrows():
                    smiles = str(row['SMILES']).strip()
                    if self._validate_smiles(smiles):
                        smiles_map[str(idx)] = smiles

            physchem_cols = ['MW', 'logP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds', 'HeavyAtoms']
            valid_cols = [c for c in physchem_cols if c in df.columns]
            physchem_df = df[valid_cols].fillna(0).astype(float)

            scaler = StandardScaler()
            physchem_values = scaler.fit_transform(physchem_df)
            physchem_df = pd.DataFrame(physchem_values, index=physchem_df.index, columns=valid_cols)

            return smiles_map, physchem_df

        except Exception as e:
            logger.error(f"Failed to load drug data: {e}")
            raise

    def _validate_smiles(self, smiles):
        if not smiles or pd.isna(smiles) or len(str(smiles)) < 2:
            return False
        return True

    def _load_and_process_targets(self, drug_target_path, target_features_path):
        try:
            logger.info("Loading 1024-dimensional continuous target features...")

            # 1. Load pre-computed target embeddings
            target_df = pd.read_csv(target_features_path)
            target_embeddings = {}
            for _, row in target_df.iterrows():
                t_name = str(row['Target_Name'])
                # Extract the 1024-d features (assuming first column is Target_Name)
                features = row.iloc[1:].values.astype(np.float32)
                target_embeddings[t_name] = features

            # 2. Map drugs to their corresponding target list
            dt_df = pd.read_csv(drug_target_path)
            dt_df['csv_drug_name'] = dt_df['csv_drug_name'].astype(str).str.strip()
            drug_to_targets = dt_df.groupby('csv_drug_name')['target_name'].apply(list).to_dict()

            # 3. Apply Mean Pooling to aggregate multiple target features for each drug
            drug_target_features = {}
            feature_dim = len(next(iter(target_embeddings.values()))) if target_embeddings else 1024

            for drug, targets in drug_to_targets.items():
                embeddings = []
                for t in targets:
                    if t in target_embeddings:
                        embeddings.append(target_embeddings[t])

                if len(embeddings) > 0:
                    drug_target_features[drug] = np.mean(embeddings, axis=0)
                else:
                    drug_target_features[drug] = np.zeros(feature_dim, dtype=np.float32)

            return drug_target_features

        except Exception as e:
            logger.error(f"Failed to load target data: {e}")
            return {}

    def _load_cell_features(self, file_path):
        try:
            df = pd.read_csv(file_path)
            if 'Name' in df.columns:
                df.rename(columns={'Name': 'cell_line'}, inplace=True)
            elif 'cell_line' not in df.columns:
                raise ValueError("Cell line file is missing 'cell_line' column")
                
            df.set_index('cell_line', inplace=True)
            return df.select_dtypes(include=[np.number]).astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to load cell line data: {e}")
            raise

    def get_drug_smiles(self, drug_name):
        drug_name = str(drug_name).strip()
        return self.drug_smiles_map.get(drug_name, 'C')

    def get_physchem_features(self, drug_name):
        drug_name = str(drug_name).strip()
        if self.drug_physchem is not None and drug_name in self.drug_physchem.index:
            vals = self.drug_physchem.loc[drug_name].values.astype(np.float32)
            return torch.from_numpy(vals)
        return torch.zeros(self.physchem_dim, dtype=torch.float32)

    def get_target_features(self, drug_name):
        drug_name = str(drug_name).strip()
        features = self.drug_targets.get(drug_name, np.zeros(self.target_dim, dtype=np.float32))
        return torch.tensor(features)

    def get_cell_line_features(self, cell_line):
        cell_line = str(cell_line).strip()
        if cell_line in self.cell_line_expr.index:
            vals = self.cell_line_expr.loc[cell_line].values.astype(np.float32)
            return torch.from_numpy(vals)
        logger.warning(f"Cell line not found: {cell_line}")
        return torch.zeros(self.cell_dim, dtype=torch.float32)

    def get_atom_features(self, atom):
        if not RDKIT_AVAILABLE:
            return np.random.randn(self.atom_feature_dim)
        try:
            features = []
            atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']
            atom_type = atom.GetSymbol()
            features.extend([1 if atom_type == t else 0 for t in atom_types])
            features.append(atom.GetDegree())
            features.append(atom.GetFormalCharge())
            features.append(int(atom.GetChiralTag()))
            features.append(int(atom.GetIsAromatic()))
            features.append(atom.GetTotalNumHs())
            features.append(atom.GetMass() / 100.0)
            features.append(atom.GetAtomicNum())
            if len(features) < self.atom_feature_dim:
                features.extend([0] * (self.atom_feature_dim - len(features)))
            else:
                features = features[:self.atom_feature_dim]
            return np.array(features, dtype=np.float32)
        except Exception:
            return np.random.randn(self.atom_feature_dim)

    def smiles_to_graph(self, smiles):
        cache_key = f"graph_{hash(smiles)}"
        if cache_key in self.graph_cache:
            return self.graph_cache[cache_key]

        try:
            if RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError()

                node_features = [self.get_atom_features(atom) for atom in mol.GetAtoms()]
                edge_index = []
                for bond in mol.GetBonds():
                    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    edge_index += [[i, j], [j, i]]

                if not edge_index:
                    edge_index = [[0, 0]]

                x = torch.from_numpy(np.array(node_features, dtype=np.float32))
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            else:
                x = torch.randn(5, self.atom_feature_dim)
                edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        except Exception:
            x = torch.randn(5, self.atom_feature_dim)
            edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

        res = (edge_index, x)
        self.graph_cache[cache_key] = res
        return res

    def augment_molecular_data(self, data, node_mask_rate=0.10,
                               edge_drop_rate=0.10, feature_noise_std=0.02):

        edge_index, x = data
        x = x.clone()

        if node_mask_rate > 0 and x.size(0) > 1:
            mask = torch.rand(x.size(0)) > node_mask_rate
            x = x * mask.float().unsqueeze(1)

        if edge_drop_rate > 0 and edge_index.size(1) > 2:
            keep_mask = torch.rand(edge_index.size(1)) > edge_drop_rate
            if keep_mask.sum() == 0:
                keep_mask[0] = True
            edge_index = edge_index[:, keep_mask]

        if feature_noise_std > 0:
            x = x + torch.randn_like(x) * feature_noise_std

        return edge_index, x

    def process_sample(self, drug1, drug2, cell_line, augment=False):
        try:
            smiles1 = self.get_drug_smiles(drug1)
            smiles2 = self.get_drug_smiles(drug2)

            edge_index1, node_features1 = self.smiles_to_graph(smiles1)
            edge_index2, node_features2 = self.smiles_to_graph(smiles2)

            if augment:
                edge_index1, node_features1 = self.augment_molecular_data(
                    (edge_index1, node_features1)
                )
                edge_index2, node_features2 = self.augment_molecular_data(
                    (edge_index2, node_features2)
                )

            return {
                'graph1': (edge_index1, node_features1),
                'graph2': (edge_index2, node_features2),
                'target1': self.get_target_features(drug1),
                'target2': self.get_target_features(drug2),
                'physchem1': self.get_physchem_features(drug1),
                'physchem2': self.get_physchem_features(drug2),
                'cell_expr': self.get_cell_line_features(cell_line),
                'drug1_smiles': smiles1,
                'drug2_smiles': smiles2
            }
        except Exception as e:
            logger.error(f"Error processing {drug1}-{drug2}: {e}")
            return self._create_default_sample()

    def _create_default_sample(self):
        x = torch.randn(5, self.atom_feature_dim)
        edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        return {
            'graph1': (edge, x), 'graph2': (edge, x),
            'target1': torch.zeros(self.target_dim), 'target2': torch.zeros(self.target_dim),
            'physchem1': torch.zeros(self.physchem_dim), 'physchem2': torch.zeros(self.physchem_dim),
            'cell_expr': torch.zeros(self.cell_dim),
            'drug1_smiles': 'C', 'drug2_smiles': 'C'
        }