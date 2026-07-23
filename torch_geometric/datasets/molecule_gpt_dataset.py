import gzip
import json
import multiprocessing
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from typing import Callable, List, Optional, Tuple

import numpy as np
import requests
import torch
from tqdm import tqdm

from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.io import fs
from torch_geometric.llm.models import LLM
from torch_geometric.utils import one_hot


def clean_up_description(description: str) -> str:
    description = description + " "

    if description.startswith("Pure "):
        description = description.replace("Pure ", "")
    if description.startswith("Mercurycombines"):
        description = description.replace("Mercurycombines",
                                          "Mercury combines")

    description = description.replace(
        "17-Hydroxy-6-methylpregna-3,6-diene-3,20-dione. ",
        "17-Hydroxy-6-methylpregna-3,6-diene-3,20-dione is ")

    description = description.replace("5-Thymidylic acid. ",
                                      "5-Thymidylic acid. is ")

    description = description.replace(
        "5'-S-(3-Amino-3-carboxypropyl)-5'-thioadenosine. ",
        "5'-S-(3-Amino-3-carboxypropyl)-5'-thioadenosine. is ")

    description = description.replace(
        ("Guanosine 5'-(trihydrogen diphosphate), monoanhydride"
         " with phosphorothioic acid. "),
        ("Guanosine 5'-(trihydrogen diphosphate), monoanhydride"
         " with phosphorothioic acid is "))

    description = description.replace("5'-Uridylic acid. ",
                                      "5'-Uridylic acid is ")

    description = description.replace("5'-Adenylic acid, ",
                                      "5'-Adenylic acid is ")

    description = description.replace(
        "Uridine 5'-(tetrahydrogen triphosphate). ",
        "Uridine 5'-(tetrahydrogen triphosphate). is ")

    description = description.replace("Inosine 5'-Monophosphate. ",
                                      "Inosine 5'-Monophosphate. is ")

    description = description.replace("Pivaloyloxymethyl butyrate (AN-9), ",
                                      "Pivaloyloxymethyl butyrate (AN-9) is ")

    description = description.replace(
        "4-Amino-5-cyano-7-(D-ribofuranosyl)-7H- pyrrolo(2,3-d)pyrimidine. ",
        "4-Amino-5-cyano-7-(D-ribofuranosyl)-7H- pyrrolo(2,3-d)pyrimidine is ")

    description = description.replace(
        "Cardamonin (also known as Dihydroxymethoxychalcone), ",
        "Cardamonin (also known as Dihydroxymethoxychalcone) is ")

    description = description.replace("Lithium has been used to treat ",
                                      "Lithium is ")

    description = description.replace("4,4'-Methylenebis ",
                                      "4,4'-Methylenebis is ")

    description = description.replace(
        "2,3,7,8-Tetrachlorodibenzo-p-dioxin",
        "2,3,7,8-Tetrachlorodibenzo-p-dioxin is ")

    description = description.replace("Exposure to 2,4,5-trichlorophenol ",
                                      "2,4,5-Trichlorophenol exposure ")

    index = 0
    L = len(description)
    if description.startswith('C.I. '):
        start_index = len('C.I. ')
    elif description.startswith('Nectriapyrone. D '):
        start_index = len('Nectriapyrone. D ')
    elif description.startswith(
            'Salmonella enterica sv. Minnesota LPS core oligosaccharide'):
        start_index = len(
            'Salmonella enterica sv. Minnesota LPS core oligosaccharide')
    else:
        start_index = 0
    for index in range(start_index, L - 1):
        if index < L - 2:
            if description[index] == '.' and description[
                    index + 1] == ' ' and 'A' <= description[index + 2] <= 'Z':
                break
        elif index == L - 2:
            break

    first_sentence = description[:index + 1]
    return first_sentence


def extract_name(
    name_raw: str,
    description: str,
) -> Tuple[Optional[str], str, str]:
    first_sentence = clean_up_description(description)

    splitter = '  --  --  '
    if ' are ' in first_sentence or ' were ' in first_sentence:
        replaced_words = 'These molecules'
    else:
        replaced_words = 'This molecule'

    first_sentence = first_sentence.replace(' is ', splitter)
    first_sentence = first_sentence.replace(' are ', splitter)
    first_sentence = first_sentence.replace(' was ', splitter)
    first_sentence = first_sentence.replace(' were ', splitter)
    first_sentence = first_sentence.replace(' appears ', splitter)
    first_sentence = first_sentence.replace(' occurs ', splitter)
    first_sentence = first_sentence.replace(' stands for ', splitter)
    first_sentence = first_sentence.replace(' belongs to ', splitter)
    first_sentence = first_sentence.replace(' exists ',
                                            splitter)  # only for CID=11443
    first_sentence = first_sentence.replace(' has been used in trials ',
                                            splitter)
    first_sentence = first_sentence.replace(' has been investigated ',
                                            splitter)
    first_sentence = first_sentence.replace(' has many uses ', splitter)

    if splitter in first_sentence:
        extracted_name = first_sentence.split(splitter, 1)[0]
    elif first_sentence.startswith(name_raw):
        extracted_name = name_raw
    elif name_raw in first_sentence:
        extracted_name = name_raw
        extracted_name = None
        print("=====", name_raw)
        print("first sentence: ", first_sentence)
    else:
        extracted_name = None

    if extracted_name is not None:
        extracted_description = description.replace(extracted_name,
                                                    replaced_words)
    else:
        extracted_description = description

    return extracted_name, extracted_description, first_sentence


class MoleculeGPTDataset(InMemoryDataset):
    description_url = (
        'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/annotations/'
        'heading/json?heading_type=Compound&heading=Record+Description&page={}'
    )
    compound_url = ('https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/'
                    'CURRENT-Full/SDF')

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        total_page_num: int = 10,
        total_block_num: int = 1,
        num_units: int = -1,
    ):
        self.total_page_num = total_page_num
        self.total_block_num = total_block_num
        self.num_units = num_units

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        pass

    @property
    def processed_file_names(self) -> List[str]:
        pass

    def download(self) -> None:
        step1_folder = f"{self.raw_dir}/step_01_PubChemSTM_description"
        if not os.path.exists(step1_folder):
            os.makedirs(step1_folder)
            valid_CID_set = set()
            CID2name_raw, CID2name_extracted = defaultdict(list), defaultdict(
                list)
            CID2text_raw, CID2text_extracted = defaultdict(list), defaultdict(
                list)

            for page_index in tqdm(range(self.total_page_num)):
                page_num = page_index + 1
                f_out = open(
                    f"{step1_folder}/Compound_description_{page_num}.txt", "w")

                description_data = requests.get(
                    self.description_url.format(page_num)).json()

                description_data = description_data["Annotations"]
                assert description_data["Page"] == page_num

                record_list = description_data["Annotation"]

                for record in record_list:
                    try:
                        CID = record["LinkedRecords"]["CID"][0]
                        if "Name" in record:
                            name_raw = record["Name"]
                            CID2name_raw[CID].append(name_raw)
                        else:
                            name_raw = None

                        data_list = record["Data"]
                        for data in data_list:
                            description = data["Value"]["StringWithMarkup"][0][
                                "String"].strip()

                            extracted_name, extracted_description, _ = extract_name(  # noqa: E501
                                name_raw, description)
                            if extracted_name is not None:
                                CID2name_extracted[CID].append(extracted_name)

                            CID2text_raw[CID].append(description)
                            CID2text_extracted[CID].append(
                                extracted_description)

                            valid_CID_set.add(CID)
                            f_out.write(f"{CID}\n")
                            f_out.write(f"{extracted_description}\n\n")
                    except Exception:
                        continue

            valid_CID_list = sorted(list(valid_CID_set))
            print(f"Total CID (with raw name) {len(CID2name_raw)}")
            print(f"Total CID (with extracted name) {len(CID2name_extracted)}")
            print(f"Total CID {len(valid_CID_list)}")

            with open(f"{self.raw_dir}/CID2name_raw.json", "w") as f:
                json.dump(CID2name_raw, f)

            with open(f"{self.raw_dir}/CID2name.json", "w") as f:
                json.dump(CID2name_extracted, f)

            with open(f"{self.raw_dir}/CID2text_raw.json", "w") as f:
                json.dump(CID2text_raw, f)

            with open(f"{self.raw_dir}/CID2text.json", "w") as f:
                json.dump(CID2text_extracted, f)

        step2_folder = f"{self.raw_dir}/step_02_PubChemSTM_SDF"
        if not os.path.exists(step2_folder):
            for block_id in tqdm(range(self.total_block_num)):
                block_size = 500000
                l_id = block_id * block_size + 1
                r_id = (block_id + 1) * block_size

                compound_file_name = f"Compound_{l_id:09d}_{r_id:09d}.sdf.gz"
                download_url(f"{self.compound_url}/{compound_file_name}",
                             step2_folder)

    def process(self, use_mp: bool = False) -> None:
        try:
            from rdkit import Chem
            from rdkit.Chem.rdchem import BondType as BT
            WITH_RDKIT = True

        except ImportError:
            WITH_RDKIT = False

        if not WITH_RDKIT:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = fs.torch_load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.save(data_list, self.processed_paths[0])
            return

        step2_folder = f"{self.raw_dir}/step_02_PubChemSTM_SDF"
        step3_folder = f"{self.raw_dir}/step_03_PubChemSTM_filtered"
        if not os.path.exists(step3_folder):
            os.makedirs(step3_folder)
            with open(f"{self.raw_dir}/CID2text.json") as f:
                CID2text = json.load(f)
            target_CID_list = set(CID2text.keys())

            block_size = 500000

            def extract_one_SDF_file(block_id: int) -> None:
                valid_mol_count = 0

                writer = Chem.SDWriter(
                    f'{step3_folder}/filtered_{block_id}.sdf')
                l_id = block_id * block_size + 1
                r_id = (block_id + 1) * block_size

                compound_file_name = f"Compound_{l_id:09d}_{r_id:09d}.sdf.gz"
                gzip_loader = gzip.open(f"{step2_folder}/{compound_file_name}")
                suppl = Chem.ForwardSDMolSupplier(gzip_loader)

                for mol in tqdm(suppl):
                    if mol is None:
                        continue
                    cid = mol.GetProp("PUBCHEM_COMPOUND_CID")

                    if cid not in target_CID_list:
                        continue

                    writer.write(mol)
                    valid_mol_count += 1

                writer.close()
                print(f"block id: {block_id}\nfound {valid_mol_count}\n\n")
                sys.stdout.flush()
                return

            if use_mp:
                num_process = multiprocessing.cpu_count()
                print(f"{num_process} CPUs")
                num_process = 8
                p = Pool(num_process)

                block_id_list = np.arange(self.total_block_num)
                with p:
                    p.map(extract_one_SDF_file, block_id_list)
            else:
                for block_id in range(self.total_block_num):
                    extract_one_SDF_file(block_id)

        with open(f"{self.raw_dir}/CID2text.json") as f:
            CID2text = json.load(f)
        target_CID_list = set(CID2text.keys())
        print(f'The length of target_CID_list: {len(target_CID_list)}')

        writer = Chem.SDWriter(f'{self.raw_dir}/molecules.sdf')

        found_CID_set = set()
        for block_id in range(self.total_block_num + 1):
            compound_file_path = f"{step3_folder}/filtered_{block_id}.sdf"
            try:
                suppl = Chem.SDMolSupplier(compound_file_path)

                for mol in tqdm(suppl):
                    writer.write(mol)
                    cid = mol.GetProp("PUBCHEM_COMPOUND_CID")
                    found_CID_set.add(cid)
            except Exception:
                print(f"block id: {block_id} with 0 valid SDF file")
                continue

        writer.close()
        print(f"In total: {len(found_CID_set)} molecules")

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'Unknow': 5}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        data_list = []
        CID2text_file = f'{self.raw_dir}/CID2text.json'

        with open(CID2text_file) as f:
            CID2text_data = json.load(f)

        suppl = Chem.SDMolSupplier(f'{self.raw_dir}/molecules.sdf')

        llm = LLM(
            model_name='Qwen/Qwen3-0.6B',
            num_params=1,
            dtype=torch.bfloat16,
            sys_prompt='You are an agent, answer my questions.',
        )
        prompt = ("Propose a question regarding the molecule '∼' "
                  "whose answer is: {}:")
        for mol in tqdm(suppl):
            if mol.HasProp('PUBCHEM_COMPOUND_CID'):
                CID = mol.GetProp("PUBCHEM_COMPOUND_CID")
                CAN_SMILES = mol.GetProp("PUBCHEM_SMILES")

                m: Chem.Mol = Chem.MolFromSmiles(CAN_SMILES)
                if m is None:
                    continue
                RDKit_CAN_SMILES = Chem.MolToSmiles(m)

                ground_truth = CID2text_data[CID][0]

                instruction = llm.inference([prompt.format(ground_truth)])[0]

                x: torch.Tensor = torch.tensor([
                    types[atom.GetSymbol()] if atom.GetSymbol() in types else 5
                    for atom in m.GetAtoms()
                ])
                x = one_hot(x, num_classes=len(types), dtype=torch.float)

                rows, cols, edge_types = [], [], []
                for bond in m.GetBonds():
                    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    edge_types += [bonds[bond.GetBondType()]] * 2
                    rows += [i, j]
                    cols += [j, i]

                edge_index = torch.tensor([rows, cols], dtype=torch.long)
                edge_type = torch.tensor(edge_types, dtype=torch.long)
                edge_attr = one_hot(edge_type, num_classes=len(bonds))

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    smiles=RDKit_CAN_SMILES,
                    instruction=instruction,
                    y=ground_truth,
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

                if self.num_units > 0 and len(data_list) >= self.num_units:
                    break

        self.save(data_list, self.processed_paths[0])
