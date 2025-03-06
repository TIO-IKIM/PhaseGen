import torch
from glob import glob
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

    
def get_files(input_path, save_path):
    gen_data_list = glob(f"{save_path}/*.pt")
    gen_data_list.sort()
    data_list = glob(f"{input_path}/*.pt")
    data_list.sort()
    
    return data_list, gen_data_list

def process_file(file, gen_data_list, save_path):
    filename = Path(file).stem
    gen_file = [gen_file for gen_file in gen_data_list if filename in gen_file][0]
    data = torch.load(file, weights_only=True)
    gen_data = torch.load(gen_file, weights_only=True)
    while isinstance(gen_data, dict) is True:
        gen_data = gen_data["kspace"]
    gen_data_new = { 
        'kspace': gen_data,
        'reconstruction_rss': data['reconstruction_rss'],
    }
    torch.save(gen_data_new, f"{save_path}/{filename}.pt")
        
if __name__ == "__main__":
    
    input_path = "/home/jovyan/radiology/fastmri_knee/singlecoil_train_pt"
    save_path = "/home/jovyan/radiology/fastmri_knee/singlecoil_train_pt_naive"
    
    data_list, gen_data_list = get_files(
        input_path=input_path,
        save_path=save_path
    )
    with tqdm(total=len(data_list)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_file, file, gen_data_list, save_path) for file in data_list]
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)