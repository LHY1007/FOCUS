import json
import csv

    
with open('/home/cbtil4/ST/datasets/Xenium_V1_hColon_Non_diseased_Base_FFPE_outs/gene_panel.json', 'r') as json_file:
    data = json.load(json_file)


names = []


if 'payload' in data:
    if 'targets' in data['payload']:
        for target in data['payload']['targets']:
            if 'type' in target and 'data' in target['type'] and 'name' in target['type']['data']:
                names.append({'name': target['type']['data']['name']})

with open('/home/cbtil4/ST/datasets/Xenium_V1_hColon_Non_diseased_Base_FFPE_outs/gene_groups.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['name'])
    csv_writer.writerows([[entry['name']] for entry in names])

print("名称已成功提取并保存到 output.csv 文件中")

# from scipy import sparse as sp_sparse
# a=sp_sparse.load_npz('/home/cbtil4/ST/datasets/Xenium5k_ovaryovariancancer/HR_ST/extract/20241218ovary/3_0/HR_ST_256.npz').toarray()
# print(a.shape)