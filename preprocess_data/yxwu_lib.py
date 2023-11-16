import os
import math
import numpy as np
import glob

from timeit import default_timer as timer
from datetime import timedelta


# pdb_prep

def dry_reduce(folder_name, file_name):
    '''
    Use the --dry flag to remove crystallographic waters and the --reduce flag to add hydrogen atoms in their optimal locations with pdb4amber.
    '''
    print(f'Remove WAT and add H in `{folder_name}`...\n')

    start_time = timer()

    result=os.system(f'pdb4amber -i {file_name}.pdb -o {file_name}_dry_H.pdb --dry --reduce')
    if result != 0:
        raise RuntimeError('The simulation is broken.')
    finish_time = timer()
    elapsed_time = timedelta(seconds=finish_time - start_time)

    # report elapsed time
    msg = f'\nProcess finished. Time cost {elapsed_time}.'
    print(msg)

    print("-" * 80)

    return {
        'folder': folder_name,
        'file': str(file_name),
        'msg': msg,
        'time': str(elapsed_time)
    }
        

# lig_del

def read_pdb(file_path):
    print(f"Reading pdb from {file_path}...")
    '''
    read pdb file from given path
    '''
    file = open(file_path, 'r')
    lines = file.readlines()
    return lines


def find_lig(lines, file_name, lig_name):
    print(f"Removing {lig_name} info from {file_name}...")
    '''
    remove ligand info from given content and write in new pdb
    '''
    with open(f"{file_name}_del_{lig_name}.pdb", "w") as f:
        for line in lines:
            if f"{lig_name}" not in line:
                f.write(line)
    f.close()

    print(f"{lig_name} info has been removed and a new protein pdb has been created")


# cat_protein_lig

def cat_pdbs(folder, protein_pdb, lig_pdb, cat_pdb):
    '''
    Combine the protein and ligand PDB files using a cat command
    '''
    print(f"Combine {protein_pdb} and {lig_pdb} in {folder}...\n")

    start_time = timer()

    result=os.system(f'cat {protein_pdb}.pdb {lig_pdb}.pdb > {cat_pdb}.pdb')
    if result != 0:
        raise RuntimeError('The simulation is broken.')
    finish_time = timer()
    elapsed_time = timedelta(seconds=finish_time - start_time)

    # report elapsed time
    msg = f'\nProcess finished. Time cost {elapsed_time}.'
    print(msg)

    print("-" * 80)

    return {
        'folder': folder,
        'file': str(cat_pdb),
        'msg': msg,
        'time': str(elapsed_time)
    }


# pdb_prep_routine

def prep_pipeline(file_names):

    cwrk = os.path.abspath(os.getcwd())

    '''
    Remove WAT and add hydrogen atoms
    '''
    for file_name in file_names:
        folder_one = f'/home/yxwu/SHP2/3rd_MD/{file_name}/tleap'
        os.chdir(folder_one)
        dry_reduce(folder_one, file_name)
        os.chdir(cwrk)

        print("*" * 80)

        '''
        read pdb file from given path and remove ligand info for a new pdb
        '''

        file = f'/home/yxwu/SHP2/3rd_MD/{file_name}/tleap/{file_name}_dry_H.pdb'
        data = read_pdb(file)
        os.chdir(f'/home/yxwu/SHP2/3rd_MD/{file_name}/pdb_prep')
        lig_name = "5OD"
        find_lig(data, file_name, lig_name)
        os.chdir(cwrk)

        print("*" * 80)

        '''
        combine the protein and ligand PDB files
        '''
        if 'unbound' not in file_name:
            folder_two = f"/home/yxwu/SHP2/3rd_MD/{file_name}/pdb_prep"

            os.chdir(folder_two)
            os.system(f"cp /home/yxwu/SHP2/3rd_MD/{file_name}/tleap/Shp099_new.pdb ./")

            protein = f"{file_name}_del_5OD"
            lig = "Shp099_new"
            cat_pdb = f"{file_name}_del_5OD_cat"

            cat_pdbs(folder_two, protein, lig, cat_pdb)
            os.chdir(cwrk)

            print("*" * 80)


# add_ions_calculation

def read_log(file_path):
    '''
    read leap.log data from given path
    '''
    file = open(file_path, 'r')
    lines = file.readlines()
    return lines


def find_volume(lines):
    '''
    find volume data from given content
    '''
    volumes = []
    for line in lines:
        if 'Volume:' in line:
            data = line.split(" ")
            value = float(data[3])
            volumes.append(value)
            print(line)
    return volumes

def find_ions(lines):
    '''
    find ions required to neutralize the system
    '''
    sodium_ions_required = []
    for line in lines:
        if "Na+ ion required to neutralize." in line or "Na+ ions required to neutralize." in line:
            data = line.split(" ")
            value = int(data[0])
            sodium_ions_required.append(value)
            print(line)
    return sodium_ions_required


def compute_add_ions(volume, desired_concentration):
    '''
    Convert the volume of the systems in  A^3 to liters
    '''
    conv_vol = volume * math.pow(math.pow(10, 2), 3) / (math.pow(math.pow(10, 10), 3) * math.pow(10, 3))
    print('Volume = {} L'.format(conv_vol))

    '''
    Determine how many chloride ions are present in one liter of solution at a certain concentration
    '''
    # desired_conc = float(input("Desired Concentration in mM:"))
    chloride_ions_present = desired_concentration * 6.022 * math.pow(10, 23) / math.pow(10, 3)

    '''
    Determine how many chloride ions are needed in the system
    '''
    chloride_ions_needed = round(conv_vol * chloride_ions_present)

    '''
    Determine the number of sodium ions needed
    '''
    sodium_ions_needed = chloride_ions_needed

    print("# Na+ = # Cl- = {}".format(sodium_ions_needed))
    return chloride_ions_needed, sodium_ions_needed


def compute_overall_add_ions(sodium_ions_required, chloride_ions_needed, sodium_ions_needed):
    '''
    Compute overall Na+ and Cl- ions needed to add
    '''
    sodium_ions_add = sodium_ions_needed + sodium_ions_required
    chloride_ions_add = chloride_ions_needed

    addIons_command = "addIons2 mol Na+ {} Cl- {} ".format(sodium_ions_add, chloride_ions_add)

    print(addIons_command)
    return addIons_command


def addIons_pipeline(log_file, desired_concentration):
    print('*' * 80)
    print("current computiing: {}".format(log_file))
    data = read_log(log_file)
    volumes = find_volume(data)
    last_volume = volumes[-1]
    sodium_ions = find_ions(data)
    last_sodium_ions = sodium_ions[-1]
    add_ions = compute_add_ions(last_volume, desired_concentration)
    addIons_command = compute_overall_add_ions(last_sodium_ions, add_ions[0], add_ions[1])

    print()
    return addIons_command


# balanced_ion_tleap_script

def balanced_script(file_names, desired_concentration):
    print(f"Writing balanced tleap script with add_ions calculation results...")

    for file_name in file_names:
        '''
        write balanced tleap input file in given path
        '''
        log_path = f"/home/yxwu/SHP2/3rd_MD/{file_name}/pdb_prep/leap.log"

        addIons_command = addIons_pipeline(log_path, desired_concentration)

        print("-" * 80)
        
        folder_path = f"/home/yxwu/SHP2/3rd_MD/{file_name}/pdb_prep"

        lig_file = "Shp099_gaff2"

        cwrk = os.path.abspath(os.getcwd())

        if 'unbound' not in file_name:
            script = [
                "source leaprc.water.tip3p", "\n",
                "source leaprc.protein.ff19SB", "\n",
                "source leaprc.gaff2", "\n",
                f"loadamberparams {lig_file}.frcmod", "\n",
                f"5OD=loadmol2 {lig_file}.mol2", "\n",
                f"mol =loadpdb {file_name}_del_5OD_cat.pdb", "\n",
                f"{addIons_command}", "\n",
                "solvateOct mol TIP3PBOX 8.0", "\n",
                f"saveamberparm mol {file_name}_del_50D_gaff2.prmtop {file_name}_del_50D_gaff2.inpcrd", "\n",
                f"savepdb mol {file_name}_del_50D_gaff2_H.pdb", "\n",
                "quit"
            ]

        else:
            script = [
                "source leaprc.water.tip3p", "\n",
                "source leaprc.protein.ff19SB", "\n",
                "source leaprc.gaff2", "\n",
                f"loadamberparams {lig_file}.frcmod", "\n",
                f"5OD=loadmol2 {lig_file}.mol2", "\n",
                f"mol =loadpdb {file_name}_del_5OD.pdb", "\n",
                f"{addIons_command}", "\n"
                "solvateOct mol TIP3PBOX 8.0", "\n",
                f"saveamberparm mol {file_name}_del_50D_gaff2.prmtop {file_name}_del_50D_gaff2.inpcrd", "\n",
                f"savepdb mol {file_name}_del_50D_gaff2_H.pdb", "\n",
                "quit"
            ]


        script_name = f"balanced_{file_name}_tleap"

        os.chdir(folder_path)
        write_script(folder_path, script_name, script)
        os.chdir(cwrk)

        print("*" * 80)


# md_prep

def md_input_prep(file_name, file):
    '''
    Copy MD input files to destinated folder
    '''
    location = f"/home/yxwu/SHP2/1st_MD/{file_name}/md/{file}"
    destination = f"/home/yxwu/SHP2/3rd_MD/{file_name}/MD"

    print(f"Copying {location} to {destination}...")
    os.system(f"cp {location} {destination}")
    print("Copying finished.")


# run_tleap

def start_tleap(folder_name, script):
    print(f'Run tleap in `{folder_name}`...\n')

    start_time = timer()

    result=os.system(f'tleap -s -f {script}')
    if result != 0:
        raise RuntimeError('The simulation is broken.')
    finish_time = timer()
    elapsed_time = timedelta(seconds=finish_time - start_time)

    # report elapsed time
    msg = f'\nRun finished. Time cost {elapsed_time}.'
    print(msg)
    return {
        'folder': folder_name,
        'script': str(script),
        'msg': msg,
        'time': str(elapsed_time)
    }


# write_tleap_script

def write_script(folder_path, script_name, script):
    print(f"Writing tleap script in {folder_path}...")
    '''
    write tleap input file in given path
    '''
    with open(f"{script_name}.in", "w") as f:
        f.writelines(script)
    f.close()

    print(f"{script_name}.in has been created and saved")


# tleap_prep_routine

def tleap_pipeline(file_names):

    cwrk = os.path.abspath(os.getcwd())

    '''
    prepare tleap input file
    '''

    for file_name in file_names:
        folder_path = f"/home/yxwu/SHP2/3rd_MD/{file_name}/pdb_prep"

        lig_file = "Shp099_gaff2"

        if 'unbound' not in file_name:
            script = [
                "source leaprc.water.tip3p", "\n",
                "source leaprc.protein.ff19SB", "\n",
                "source leaprc.gaff2", "\n",
                f"loadamberparams {lig_file}.frcmod", "\n",
                f"5OD=loadmol2 {lig_file}.mol2", "\n",
                f"mol =loadpdb {file_name}_del_5OD_cat.pdb", "\n",
                "addIons2 mol Na+ 0", "\n",
                "addIons2 mol Cl- 0", "\n",
                "solvateOct mol TIP3PBOX 8.0", "\n",
                f"saveamberparm mol {file_name}_del_50D_gaff2.prmtop {file_name}_del_50D_gaff2.inpcrd", "\n",
                f"savepdb mol {file_name}_del_50D_gaff2_H.pdb", "\n",
                "quit"
            ]

        else:
            script = [
                "source leaprc.water.tip3p", "\n",
                "source leaprc.protein.ff19SB", "\n",
                "source leaprc.gaff2", "\n",
                f"loadamberparams {lig_file}.frcmod", "\n",
                f"5OD=loadmol2 {lig_file}.mol2", "\n",
                f"mol =loadpdb {file_name}_del_5OD.pdb", "\n",
                "addIons2 mol Na+ 0", "\n",
                "addIons2 mol Cl- 0", "\n",
                "solvateOct mol TIP3PBOX 8.0", "\n",
                f"saveamberparm mol {file_name}_del_50D_gaff2.prmtop {file_name}_del_50D_gaff2.inpcrd", "\n",
                f"savepdb mol {file_name}_del_50D_gaff2_H.pdb", "\n",
                "quit"
            ]
            

        script_name = f"{file_name}_tleap"
        
        os.chdir(folder_path)
        os.system(f"cp /home/yxwu/SHP2/3rd_MD/{file_name}/tleap/{lig_file}.frcmod ./")
        os.system(f"cp /home/yxwu/SHP2/3rd_MD/{file_name}/tleap/{lig_file}.mol2 ./")
        write_script(folder_path, script_name, script)
        os.chdir(cwrk)

        print("*" * 80)

        '''
        run tleap
        '''

        folder_path = f"/home/yxwu/SHP2/3rd_MD/{file_name}/pdb_prep"

        script_run = f"{file_name}_tleap.in"

        os.chdir(folder_path)
        start_tleap(folder_path, script_run)
        os.chdir(cwrk)
        
        print("*" * 160)


# write_MD_script

def read_file(ref_file_path):
    print(f"Reading file from {ref_file_path}...")
    '''
    Read in file from given path
    '''

    file = open(ref_file_path, 'r')
    lines = file.readlines()

    return lines


def modi_script(lines, folder_path, script_name, content, new_content):
    print(f"Writing new script in {folder_path}...")
    '''
    Modify MD script with replacing the target string
    '''
    with open(f"{script_name}", "w") as f:
        for line in lines:
            new_script = line.replace(f"{content}", f"{new_content}")
            f.write(new_script)
    f.close()

    print(f"New script in {folder_path} has been created and saved.")


def find_intValue(lines, identifier, index):
    '''
    find specific data for identifier from given content
    '''
    for line in lines:
        if f'{identifier}' in line:
            line_strip = line.strip()
            datas = line_strip.split(" ")
            # print(datas)
            
            valid_data = []
            for data in datas:
                if data:
                    valid_data.append(data)
            # print(valid_data)

            index_num = int(f'{index}')
            value = int(valid_data[index_num])
            # print(line)
    return value


def find_floatValue(lines, identifier, index):
    '''
    find specific data for identifier from given content
    '''
    for line in lines:
        if f'{identifier}' in line:
            line_strip = line.strip()
            datas = line_strip.split(" ")
            # print(datas)
            
            valid_data = []
            for data in datas:
                if data:
                    valid_data.append(data)
            # print(valid_data)

            index_num = int(f'{index}')
            value = float(valid_data[index_num])
            # print(line)
    return value


def read_dat(file):
    print(f"Reading data from {file}...")
    '''
    Read data from given path
    '''
    data = []
    with open(f'{file}', 'r') as f:
        reader = f.readlines()
        for line in reader:
            row = line.strip()
            get_col = row.split()
            data.append([float(i) for i in get_col])
    return data


def accuracy(predict, target):
    equal_mask = predict == target
    acc = np.mean(equal_mask.astype(np.float32))
    return acc


def get_protein(path, delimiter):
    os.chdir(path)
    dirs = glob.glob(delimiter)

    proteins = []
    for dir in dirs:
        protein = dir.split(".")
        proteins.append(protein[0])
    
    return proteins