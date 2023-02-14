import os # Importa os, que permite passar comandos para o sistema
import sys
import re
import math
import time
import pymol
import shutil
import joblib
import pyrosetta
import numpy as np # Importa numpy, que é uma biblioteca voltada a cálculos matemáticos
import seaborn as sns  # Importa o seaborn, outra biblioteca para visualização de dados
import pandas as pd # Importa pandas, que é uma biblioteca para trabalhar com dados tabelados
import matplotlib.pyplot as plt # Importa o matplotlib.pyplot, que é uma biblioteca para visualização de dados
from multiprocessing import Process, Queue
from pymol import stored


# PARÂMETROS IMPORTANTES ----------------------------------------------
nsims = 3         # Número de dockings por peptídeo (recomendável 50)
pop_size = 10     # Número de peptídeos a serem gerados
iterations = 100  # Número de iterações que deverão ser feitas
# ---------------------------------------------------------------------

pyrosetta.init("-seed_offset 1111")

# Essa função é para ler o arquivo mais rápido e com menos memória
def read_file(filename):
    for line in open(filename):
        yield line

# Esta função vai ordenar os textos corretamente como um humano ordenaria: 0, 1, 2, 3, 4, ..., 10, 11, 12, 13... e não 0, 1, 10, 11,12,...., 2, 3, 4, 5...
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

# Esta função serve para criar um lote de peptídeos para serem processados pelo Roseta em paralelo
### Ela não é usada neste script
def makeBatch(population, cores, add_fake = False):
    # population é a lista de strings representando as sequências dos peptídeos
    # cores é o número de processadores usados para fazer o docking.
    # O add_fake é porque o Rosetta algumas vezes ignora o primeiro peptídeo, por isso adicionei o BJOUXZ ao início 
    batches = list()
    x = int(len(population)/cores)
    if len(population) < cores:
        return [population]
    
    for i in range(x):
        if add_fake == True:
            new_batch = ["BJOUXZ"] # Este é um "peptídeo" falso para ser ignorado pelo Rosetta
        else:
            new_batch = list()
        for j in range(cores):
            pos = i*cores + j
            new_batch.append(population[pos])
        batches.append(new_batch)
        
    if len(population)%cores != 0: # If true, there is an excess of peptides not covered yet
        if add_fake == True:
            final_batch = ["BJOUXZ"]
        else:
            final_batch = list()
        
        for k in range(len(population)%cores):
            pos = x*cores + k
            final_batch.append(population[pos])
        batches.append(final_batch)
    return batches

# Esta função é utilizada para criar estruturas de peptídeos com torções e conformações aleatórias 
# Ela também cria o complexo peptídeo-alvo para realização do docking
def makePeptide(sequence, gen=0):
    # sequence: sequência do peptídeo, como string
    # gen: número da iteração em que o peptídeo será criado
    if not os.path.isdir(f"Inputs/I{gen}/"):
        os.mkdir(f"Inputs/I{gen}/")
    peptide = pyrosetta.pose_from_sequence(sequence)
    angles = np.random.rand(3,len(sequence))*360 - 180
    for i in range(len(sequence)):
        peptide.set_phi(i+1, angles[0,i])
        peptide.set_psi(i+1, angles[1,i])
        try:
            peptide.set_chi(1, i+1, angles[2,i])  
        except:
            pass
    src = f"Temp/{sequence}.pdb"
    peptide.dump_pdb(src)
    dst = f"Inputs/I{gen}/{sequence}.pdb"
    pymol.cmd.load("template.pdb", "template") # Aqui carregamos a estrutura do template (neste caso a ACE2)
    pymol.cmd.load(src, "new")
    pymol.cmd.align("new", "template") # Aqui alinhamos o novo peptídeo criado com o template, para posicionar no sítio 
    pymol.cmd.save(src, selection = "new")
    pymol.cmd.reinitialize()
    pymol.cmd.load("target.pdb") # Aqui carregamos a estrutura do alvo em PDB
    pymol.cmd.load(src)
    pymol.cmd.alter('chain A', 'chain="B"') # É necessário trocar esses nomes, pois o Rosetta reconhece sempre cadeia A como
    pymol.cmd.alter('chain E', 'chain="A"') # receptora e cadeia B como peptídeo
    pymol.cmd.save(dst) # Salva arquivo do complexo alvo-peptídeo
    os.remove(src)
    pymol.cmd.reinitialize()
    return dst

# Esta função é para contar os contatos únicos que são realizados 
# O objetivo é calcular o percentual dos resíduos do sítio de ligação que estão "ocupados"
def countContacts(peptide_file, target = None, cutoff = 6.0, count_on = 'target'):
    # peptide_file é o arquivo PDB do complexo peptídeo-alvo depois do docking
    # Target é uma lista como a mostrada abaixo contendo os resíduos do sítio de ligação da proteína alvo
    # O parâmetro count_on pode ser ou target ou peptide e indica em qual dos dois serão contados os contatos
    if target == None:
        target = [('417', 'LYS'), ('446', 'GLY'), ('449', 'TYR'), ('453', 'TYR'), 
                  ('455', 'LEU'), ('456', 'PHE'), ('475', 'ALA'), ('476', 'GLY'),
                  ('477', 'SER'), ('486', 'PHE'), ('487', 'ASN'), ('489', 'TYR'),
                  ('490', 'PHE'), ('493', 'GLN'), ('494', 'SER'), ('495', 'TYR'),
                  ('496', 'GLY'), ('498', 'GLN'), ('500', 'THR'), ('501', 'ASN'),
                  ('502', 'GLY'), ('503', 'VAL'), ('505', 'TYR')] # ALterar para a RBM da Ana
    pymol.cmd.reinitialize()
    stored.list = []
    pymol.cmd.load(peptide_file)
    if count_on == 'target':
        pymol.cmd.indicate(''.join(['chain A within ', str(cutoff), ' of chain B']))
        pymol.cmd.iterate('indicate', 'stored.list.append((resi,resn))')
        pymol.cmd.delete('indicate')
        count = 0
        for aa in pd.unique(stored.list):
            if aa in target:
                count += 1
        occupancy = count/len(target)
        return occupancy
    elif count_on == 'peptide':
        pymol.cmd.indicate(''.join(['chain B within ', str(cutoff), ' of chain A']))
        pymol.cmd.iterate('indicate', 'stored.list.append((resi,resn))')
        pymol.cmd.delete('indicate')
        return pd.unique(stored.list)
    

# Esta é a função que realiza o docking em paralelo. Ela faz um docking por processador ao mesmo tempo.
def calculateFitness_Parallel(candidate, protocol, scorefxn, gen, reps, queue):
    best_pose = None
    best_fitness = 999999999
    scores = list() 
    for i in range(reps):
        # Carrega o complexo
        comp = pyrosetta.pose_from_pdb("Inputs/I" + str(gen) + "/comp(" + candidate + ").pdb")
        # Aplica o protocolo de docking e depois calcula o score
        protocol.apply(comp)
        fitness = scorefxn(comp)
        scores.append(fitness)
        if fitness < best_fitness:
            best_pose = comp.clone()
            best_fitness = fitness
    # Salva a melhor pose como PDB para ficar de registro
    best_pose.dump_pdb("Results/G" + str(gen) + "/dock(" + candidate + ").pdb")
    queue.put([candidate, scores])
    
    #Aqui só é para escrever no arquivo de registro
    scores_txt = [str(s) for s in scores]
    scores_txt = ",".join(scores_txt)
    p_info = open("Results/peptides_Potter.csv", mode="a")
    p_info.write("".join(["\n", str(gen), ",", candidate, scores_txt, ",", str(best_fitness)]))
    p_info.close()

# PepPlot é aquele plot que fizemos com as "colunas de peptídeos"
def PepPlot(data, ligcolumn, reccolumn, hue, palette, ax=None):
    receptors = list(natural_sort(pd.unique(data[reccolumn])))
    ligands = list(natural_sort(pd.unique(data[ligcolumn])))
    huetypes = list(pd.unique(data[hue]))
    figwidth = plt.gcf().get_figwidth()
    figheight = plt.gcf().get_figheight()
    dpi = plt.gcf().get_dpi()
    maxstack = data.groupby([reccolumn,ligcolumn]).count().groupby([reccolumn]).count().max()[0]
    maxwidth = figwidth*dpi/(3*len(receptors)*0.7)
    maxheight = figheight*dpi/maxstack
    if maxwidth < maxheight:
        maxsize = maxwidth
        top = maxwidth/maxheight
    else:
        top = 1
    print(top)
    maxsize = min([maxwidth,maxheight])
    ticks = np.linspace(0,1,len(receptors))
    yticks = np.linspace(0,top,maxstack)
    if ax==None:
        fig,ax = plt.subplots(figsize=(figwidth, figheight))
        ax.set_xlim(-0.2,1.2)
        ax.set_ylim(-0.2,1.2)
        ax.set_xticks(ticks,rotation=90)
        ax.set_xticklabels(receptors,rotation=90)
    else:
        ax.set_xticks(ticks)
        ax.set_xlim(-0.01,1.03)
        ax.set_ylim(-0.005,1.03)
        ax.set_xticklabels(receptors,rotation=90)
        i = 0
        for rec in receptors:
            j = 0
            used = []
            for lig in reversed(natural_sort(list(data.loc[data[reccolumn]==rec,ligcolumn]))):
                if lig not in used:
                    h = data.loc[(data[reccolumn]==rec)&(data[ligcolumn]==lig),hue].values[0]
                    k = huetypes.index(h)
                    c = palette[k]
                    ax.text(ticks[i]-0.01,yticks[j], lig.replace('_',''), fontsize=maxsize*0.9, color=c, fontname='monospace', fontweight='bold')
                    j += 1
                    used.append(lig)
            i += 1
            
        for i in range(len(huetypes)):
            ax.text(0.30*i+0.13, 1.05, huetypes[i],fontsize=maxsize*0.9, color=palette[i])

def random_variation(peptide, num_changes=1):
    n = len(peptide)
    pos = np.random.randint(n, size=num_changes)
    pep = list(peptide)
    for p in pos:
        res = peptide[p]
        pep[p] = np.random.choice(conservative_list[res])
    return ''.join(pep)  
    
def pre_check(nsims):
    if not os.path.isdir("Results/"):
        os.mkdir("Results/")
    if not os.path.isdir("Inputs/"):
        os.mkdir("Inputs/")
    if not os.path.isdir("Temp/"):
        os.mkdir("Temp/")
    if not os.path.isdir("Results/Docking/"):
        os.mkdir("Results/Docking/")
    if not os.path.isdir("Results/VTR/"):
        os.mkdir("Results/VTR/")
    if not os.path.isfile("Results/summary_Potter.csv"):
        summary = open("Results/summary_Potter.csv", mode="w+") # Create the summary csv file for later visualization of results
        summary.write("iteration,best peptide,score,%surface occupation,contacts,iteration time" + "\n") # The headers of summary
        summary.close()
    if not os.path.isfile("Results/peptides_Potter.csv"):
        p_info = open("Results/peptides_Potter.csv", mode="w+") # Create populations csv file for posterior checks
        p_info.write("iteration,sequence,") # The headers of summary
        for i in range(nsims):
            p_info.write(f"score_{i+1},")
        p_info.write("best,occupancy,contacts,hydrophobic,aromatic stacking,attractive,hydrogen bond,repulsive,salt bridge") 
        p_info.close()   


def start(pdb_file='6m0j.pdb', target_chain='E', ligand_chain='A', cutoff=12.0):
    pymol.cmd.reinitialize()
    # Checar se o arquivo que você indicou existe
    if os.path.isfile(pdb_file):
        pymol.cmd.load(pdb_file, 'base')
    else:
        pymol.cmd.fetch(pdb_file.split('.pdb')[0], 'base')
        
    # Seleciona todos os resíduos do receptor (neste caso ACE2) que estão a cutoff do nosso alvo (RBD)
    pymol.cmd.indicate(f'chain {ligand_chain} within {cutoff} of chain {target_chain}')
    peptide = pymol.cmd.get_fastastr(selection='indicate')
    pymol.cmd.save('template.pdb', selection ='indicate')
    pymol.cmd.indicate(f'chain {target_chain}')
    pymol.cmd.save('target.pdb', selection = 'indicate')
    pymol.cmd.reinitialize()
    return peptide.split('\n')[1]

def docking(protocol, scorefxn, sequence, gen=0, nsims=1):
    dst = f"Results/Docking/I{gen}/{sequence}.pdb"
    best_pose = None
    best_fitness = 999999999
    scores = list() 
    score_dict = {}
    for i in range(nsims):
    #       comp = pyrosetta.pose_from_pdb(f"Inputs/I0/{sequence}.pdb")
        comp = pyrosetta.pose_from_pdb(f"Inputs/I{gen}/{sequence}.pdb")
    # Aplica o protocolo de docking e depois calcula o score
        protocol.apply(comp)
        fitness = scorefxn(comp)
        scores.append(fitness)
        score_dict[f'{sequence}_{i}'] = fitness
        comp.dump_pdb(f'{dst[:-4]}_{i}.pdb')
        if fitness < best_fitness:
            best_pose = comp.clone()
            best_fitness = fitness
    # Salva a melhor pose como PDB para ficar de registro
    best_pose.dump_pdb(dst)
    scores_txt = [str(s) for s in scores]
    scores_txt = ",".join(scores_txt)
    p_info = open("Results/peptides_Potter.csv", mode="a")
    p_info.write("".join(["\n", str(gen), ",", sequence, scores_txt, ",", str(best_fitness)]))
    p_info.close()
    return score_dict

'''
Script: contatos.py
Versão: 0.1
Função: Script simples para cálculos de contatos
Autor: @dcbmariano
Data: 2022
'''
# -----------------------------------------------------------------------------
# 0. DEFINIÇÕES
# -----------------------------------------------------------------------------

# INPUT: arquivo PDB de entrada
def VTR(entrada, outfile):
    mostrar_contato = {
                        'ligacao_hidrogenio': True,
                        'hidrofobico': True,
                        'aromatico': True,
                        'repulsivo': True,
                        'atrativo': True,
                        'ponte_salina': True
                        }

    # OUTPUT: saida = tela | csv
    saida = 'csv'

    # -----------------------------------------------------------------------------
    # Definições padrão do sistema
    # -----------------------------------------------------------------------------

    if saida == 'csv':
        w = open(outfile,'w')
        w.write('CONTACT;RES1:ATOM;RES2:ATOM;DIST\n') # cabeçalho do CSV

    # CONTATOS (baseado na definição do nAPOLI) 
    # tipo = (distancia_minima, distancia_maxima)
    aromatic = (2, 4)
    hidrogen_bond = (0, 3.9)
    hidrophobic = (2, 4.5)
    repulsive = (2, 6)
    atractive = (2, 6)
    salt_bridge = (0, 3.9)


    # REGRAS
    # 1 - deve ser feito por átomos de resíduos diferentes
    # 2 - aromatic = aromatic + aromatic
    # 3 - hb => aceptor + donor
    # 4 - hidrophobic: hidrofobic + hidrofobic
    # 5 - Repulsive: positive=>positive ou negative=>negative
    # 6 - Atractive: positive=>negative ou negative=>positive
    # 7 - salt_bridge: positive=>negative ou negative=>positive

    # 'RES:ATOM':[Hydrophobic,Aromatic,Positive,Negative,Donor,Acceptor]
    # 'ALA:CA':[0|1,0|1,0|1,0|1,0|1,0|1]
    contatos = { 'ALA:N': [0, 0, 0, 0, 1, 0], 'ALA:CA': [0, 0, 0, 0, 0, 0], 'ALA:C': [0, 0, 0, 0, 0, 0], 'ALA:O': [0, 0, 0, 0, 0, 1], 'ALA:CB': [1, 0, 0, 0, 0, 0], 'ARG:N': [0, 0, 0, 0, 1, 0], 'ARG:CA': [0, 0, 0, 0, 0, 0], 'ARG:C': [0, 0, 0, 0, 0, 0], 'ARG:O': [0, 0, 0, 0, 0, 1], 'ARG:CB': [1, 0, 0, 0, 0, 0], 'ARG:CG': [1, 0, 0, 0, 0, 0], 'ARG:CD': [0, 0, 0, 0, 0, 0], 'ARG:NE': [0, 0, 1, 0, 1, 0], 'ARG:CZ': [0, 0, 1, 0, 0, 0], 'ARG:NH1': [0, 0, 1, 0, 1, 0], 'ARG:NH2': [0, 0, 1, 0, 1, 0], 'ASN:N': [0, 0, 0, 0, 1, 0], 'ASN:CA': [0, 0, 0, 0, 0, 0], 'ASN:C': [0, 0, 0, 0, 0, 0], 'ASN:O': [0, 0, 0, 0, 0, 1], 'ASN:CB': [1, 0, 0, 0, 0, 0], 'ASN:CG': [0, 0, 0, 0, 0, 0], 'ASN:OD1': [0, 0, 0, 0, 0, 1], 'ASN:ND2': [0, 0, 0, 0, 1, 0], 'ASP:N': [0, 0, 0, 0, 1, 0], 'ASP:CA': [0, 0, 0, 0, 0, 0], 'ASP:C': [0, 0, 0, 0, 0, 0], 'ASP:O': [0, 0, 0, 0, 0, 1], 'ASP:CB': [1, 0, 0, 0, 0, 0], 'ASP:CG': [0, 0, 0, 0, 0, 0], 'ASP:OD1': [0, 0, 0, 1, 0, 1], 'ASP:OD2': [0, 0, 0, 1, 0, 1], 'CYS:N': [0, 0, 0, 0, 1, 0], 'CYS:CA': [0, 0, 0, 0, 0, 0], 'CYS:C': [0, 0, 0, 0, 0, 0], 'CYS:O': [0, 0, 0, 0, 0, 1], 'CYS:CB': [1, 0, 0, 0, 0, 0], 'CYS:SG': [0, 0, 0, 0, 1, 1], 'GLN:N': [0, 0, 0, 0, 1, 0], 'GLN:CA': [0, 0, 0, 0, 0, 0], 'GLN:C': [0, 0, 0, 0, 0, 0], 'GLN:O': [0, 0, 0, 0, 0, 1], 'GLN:CB': [1, 0, 0, 0, 0, 0], 'GLN:CG': [1, 0, 0, 0, 0, 0], 'GLN:CD': [0, 0, 0, 0, 0, 0], 'GLN:OE1': [0, 0, 0, 0, 0, 1], 'GLN:NE2': [0, 0, 0, 0, 1, 0], 'GLU:N': [0, 0, 0, 0, 1, 0], 'GLU:CA': [0, 0, 0, 0, 0, 0], 'GLU:C': [0, 0, 0, 0, 0, 0], 'GLU:O': [0, 0, 0, 0, 0, 1], 'GLU:CB': [1, 0, 0, 0, 0, 0], 'GLU:CG': [1, 0, 0, 0, 0, 0], 'GLU:CD': [0, 0, 0, 0, 0, 0], 'GLU:OE1': [0, 0, 0, 1, 0, 1], 'GLU:OE2': [0, 0, 0, 1, 0, 1], 'GLY:N': [0, 0, 0, 0, 1, 0], 'GLY:CA': [0, 0, 0, 0, 0, 0], 'GLY:C': [0, 0, 0, 0, 0, 0], 'GLY:O': [0, 0, 0, 0, 0, 1], 'HIS:N': [0, 0, 0, 0, 1, 0], 'HIS:CA': [0, 0, 0, 0, 0, 0], 'HIS:C': [0, 0, 0, 0, 0, 0], 'HIS:O': [0, 0, 0, 0, 0, 1], 'HIS:CB': [1, 0, 0, 0, 0, 0], 'HIS:CG': [0, 1, 0, 0, 0, 0], 'HIS:ND1': [0, 1, 1, 0, 1, 1], 'HIS:CD2': [0, 1, 0, 0, 0, 0], 'HIS:CE1': [0, 1, 0, 0, 0, 0], 'HIS:NE2': [0, 1, 1, 0, 1, 1], 'ILE:N': [0, 0, 0, 0, 1, 0], 'ILE:CA': [0, 0, 0, 0, 0, 0], 'ILE:C': [0, 0, 0, 0, 0, 0], 'ILE:O': [0, 0, 0, 0, 0, 1], 'ILE:CB': [1, 0, 0, 0, 0, 0], 'ILE:CG1': [1, 0, 0, 0, 0, 0], 'ILE:CG2': [1, 0, 0, 0, 0, 0], 'ILE:CD1': [1, 0, 0, 0, 0, 0], 'LEU:N': [0, 0, 0, 0, 1, 0], 'LEU:CA': [0, 0, 0, 0, 0, 0], 'LEU:C': [0, 0, 0, 0, 0, 0], 'LEU:O': [0, 0, 0, 0, 0, 1], 'LEU:CB': [1, 0, 0, 0, 0, 0], 'LEU:CG': [1, 0, 0, 0, 0, 0], 'LEU:CD1': [1, 0, 0, 0, 0, 0], 'LEU:CD2': [1, 0, 0, 0, 0, 0], 'LYS:N': [0, 0, 0, 0, 1, 0], 'LYS:CA': [0, 0, 0, 0, 0, 0], 'LYS:C': [0, 0, 0, 0, 0, 0], 'LYS:O': [0, 0, 0, 0, 0, 1], 'LYS:CB': [1, 0, 0, 0, 0, 0], 'LYS:CG': [1, 0, 0, 0, 0, 0], 'LYS:CD': [1, 0, 0, 0, 0, 0], 'LYS:CE': [0, 0, 0, 0, 0, 0], 'LYS:NZ': [0, 0, 1, 0, 1, 0], 'MET:N': [0, 0, 0, 0, 1, 0], 'MET:CA': [0, 0, 0, 0, 0, 0], 'MET:C': [0, 0, 0, 0, 0, 0], 'MET:O': [0, 0, 0, 0, 0, 1], 'MET:CB': [1, 0, 0, 0, 0, 0], 'MET:CG': [1, 0, 0, 0, 0, 0], 'MET:SD': [0, 0, 0, 0, 0, 1], 'MET:CE': [1, 0, 0, 0, 0, 0], 'PHE:N': [0, 0, 0, 0, 1, 0], 'PHE:CA': [0, 0, 0, 0, 0, 0], 'PHE:C': [0, 0, 0, 0, 0, 0], 'PHE:O': [0, 0, 0, 0, 0, 1], 'PHE:CB': [1, 0, 0, 0, 0, 0], 'PHE:CG': [1, 1, 0, 0, 0, 0], 'PHE:CD1': [1, 1, 0, 0, 0, 0], 'PHE:CD2': [1, 1, 0, 0, 0, 0], 'PHE:CE1': [1, 1, 0, 0, 0, 0], 'PHE:CE2': [1, 1, 0, 0, 0, 0], 'PHE:CZ': [1, 1, 0, 0, 0, 0], 'PRO:N': [0, 0, 0, 0, 0, 0], 'PRO:CA': [0, 0, 0, 0, 0, 0], 'PRO:C': [0, 0, 0, 0, 0, 0], 'PRO:O': [0, 0, 0, 0, 0, 1], 'PRO:CB': [1, 0, 0, 0, 0, 0], 'PRO:CG': [1, 0, 0, 0, 0, 0], 'PRO:CD': [0, 0, 0, 0, 0, 0], 'SER:N': [0, 0, 0, 0, 1, 0], 'SER:CA': [0, 0, 0, 0, 0, 0], 'SER:C': [0, 0, 0, 0, 0, 0], 'SER:O': [0, 0, 0, 0, 0, 1], 'SER:CB': [0, 0, 0, 0, 0, 0], 'SER:OG': [0, 0, 0, 0, 1, 1], 'THR:N': [0, 0, 0, 0, 1, 0], 'THR:CA': [0, 0, 0, 0, 0, 0], 'THR:C': [0, 0, 0, 0, 0, 0], 'THR:O': [0, 0, 0, 0, 0, 1], 'THR:CB': [0, 0, 0, 0, 0, 0], 'THR:OG1': [0, 0, 0, 0, 1, 1], 'THR:CG2': [1, 0, 0, 0, 0, 0], 'TRP:N': [0, 0, 0, 0, 1, 0], 'TRP:CA': [0, 0, 0, 0, 0, 0], 'TRP:C': [0, 0, 0, 0, 0, 0], 'TRP:O': [0, 0, 0, 0, 0, 1], 'TRP:CB': [1, 0, 0, 0, 0, 0], 'TRP:CG': [1, 1, 0, 0, 0, 0], 'TRP:CD1': [0, 1, 0, 0, 0, 0], 'TRP:CD2': [1, 1, 0, 0, 0, 0], 'TRP:NE1': [0, 1, 0, 0, 1, 0], 'TRP:CE2': [0, 1, 0, 0, 0, 0], 'TRP:CE3': [1, 1, 0, 0, 0, 0], 'TRP:CZ2': [1, 1, 0, 0, 0, 0], 'TRP:CZ3': [1, 1, 0, 0, 0, 0], 'TRP:CH2': [1, 1, 0, 0, 0, 0], 'TYR:N': [0, 0, 0, 0, 1, 0], 'TYR:CA': [0, 0, 0, 0, 0, 0], 'TYR:C': [0, 0, 0, 0, 0, 0], 'TYR:O': [0, 0, 0, 0, 0, 1], 'TYR:CB': [1, 0, 0, 0, 0, 0], 'TYR:CG': [1, 1, 0, 0, 0, 0], 'TYR:CD1': [1, 1, 0, 0, 0, 0], 'TYR:CD2': [1, 1, 0, 0, 0, 0], 'TYR:CE1': [1, 1, 0, 0, 0, 0], 'TYR:CE2': [1, 1, 0, 0, 0, 0], 'TYR:CZ': [0, 1, 0, 0, 0, 0], 'TYR:OH': [0, 0, 0, 0, 1, 1], 'VAL:N': [0, 0, 0, 0, 1, 0], 'VAL:CA': [0, 0, 0, 0, 0, 0], 'VAL:C': [0, 0, 0, 0, 0, 0], 'VAL:O': [0, 0, 0, 0, 0, 1], 'VAL:CB': [1, 0, 0, 0, 0, 0], 'VAL:CG1': [1, 0, 0, 0, 0, 0], 'VAL:CG2': [1, 0, 0, 0, 0, 0]}

    # -----------------------------------------------------------------------------
    # 1. LÊ ARQUIVO PDB E CONVERTE EM DICIONÁRIO
    # -----------------------------------------------------------------------------
    # lê átomos
    with open(entrada) as pdb_file:
        linhas = pdb_file.readlines()
        pdb = {}
        for linha in linhas:
            if linha[0:4] == 'ATOM':
                residuo = linha[17:20].strip()
                atomo = linha[13:16].strip()
                res_num = linha[22:26].strip()
                cadeia = linha[21]

                chave = cadeia+':'+res_num+'-'+residuo+':'+atomo

                x = float(linha[30:38].strip())
                y = float(linha[38:46].strip())
                z = float(linha[46:54].strip())

                coord = (x, y, z)

                pdb[chave] = coord

    # -----------------------------------------------------------------------------
    # 2. CALCULA matriz de distância => TODOS contra TODOS
    # -----------------------------------------------------------------------------

    def distancia(x1,y1,z1,x2,y2,z2):
        ''' Calcula a distância euclidiana usando a fórmula:
            dist² = (x2-x1)² + (y2-y1)² + (z2-z1)²
        '''
        dist = math.sqrt(
            (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2 
        )

        return round(dist, 1)

    matriz_distancia = {}

    # contatores numéricos
    ni = 0
    nj = 0


    # preenche linha
    for i in pdb:

        matriz_distancia[i] = {}

        # preenche coluna
        for j in pdb:

            if ni<nj:  # remove redundância (diagonal inferior)

                matriz_distancia[i][j] = distancia(
                    pdb[i][0], pdb[i][1], pdb[i][2],
                    pdb[j][0], pdb[j][1], pdb[j][2]
                )

            nj+=1
        ni+=1

    # -----------------------------------------------------------------------------
    # 3. CÁLCULO DE CONTATOS 
    # -----------------------------------------------------------------------------
    # contatos = {'RES:ATOM': [Hydrophobic, Aromatic, Positive, Negative, Donor, Acceptor]}

    # Analisa matriz de distância
    for i in matriz_distancia:

        r1 = i.split('-')
        r1_num = r1[0]
        r1_name = r1[1]

        for j in matriz_distancia[i]:

            r2 = j.split('-')
            r2_num = r2[0]
            r2_name = r2[1]

            # evita comparações com o mesmo resíduo
            if r1_num != r2_num:


                # hidrofobico -----------------------------------------------------
                if mostrar_contato['hidrofobico']:
                    if matriz_distancia[i][j] >= hidrophobic[0] and matriz_distancia[i][j] <= hidrophobic[1]:
                        try:
                            if contatos[r1_name][0] == 1 and contatos[r2_name][0] == 1:
                                if saida == 'tela':
                                    print('HIDROFÓBICO', i, j, matriz_distancia[i][j], sep=';')
                                else:
                                    w.write("HY;"+i+";"+j+";"+str(matriz_distancia[i][j])+"\n")
                        except:
                            warning = 'Ignorando chave inexistente: '+i+','+j


                # aromatic --------------------------------------------------------
                if mostrar_contato['aromatico']:
                    if matriz_distancia[i][j] >= aromatic[0] and matriz_distancia[i][j] <= aromatic[1]:
                        try:
                            if contatos[r1_name][1] == 1 and contatos[r2_name][1] == 1:
                                if saida == 'tela':
                                    print('AROMÁTICO', i, j, matriz_distancia[i][j], sep=';')
                                else:
                                    w.write("AR;"+i+";"+j+";"+str(matriz_distancia[i][j])+"\n")
                        except:
                            warning = 'Ignorando chave inexistente: '+i+','+j


                # repulsive -------------------------------------------------------
                if mostrar_contato['repulsivo']:
                    if matriz_distancia[i][j] >= repulsive[0] and matriz_distancia[i][j] <= repulsive[1]:
                        try:
                            if ( 
                                (contatos[r1_name][2] == 1 and contatos[r2_name][2] == 1) or # positivos vs. positivo
                                (contatos[r1_name][3] == 1 and contatos[r2_name][3] == 1)    # negativo vs. negativo
                            ):
                                if saida == 'tela':
                                    print('REPULSIVO', i, j, matriz_distancia[i][j], sep=';')
                                else:
                                    w.write("RE;"+i+";"+j+";"+str(matriz_distancia[i][j])+"\n")
                        except:
                            warning = 'Ignorando chave inexistente: '+i+','+j


                # atrativo -------------------------------------------------------
                if mostrar_contato['atrativo']:
                    if matriz_distancia[i][j] >= repulsive[0] and matriz_distancia[i][j] <= repulsive[1]:
                        try:
                            if ( 
                                (contatos[r1_name][2] == 1 and contatos[r2_name][3] == 1) or # positivos vs. negativo
                                (contatos[r1_name][3] == 1 and contatos[r2_name][2] == 1)    # negativo vs. positivo
                            ):
                                if saida == 'tela':
                                    print('ATRATIVO', i, j, matriz_distancia[i][j], sep=';')
                                else:
                                    w.write("AT;"+i+";"+j+";"+str(matriz_distancia[i][j])+"\n")
                        except:
                            warning = 'Ignorando chave inexistente: '+i+','+j


                # ligação de hidrogênio (hb) ------------------------------------
                if mostrar_contato['ligacao_hidrogenio']:
                    if matriz_distancia[i][j] >= hidrogen_bond[0] and matriz_distancia[i][j] <= hidrogen_bond[1]:
                        try:
                            if ( 
                                (contatos[r1_name][4] == 1 and contatos[r2_name][5] == 1) or # donor vs. aceptor
                                (contatos[r1_name][5] == 1 and contatos[r2_name][4] == 1)    # aceptor vs. donor
                            ):
                                if saida == 'tela':
                                    print('LIGAÇÃO_DE_HIDROGÊNIO', i, j, matriz_distancia[i][j], sep=';')
                                else:
                                    w.write("HB;"+i+";"+j+";"+str(matriz_distancia[i][j])+"\n")
                        except:
                            warning = 'Ignorando chave inexistente: '+i+','+j


                # ponte salina (atrativo + hb) ------------------------------------
                if mostrar_contato['ponte_salina']:
                    if matriz_distancia[i][j] >= salt_bridge[0] and matriz_distancia[i][j] <= salt_bridge[1]:
                        try:
                            # atrativo
                            if ( 
                                (contatos[r1_name][2] == 1 and contatos[r2_name][3] == 1) or # positivos vs. negativo
                                (contatos[r1_name][3] == 1 and contatos[r2_name][2] == 1)    # negativo vs. positivo
                            ):
                                # ligação de hidrogênio
                                if ( 
                                    (contatos[r1_name][4] == 1 and contatos[r2_name][5] == 1) or # donor vs. aceptor
                                    (contatos[r1_name][5] == 1 and contatos[r2_name][4] == 1)    # aceptor vs. donor
                                ):
                                    if saida == 'tela':
                                        print('PONTE_SALINA', i, j, matriz_distancia[i][j], sep=';')
                                    else:
                                        w.write("SB;"+i+";"+j+";"+str(matriz_distancia[i][j])+"\n")
                        except:
                            warning = 'Ignorando chave inexistente: '+i+','+j


    if(saida == 'csv'):
        print('Cálculo de contatos realizado com sucesso. \nResultado gravado em: "contato.csv"')
        print('\nEspecificações do CSV:\nTipo de contato; Resíduo:átomo 1; Resíduo:átomo 2; Distância\n')
        print('Tipos de contatos:\n\tHB: ligação de hidrogênio\n\tHY: hidrofóbio\n\tAR: aromático\n\tAT: atrativo\n\tRE: repulsivo\n\tSB: ponte salida\n')
        

def processa_VTR(gen):
    # É preciso identificar qual peptídeo que originou o contato
    # Criamos dicionários para qualquer as informações que nos importam do VTR
    df = {'Protein':[],
          'Peptide':[],
          'Distance':[],
          'Contact':[],
          'File': []
         }

    modelos = {'Iteration':[],
               'Peptide':[],
               'Hydrophobic':[],
               'Aromatic Stacking':[],
               'Attractive':[],
               'Hydrogen Bond':[],
               'Repulsive':[],
               'Salt Bridge':[],
    }

    # Criamos um dicionário para padronizar os tipos de ligação
    new = {"HB": 'Hydrogen Bond',
           "HY": 'Hydrophobic',
           "AR": 'Aromatic Stacking',
           "AT": 'Attractive',
           "RE": 'Repulsive',
           "SB": 'Salt Bridge',
          }

    # Indicamos o nome da pasta onde os resultados do VTR a serem tratados estão
    pasta = f'Results/VTR/I{gen}/'

    # O Loop/laço for a seguir lê cada linha dos arquivos do VTR e as interpreta
    for file in os.listdir(pasta):
        filename = pasta + file
        dist = False
        pep_name = file.split('.')[0]
        modelos['Peptide'].append(pep_name) #Adicionamos o nome do peptídeo
        modelos['Iteration'].append(gen)
        # Inicializamos as contagens dos tipos de interação que o VTR identificou
        tipos = {'Hydrophobic':0,
                 'Aromatic Stacking':0,
                 'Attractive':0,
                 'Hydrogen Bond':0,
                 'Repulsive':0,
                 'Salt Bridge':0,
                }

        # Neste laço/loop for, nós preenchemos nossa tabela com as informações interpretadas
        for line in read_file(filename):
            l = re.split(';|:|-|\n',line) # Dividimos a linha em palavras, ignorando os espaços
            if len(l) == 11 and l[1] in ['A','B']: # O VTR tem o arquivo padronizado. Linhas com informações que queremos tem 11 campos
                if l[1] != l[5]: # Se a cadeia 1 l[2] é diferente da cadeia 2 l[7], processamos
                    if l[1] == 'A':
                        df['Protein'].append(' '.join(l[2:4]))
                        df['Peptide'].append(' '.join(l[6:8]))
                    else:
                        df['Protein'].append(' '.join(l[6:8]))
                        df['Peptide'].append(' '.join(l[2:4]))
                    df['Distance'].append(l[9])
                    df['Contact'].append(new[l[0]])
                    df['File'].append(file)
                    tipos[new[l[0]]] += 1  # Contabilizamos o tipo de contato encontrado


        # Neste laço/loop for, nós registramos as contagens de tipos de ligação para o modelo
        for k in tipos.keys():
            modelos[k].append(tipos[k])

    # CRIAR A TABELA
    df = pd.DataFrame(df)
    modelos = pd.DataFrame(modelos)
    return df, modelos

# Scripts para testes

def fake_docking(protocol=1, scorefxn=1, sequence=1, gen=0, nsims=1):
    origin = f"Inputs/I{gen}/{pept}.pdb"
    score_dict = {}
    for i in range(nsims):
        shutil.copy(src=origin, dst=f'Results/Docking/I{gen}/{pept}_{i}.pdb')
        fitness = np.random.choice([-1,1])*np.random.rand()*np.random.choice([50,100,150,200,250,300,350,400,450,500])
        score_dict[f'{sequence}_{i}'] = fitness
    return score_dict

def fake_VTR(entrada=0, outfile='fail.csv'):
    df = pd.read_csv('Results/ref.csv')
    df.to_csv(outfile, index=False)
    return None

def fake_countContacts(peptide_file=None, target = None, cutoff = 6.0, count_on = 'target'):
    return np.random.rand()

aminoletters = {'ALA':'A',
                'CYS':'C',
                'ASP':'D', 
                'GLU':'E', 
                'PHE':'F', 
                'GLY':'G',
                'HIS':'H',
                'ILE':'I',
                'LYS':'K',
                'LEU':'L',
                'MET':'M',
                'ASN':'N',
                'PRO':'P',
                'GLN':'Q',
                'ARG':'R',
                'SER':'S',
                'THR':'T',
                'VAL':'V',
                'TRP':'W',
                'TYR':'Y',
                }

conservative_list = {'A':['E', 'S', 'T'],
                     'C':['S', 'T','V'],
                     'D':['E'], 
                     'E':['A', 'D'], 
                     'F':['M', 'W', 'Y'], 
                     'G':['N', 'P'],
                     'H':['K', 'R'],
                     'I':['L', 'V'],
                     'K':['H', 'R'],
                     'L':['I', 'V'],
                     'M':['F', 'W', 'Y'],
                     'N':['G', 'D', 'Q'],
                     'P':['G'],
                     'Q':['N', 'E'],
                     'R':['H', 'K'],
                     'S':['A', 'T'],
                     'T':['A', 'I', 'S'],
                     'V':['A', 'I', 'L'],
                     'W':['F', 'M', 'Y'],
                     'Y':['F', 'M', 'W'],
                    }

# Precisaria completar para ser realmente usável
nonconservative_list = {'A':['K'],
                     'C':[],
                     'D':[], 
                     'E':['K', 'Q'], 
                     'F':[], 
                     'G':[],
                     'H':['A'],
                     'I':[],
                     'K':['A', 'E'],
                     'L':[],
                     'M':[],
                     'N':['Y'],
                     'P':[],
                     'Q':[],
                     'R':[],
                     'S':[],
                     'T':[],
                     'V':[],
                     'W':[],
                     'Y':[],
                    }

RBM = []
for line in read_file('RBM.txt'):
    l = eval(line)
    RBM.append((l[0],aminoletters[l[1]]))
    
# Define the protocol
protocol = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string("""
<ROSETTASCRIPTS>
<SCOREFXNS>
    <ScoreFunction name="fa_standard" weights="ref2015.wts"/>
</SCOREFXNS>
<MOVERS>
    <FlexPepDock name="ppack"  ppk_only="true"/>
    <FlexPepDock name="fpd" lowres_abinitio="true" pep_refine="true"/>
    <FlexPepDock name="minimize"  min_only="true"/>
</MOVERS>
<PROTOCOLS>
    <Add mover="ppack"/>
    <Add mover="fpd"/>
    <Add mover="minimize"/>
</PROTOCOLS>
<OUTPUT/>
</ROSETTASCRIPTS>""").get_mover("ParsedProtocol")

# Define the scorefxn 
scorefxn = pyrosetta.create_score_function("ref2015") # This is the standard full_atom scoring from pyrosetta

#Criar pastas, caso não existam

gen = 0 #Não modificar!
pre_check(nsims)
if not os.path.isdir(f"Results/Docking/I{gen}/"):
    os.mkdir(f"Results/Docking/I{gen}/")
if not os.path.isdir(f"Results/VTR/I{gen}/"):
    os.mkdir(f"Results/VTR/I{gen}/")

testados = []
#Passo 1 - Cortar peptídeo base
#peptide = start('6m0j.pdb')
peptide = 'STIEEQAKTFLDKFNHEAEDLFYQSSLASWN'
testados.append(peptide)

#Passo 2 - Docking
makePeptide(peptide, gen=0)
scores_dict = docking(protocol, scorefxn, peptide, gen=0, nsims=nsims)

#Passo 3 - Contar quantas interações são realizadas
site = '['
for s in open('RBM.txt').readlines():
    site = site + s.replace('\n',',')
site = site + ']'
site = eval(site)
ocup_dict = {}
modelos = pd.DataFrame()
for i in range(nsims):
    peptide_file = f"Results/Docking/I{gen}/{peptide}_{i}.pdb"
    occupancy = countContacts(peptide_file, target=site, cutoff=6.0)
    ocup_dict[f'{peptide}_{i}'] = occupancy
    #Passo 4 - Identificar os tipos de interações com o VTR
    VTR(entrada=peptide_file, outfile=f'Results/VTR/I{gen}/{peptide}_{i}.csv')

df, modelos = processa_VTR(gen)
modelos['Contatos'] = 0
modelos['Ocup'] = 0
modelos['Scores'] = 0
df['iteration'] = gen
for i in range(nsims):
    contatos = modelos.loc[modelos['Peptide']==f'{peptide}_{i}', ['Hydrophobic', 'Aromatic Stacking', 'Attractive', 'Hydrogen Bond', 'Repulsive', 'Salt Bridge']].t
o_numpy().sum() 
    modelos.loc[modelos['Peptide']==f'{peptide}_{i}', 'Contatos'] = contatos
    modelos.loc[modelos['Peptide']==f'{peptide}_{i}', 'Ocup'] = ocup_dict[f'{peptide}_{i}']
    modelos.loc[modelos['Peptide']==f'{peptide}_{i}', 'Scores'] = scores_dict[f'{peptide}_{i}']

df.to_csv(f'Results/contacts_{gen}.csv')
modelos.to_csv('Results/Modelos.csv', index=False)

#Passo 5 - Identifica porção que interage com a RBM
# O critério usado aqui é a máxima ocupação
max_ocup_ite = modelos.loc[modelos['Iteration']==gen, 'Ocup'].max()
pep = modelos.loc[modelos['Ocup']==max_ocup_ite, 'Peptide'].values[0]
pep_file  = f"Results/Docking/I{gen}/{pep}.pdb"
residuos = countContacts(pep_file, target = RBM, cutoff = 6.0, count_on = 'peptide')

#Passo 6 - Identificar região que interage e clivar as partes que não tem interação para construção de novos peptídeos
em_contato = ''.join([aminoletters[y] for x,y in residuos])

population = []
# Precisa definir como será feito no caso de não ser aleatório - A escolha de quantas modificações 
# e de quais resíduos precisarão ser modificados. E como
while len(population) < pop_size:
    num_changes = np.random.randint(low=1, high=len(em_contato))
    new_pep = random_variation(em_contato, num_changes)
    if new_pep not in testados:
        population.append(new_pep)
        testados.append(new_pep)

# Início das iterações:
for gen in range(1, iterations):
    print("\n******************\nGERAÇÃO",gen,"\n********************\n\n")
    if not os.path.isdir(f"Results/Docking/I{gen}/"):
        os.mkdir(f"Results/Docking/I{gen}/")
    if not os.path.isdir(f"Results/VTR/I{gen}/"):
        os.mkdir(f"Results/VTR/I{gen}/")
    
    
    for pept in population:
        makePeptide(pept, gen=gen)
        scores = docking(protocol, scorefxn, pept, gen=gen, nsims=nsims)
        scores_dict |= scores
        for i in range(nsims):
            peptide_file = f"Results/Docking/I{gen}/{pept}_{i}.pdb"
            VTR(entrada=peptide_file, outfile=f'Results/VTR/I{gen}/{pept}_{i}.csv')
    
    
    for peptide_file in os.listdir(f"Results/Docking/I{gen}/"):
        for i in range(nsims):
            peptide_file = peptide_file.split('/')[-1]  # corrige o bug do endereço duplicado
            peptide_file = f"Results/Docking/I{gen}/{peptide_file}"
            #residuos = countContacts(peptide_file, target = RBM, cutoff = 6.0, count_on = 'peptide')
    
            #Passo 4 - Contar quantas interações são realizadas
            occupancy = countContacts(peptide_file, target=site, cutoff=6.0)
            pep = peptide_file.split('/')[-1].split('.')[0]
            ocup_dict[pep] = occupancy
        
    #Passo 5 - Identificar os tipos de interações com o VTR
    old_model = modelos.copy()
    df, modelos = processa_VTR(gen)
    modelos['Contatos'] = 0
    modelos['Ocup'] = 0
    modelos['Scores'] = 0
    for pept in population:
        for i in range(nsims):
            contatos = modelos.loc[modelos['Peptide']==f'{pept}_{i}', ['Hydrophobic', 'Aromatic Stacking', 'Attractive', 'Hydrogen Bond', 'Repulsive', 'Salt Bridge']].to_numpy().sum() 
            modelos.loc[modelos['Peptide']==f'{pept}_{i}', 'Contatos'] = contatos
            modelos.loc[modelos['Peptide']==f'{pept}_{i}', 'Ocup'] = ocup_dict[f'{pept}_{i}']
            modelos.loc[modelos['Peptide']==f'{pept}_{i}', 'Scores'] = scores_dict[f'{pept}_{i}']
        
    df['iteration'] = gen
    modelos = pd.concat([old_model, modelos]).reset_index(drop=True)
    modelos.to_csv('Results/Modelos.csv', index=False)
    df.to_csv(f'Results/contacts_{gen}.csv')
    
    # Gera nova população baseada no peptídeo que teve maior ocupação
    max_ocup_ite = modelos.loc[modelos['Iteration']==gen, 'Ocup'].max()
    pep = modelos.loc[modelos['Ocup']==max_ocup_ite, 'Peptide'].values[0]
    melhor = pep.split('_')[0]

    population = []
    # Precisa definir como será feito no caso de não ser aleatório - A escolha de quantas modificações 
    # e de quais resíduos precisarão ser modificados. E como
    while len(population) < pop_size:
        num_changes = np.random.randint(low=1, high=len(melhor))
        new_pep = random_variation(melhor, num_changes)
        if new_pep not in testados:
            population.append(new_pep)
            testados.append(new_pep)

