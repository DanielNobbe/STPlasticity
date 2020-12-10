import numpy as np
from PIL import Image
import matplotlib.pyplot as plt   
from nengo.dists import Uniform
import nengo
import math
from stp_ocl_implementation import *
import os, inspect
from nengo_extras.vision import Gabor, Mask
from random import randint
import nengo.spa as spa
import os.path
from diskcache import Cache
from pdb import set_trace
from copy import deepcopy
# For tracemalloc:
import linecache
import os
import tracemalloc

import gc

# For plotting:
from nengo.utils.matplotlib import rasterplot
from matplotlib import style
from plotnine import *

# Import functions from exp1 file:
from Model_sim_exp2 import *


def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))



def main():
    tracemalloc.start()
    #SIMULATION CONTROL for GUI
    load_gabors_svd=True #set to false if you want to generate new ones
    store_representations = False #store representations of model runs (for Fig 3 & 4)
    store_decisions = False #store decision ensemble (for Fig 5 & 6)
    store_spikes_and_resources = False #store spikes, calcium etc. (Fig 3 & 4)

    #specify here which sim you want to run if you do not use the nengo GUI
    #1 = representations & spikes
    #2 = performance, decision signal
    sim_to_run = 2
    sim_no = str(sim_to_run)      #simulation number (used in the names of the outputfiles)

    #set this if you are using nengo OCL
    platform = cl.get_platforms()[0]   #select platform, should be 0
    device=platform.get_devices()[0]   #select GPU, use 0 (Nvidia 1) or 1 (Nvidia 3)
    context=cl.Context([device])


    #MODEL PARAMETERS
    D = 24  #dimensions of representations
    Ns = 1000 #number of neurons in sensory layer
    Nm = 1500 #number of neurons in memory layer
    Nc = 1500 #number of neurons in comparison
    Nd = 1000 #number of neurons in decision


    #LOAD INPUT STIMULI (images created using the psychopy package)
    #(Stimuli should be in a subfolder named 'Stimuli') 

    
    

    
    # if nengo_gui_on:
    #     generate_gabors() #generate gabors
    #     create_model(seed=0) #build model
            
    #     memory_item_first = 0 + 90
    #     probe_first = 40 + 90 
    #     memory_item_second = 0 + 90
    #     probe_second = 40 + 90

    # else: #no gui
    
    #path
    cur_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+'/data_exp2/' #store output in data subfolder
    
    #simulation 1
    if sim_to_run == 1:
    
        print('Running simulation 1')
        print('')
        
        load_gabors_svd = False #no need to randomize this D: originally False
        
        ntrials = 3
        store_representations = True
        store_decisions = False

        #store results        
        templates = np.array([0, 5, 10, 16, 24, 32, 40]) + 90
        mem_1 = np.zeros((4600,len(templates)+1)) #keep cosine sim for 9 items
        mem_2 = np.zeros((4600,len(templates)+1))
        
        #first, run 100 trials to get average cosine sim
        for run in range(ntrials):
        
            print('Run ' + str(run+1))

                    #stimuli

            phase = 180*(run % 10)
            memory_item_first = 0 + 90 + phase
            probe_first = 40 + 90 + phase
            memory_item_second = 0 + 90 + phase
            probe_second = 40 + 90 + phase

            #create new gabor filters every 10 trials
            if run % 10 == 0:
                if run>0:
                    # D: clean up / clear out old gabor filters to make room in memory for new ones
                    del sim
                    del model
                    del e_first, U_first, compressed_im_first
                    del e_second, U_second, compressed_im_second
                    gc.collect()
                    load_gabors_svd = False # Re-enable the generation
                (e_first, U_first, compressed_im_first, e_second, U_second, compressed_im_second
                                            ) = generate_gabors(load_gabors_svd=load_gabors_svd, Ns=Ns, D=D)
                
            model = create_model(seed=run, memory_item_first=memory_item_first, probe_first=probe_first, memory_item_second=memory_item_second,
                        probe_second=probe_second, Ns=Ns, D=D, Nm=Nm, Nc=Nc, Nd=Nd, e_first=e_first, U_first=U_first, compressed_im_first=compressed_im_first,
                        e_second=e_second, U_second=U_second, compressed_im_second=compressed_im_second,
                        store_representations=store_representations, store_decisions=store_decisions, store_spikes_and_resources=store_spikes_and_resources)
            sim = StpOCLsimulator(network=model, seed=run, context=context,progress_bar=False)

            #run simulation
            sim.run(4.6)

            #reset simulator, clean probes thoroughly
            #print(sim.data[model.p_mem_cued].shape)
            #calc cosine sim with templates
            temp_phase = list(templates + phase) + [1800]
            for cnt, templ in enumerate(temp_phase):
                mem_1[:,cnt] += cosine_sim(sim.data[model.p_mem_first][:,:,],compressed_im_first[templ,:])
                mem_2[:,cnt] += cosine_sim(sim.data[model.p_mem_second][:,:,],compressed_im_second[templ,:])
            # set_trace()
            sim.reset()
            for probe2 in sim.model.probes:
                del sim._probe_outputs[probe2][:]
            del sim.data
            sim.data = nengo.simulator.ProbeDict(sim._probe_outputs) 
            
        
        #average
        mem_1 /= ntrials
        mem_2 /= ntrials

        #second, run 1 trial to get calcium and spikes
        store_spikes_and_resources = True
        store_representations = False
        model = create_model(seed=0, memory_item_first=memory_item_first, probe_first=probe_first, memory_item_second=memory_item_second,
                        probe_second=probe_second, Ns=Ns, D=D, Nm=Nm, Nc=Nc, Nd=Nd, e_first=e_first, U_first=U_first, compressed_im_first=compressed_im_first,
                        e_second=e_second, U_second=U_second, compressed_im_second=compressed_im_second,
                        store_representations=store_representations, store_decisions=store_decisions, store_spikes_and_resources=store_spikes_and_resources)
        #recreate model to change probes
        sim = StpOCLsimulator(network=model, seed=0, context=context,progress_bar=False)

        print('Run ' + str(ntrials+1))
        sim.run(4.6)

        #store spikes and calcium
        sp_1 = sim.data[model.p_spikes_mem_first]
        res_1=np.mean(sim.data[model.p_res_first][:,:,],1) #take mean over neurons
        cal_1=np.mean(sim.data[model.p_cal_first][:,:,],1) #take mean over neurons

        sp_2=sim.data[model.p_spikes_mem_second]
        res_2=np.mean(sim.data[model.p_res_second][:,:,],1)
        cal_2=np.mean(sim.data[model.p_cal_second][:,:,],1)

        #plot
        # set_trace()
        plot_sim_1(sp_1,sp_2,res_1,res_2,cal_1,cal_2, mem_1, mem_2, sim=sim, Nm=Nm)
        

    #simulation 2
    if sim_to_run == 2:
    
        load_gabors_svd = False #set to false for real simulation

        n_subj = 1
        trials_per_subj = 12
        # trials_per_subj = 2*864
        store_representations = False 
        store_decisions = True 

        #np array to keep track of the input during the simulation runs
        initialangle_c = np.zeros(n_subj*trials_per_subj) #cued
        angle_index=0
        
        #set default input
        memory_item_first = 0
        probe_first = 0 
        memory_item_second = 0
        probe_second = 0 

        #orientation differences between probe and memory item for each run
        probelist=[-40, -32, -24, -16, -10, -5, 5, 10, 16, 24, 32, 40]

        for subj in range(n_subj):

            #create new gabor filters and model for each new participant
            if subj>0:
                # D: clean up / clear out old gabor filters to make room in memory for new ones
                del sim
                del model
                del e_first, U_first, compressed_im_first
                del e_second, U_second, compressed_im_second
                gc.collect()
                load_gabors_svd = False # Re-enable the generation
            (e_first, U_first, compressed_im_first, e_second, U_second, compressed_im_second
                                            ) = generate_gabors(load_gabors_svd=load_gabors_svd, Ns=Ns, D=D)
            model = create_model(seed=subj, memory_item_first=memory_item_first, probe_first=probe_first, memory_item_second=memory_item_second,
                        probe_second=probe_second, Ns=Ns, D=D, Nm=Nm, Nc=Nc, Nd=Nd, e_first=e_first, U_first=U_first, compressed_im_first=compressed_im_first,
                        e_second=e_second, U_second=U_second, compressed_im_second=compressed_im_second,
                        store_representations=store_representations, store_decisions=store_decisions, store_spikes_and_resources=store_spikes_and_resources)
            

            #use StpOCLsimulator to make use of the Nengo OCL implementation of STSP
            sim = StpOCLsimulator(network=model, seed=subj, context=context,progress_bar=False)

            #trials come in sets of 12, which we call a run (all possible orientation differences between memory and probe),
            runs = int(trials_per_subj / 12)   

            for run in range(runs):
    
                #run a trial with each possible orientation difference
                for cnt_in_run, anglediff in enumerate(probelist):

                    print('Subject ' + str(subj+1) + '/' + str(n_subj) + '; Trial ' + str(run*12 + cnt_in_run + 1) + '/' + str(trials_per_subj))

                    #set probe and stim
                    memory_item_first=randint(0, 179) #random memory
                    probe_first=memory_item_first+anglediff #probe based on that
                    probe_first=norm_p(probe_first) #normalise probe

                    #random phase
                    or_memory_item_first=memory_item_first #original
                    memory_item_first=memory_item_first+(180*randint(0, 9))
                    probe_first=probe_first+(180*randint(0, 9))
            
                    #same for secondary item
                    memory_item_second = memory_item_first
                    probe_second = probe_first
                    
                    # D: Insert new probe and memory item into input_cued node:
                    input_partial_first = partial(input_func_first, memory_item_first=memory_item_first, probe_first=probe_first)
                    input_partial_second = partial(input_func_second, memory_item_second=memory_item_second, probe_second=probe_second)

                    # cued_input_partial = partial(input_func_cued, memory_item_cued=memory_item_cued, probe_cued=probe_cued)
                    assert model.nodes[0].label == 'input_first', "First node not input_first, please fix"
                    model.nodes[0] = nengo.Node(input_partial_first,label='input_first', add_to_container=False) 
                    assert model.nodes[1].label == 'input_second', "Second node not input_second, please fix"
                    model.nodes[0] = nengo.Node(input_partial_second,label='input_second', add_to_container=False) 
                    #run simulation
                    sim.run(4.6)
                
                    #store output
                    np.savetxt(cur_path+sim_no+"_Diff_Theta_%i_subj_%i_trial_%i_probe1.csv" % (anglediff, subj+1, run*12+cnt_in_run+1), sim.data[model.p_dec_first][1800:1900,:], delimiter=",")
                    np.savetxt(cur_path+sim_no+"_Diff_Theta_%i_subj_%i_trial_%i_probe2.csv" % (anglediff, subj+1, run*12+cnt_in_run+1), sim.data[model.p_dec_second][4300:4400,:], delimiter=",")
                    
                    #reset simulator, clean probes thoroughly
                    sim.reset()
                    for probe2 in sim.model.probes:
                        del sim._probe_outputs[probe2][:]
                    del sim.data
                    sim.data = nengo.simulator.ProbeDict(sim._probe_outputs) 
            
                    angle_index=angle_index+1





if __name__=='__main__':
    main()