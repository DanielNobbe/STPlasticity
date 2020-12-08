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
from Model_sim_exp1 import *

# TODO: Finish refactoring for uncued, and for part 2 of exp1. And for other experiments


def main():     
    tracemalloc.start()

    #set this if you are using nengo OCL
    platform = cl.get_platforms()[0]   #select platform, should be 0
    device=platform.get_devices()[0]   #select GPU, use 0 (Nvidia 1) or 1 (Nvidia 3)
    context=cl.Context([device])

    nengo_gui_on = __name__ == 'builtins' #python3
    sim_to_run = 2
    sim_no = str(sim_to_run)


    if nengo_gui_on:
        generate_gabors() #generate gabors
        create_model(seed=0) #build model

        memory_item_cued = 0 + 90
        probe_cued = 42 + 90 
        memory_item_uncued = 0 + 90
        probe_uncued = 42 + 90


    else: #no gui
        
        #path
        cur_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+'/data/' #store output in data subfolder
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        #simulation 1: recreate fig 3 & 4, 100 trials for both cued and uncued with 0 and 42 degree memory items and probes
        # D: They say 100 trials, is this with a different setting? They're trying to get 'average cosine sim'
        # so that indicates each run is the same?
        if sim_to_run == 1:
        
            print('Running simulation 1')
            print('')
            
            load_gabors_svd = True #no need to randomize this
            
            ntrials = 1 # Used to be 100
            store_representations = True
            store_decisions = False
            uncued = True # Set to True for full experiment, and plotting of figure 3,4


            #store results        
            templates=np.array([90,93,97,102,108,115,123,132])
            mem_cued = np.zeros((3000,len(templates)+1)) #keep cosine sim for 9 items (templates + impulse)
            mem_uncued = np.zeros((3000,len(templates)+1))
            # D: why is this 3000???

            # DANIEL: Added cache
            # cache = Cache()

            #first, run 100 trials to get average cosine sim
            for run in range(ntrials):
                
                print('Run ' + str(run+1))

                #stimuli
                phase = 180*randint(0, 9)
                memory_item_cued = 0 + 90 + phase
                probe_cued = 42 + 90 + phase
                memory_item_uncued = memory_item_cued
                probe_uncued = probe_cued

                #create new gabor filters every 10 trials
                if run % 10 == 0:
                    if run>0:
                        # D: clean up / clear out old gabor filters to make room in memory for new ones
                        del sim
                        del model
                        del e_cued, U_cued, compressed_im_cued
                        if uncued:
                            del e_uncued, U_uncued, compressed_im_uncued
                        gc.collect()
                        load_gabors_svd = False # Re-enable the generation
                    if not uncued:
                        e_cued, U_cued, compressed_im_cued = generate_gabors(
                                            load_gabors_svd=load_gabors_svd, uncued=uncued)
                    else:
                        (e_cued, U_cued, compressed_im_cued, e_uncued, U_uncued, compressed_im_uncued
                            ) = generate_gabors(load_gabors_svd=load_gabors_svd, uncued=uncued)
                    # D: compressed_im_cued is 
            

                # D: Each run has a different seed for the model and simulation
                # This also means we can directly pass the memory_item_cued, making this a lot simpler
                # Not how I would set it up myself, but I guess it works.
                if not uncued:
                    model = create_model(seed=run, nengo_gui_on=False, store_representations=store_representations,
                            store_decisions=store_decisions, uncued=uncued, e_cued=e_cued, U_cued=U_cued, compressed_im_cued=compressed_im_cued, 
                            memory_item_cued=memory_item_cued, probe_cued=probe_cued)
                else:
                    model = create_model(seed=run, nengo_gui_on=False, store_representations=store_representations,
                            store_decisions=store_decisions, uncued=uncued, e_cued=e_cued, U_cued=U_cued, 
                            compressed_im_cued=compressed_im_cued, e_uncued=e_uncued, U_uncued=U_uncued, 
                            compressed_im_uncued=compressed_im_cued, memory_item_cued=memory_item_cued, 
                            memory_item_uncued=memory_item_uncued, probe_cued=probe_cued, probe_uncued=probe_uncued)
                sim = StpOCLsimulator(network=model, seed=run, context=context,progress_bar=False)

                #run simulation
                sim.run(3)
                # Each run only has a single input angle and probe angle. These always have the same relative angle, but the absolute angles vary by 180 deg.
                # this means that the real angle probably does not differ at all. 

                #reset simulator, clean probes thoroughly
                #print(sim.data[model.p_mem_cued].shape)
                #calc cosine sim with templates
                temp_phase = list(templates + phase) + [1800]
                # set_trace()
                for cnt, templ in enumerate(temp_phase):
                    mem_cued[:,cnt] += cosine_sim(sim.data[model.p_mem_cued][:,:,],compressed_im_cued[templ,:])
                    # In the sim.data[model.p_mem_cued], a vector of length 24 for the representation in the memory ensemble
                    # is stored for every timestep (by default total of 3000 = 3 seconds x 1 ms timesteps)

                    # In the compressed_im_cued, the optimal compression of the image in 24 dimensions is stored.
                    # They select a subset of images at specific angles (templ).

                    # Still unclear what is exactly extracted by this probe

                    mem_cued = deepcopy(mem_cued)
                    if uncued:
                        mem_uncued[:,cnt] += cosine_sim(sim.data[model.p_mem_uncued][:,:,],compressed_im_uncued[templ,:])
                # if run == 9:
                #     set_trace()
                # # D: Here, they delete some stuff, but not everything. I should take a look what exactly stays in memory
                # D: what they delete here might be the results of the simulation
                # D: which is weird, since they re-initialize the simulator after this
                sim.reset()
                for probe2 in sim.model.probes:
                    del sim._probe_outputs[probe2][:]
                del sim.data
                sim.data = nengo.simulator.ProbeDict(sim._probe_outputs) 
                
            
            #average
            mem_cued /= ntrials
            mem_uncued /= ntrials

            #second, run 1 trial to get calcium and spikes
            store_spikes_and_resources = True
            store_representations = False
            if not uncued:
                model = create_model(seed=0, nengo_gui_on=False, store_representations=store_representations, store_spikes_and_resources=store_spikes_and_resources,
                            store_decisions=store_decisions, uncued=uncued, e_cued=e_cued, U_cued=U_cued, compressed_im_cued=compressed_im_cued, 
                            memory_item_cued=memory_item_cued, probe_cued=probe_cued)
            else:
                model = create_model(seed=0, nengo_gui_on=False, store_representations=store_representations, store_spikes_and_resources=store_spikes_and_resources,
                            store_decisions=store_decisions, uncued=uncued, e_cued=e_cued, U_cued=U_cued, 
                            compressed_im_cued=compressed_im_cued, e_uncued=e_uncued, U_uncued=U_uncued, 
                            compressed_im_uncued=compressed_im_cued, memory_item_cued=memory_item_cued, 
                            memory_item_uncued=memory_item_uncued, probe_cued=probe_cued, probe_uncued=probe_uncued)
            # create_model(seed=0, nengo_gui_on=nengo_gui_on, store_representations=store_representations,
            #     store_decisions=store_decisions, store_spikes_and_resources=store_spikes_and_resources) #recreate model to change probes
            sim = StpOCLsimulator(network=model, seed=0, context=context,progress_bar=False)

            print('Run ' + str(ntrials+1))
            sim.run(3)

            #store spikes and calcium
            sp_c = sim.data[model.p_spikes_mem_cued]
            res_c=np.mean(sim.data[model.p_res_cued][:,:,],1) #take mean over neurons
            cal_c=np.mean(sim.data[model.p_cal_cued][:,:,],1) #take mean over neurons

            if uncued:
                sp_u=sim.data[model.p_spikes_mem_uncued]
                res_u=np.mean(sim.data[model.p_res_uncued][:,:,],1)
                cal_u=np.mean(sim.data[model.p_cal_uncued][:,:,],1)

            #plot
            if uncued:
                plot_sim_1(sp_c,sp_u,res_c,res_u,cal_c,cal_u, mem_cued, mem_uncued, sim=sim)
            

        #simulation 2: collect data for fig 5 & 6. 1344 trials for 30 subjects
        if sim_to_run == 2:
        
            load_gabors_svd = True #set to false for real simulation D: should be true

            n_subj = 3 # Was 30
            trials_per_subj = 14 # was 1344
            store_representations = False 
            store_decisions = True # Should be true for this exp 
            uncued = False # Only do this for cued module. TODO: Do we want to change this?


            # D: stimuli for init/test
            phase = 180*randint(0, 9)
            memory_item_cued = 0 + 90 + phase
            probe_cued = 42 + 90 + phase
            memory_item_uncued = memory_item_cued
            probe_uncued = probe_cued

            #np array to keep track of the input during the simulation runs
            initialangle_c = np.zeros(n_subj*trials_per_subj) #cued
            angle_index=0
            
            #orientation differences between probe and memory item for each run
            probelist=[-42, -33, -25, -18, -12, -7, -3, 3, 7, 12, 18, 25, 33, 42]

            for subj in range(n_subj):

                #create new gabor filters and model for each new participant
                if subj>0:
                    del sim
                    del model
                    del e_cued, U_cued, compressed_im_cued
                    if uncued:
                        del e_uncued, U_uncued, compressed_im_uncued
                    gc.collect()
                if not uncued:
                    e_cued, U_cued, compressed_im_cued = generate_gabors(
                                        load_gabors_svd=load_gabors_svd, uncued=uncued)
                else:
                    (e_cued, U_cued, compressed_im_cued, e_uncued, U_uncued, compressed_im_uncued
                        ) = generate_gabors(load_gabors_svd=load_gabors_svd, uncued=uncued)

                if not uncued:
                    model = create_model(seed=subj, nengo_gui_on=False, store_representations=store_representations,
                            store_decisions=store_decisions, uncued=uncued, e_cued=e_cued, U_cued=U_cued, compressed_im_cued=compressed_im_cued, 
                            memory_item_cued=memory_item_cued, probe_cued=probe_cued)
                    # set_trace()
                      
                else:
                    model = create_model(seed=subj, nengo_gui_on=False, store_representations=store_representations,
                            store_decisions=store_decisions, uncued=uncued, e_cued=e_cued, U_cued=U_cued, 
                            compressed_im_cued=compressed_im_cued, e_uncued=e_uncued, U_uncued=U_uncued, 
                            compressed_im_uncued=compressed_im_cued, memory_item_cued=memory_item_cued, 
                            memory_item_uncued=memory_item_uncued, probe_cued=probe_cued, probe_uncued=probe_uncued)

                #use StpOCLsimulator to make use of the Nengo OCL implementation of STSP
                sim = StpOCLsimulator(network=model, seed=subj, context=context,progress_bar=False)
                sim.run(3)
                # set_trace()
                #trials come in sets of 14, which we call a run (all possible orientation differences between memory and probe),
                runs = int(trials_per_subj / 14)   

                for run in range(runs):
        
                    #run a trial with each possible orientation difference
                    for cnt_in_run, anglediff in enumerate(probelist):
    
                        print('Subject ' + str(subj+1) + '/' + str(n_subj) + '; Trial ' + str(run*14 + cnt_in_run + 1) + '/' + str(trials_per_subj))

                        #set probe and stim
                        memory_item_cued=randint(0, 179) #random memory
                        probe_cued=memory_item_cued+anglediff #probe based on that
                        probe_cued=norm_p(probe_cued) #normalise probe

                        #random phase
                        or_memory_item_cued=memory_item_cued #original
                        memory_item_cued=memory_item_cued+(180*randint(0, 9))
                        probe_cued=probe_cued+(180*randint(0, 9))
                
                        #store orientation
                        initialangle_c[angle_index]=or_memory_item_cued
                
                        # D: Insert new probe and memory item into input_cued node:
                        cued_input_partial = partial(input_func_cued, memory_item_cued=memory_item_cued, probe_cued=probe_cued)
                        assert model.nodes[0].label == 'input_cued', "First node not input_cued, please fix"
                        model.nodes[0] = nengo.Node(cued_input_partial,label='input_cued', add_to_container=False) 
                        # set_trace()

                        #run simulation
                        sim.run(3)
                    
                        #store output
                        np.savetxt(cur_path+sim_no+"_Diff_Theta_%i_subj_%i_trial_%i.csv" % (anglediff, subj+1, run*14+cnt_in_run+1), sim.data[model.p_dec_cued][2500:2999,:], delimiter=",")
                
                        #reset simulator, clean probes thoroughly
                        sim.reset()
                        for probe2 in sim.model.probes:
                            del sim._probe_outputs[probe2][:]
                        del sim.data
                        sim.data = nengo.simulator.ProbeDict(sim._probe_outputs) 
                
                        angle_index=angle_index+1
                
            np.savetxt(cur_path+sim_no+"_initial_angles_cued.csv", initialangle_c,delimiter=",")

if __name__ == '__main__':
    main()
