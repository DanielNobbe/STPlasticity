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
# For tracemalloc:
import linecache
import os
import tracemalloc

# For plotting:
from nengo.utils.matplotlib import rasterplot
from matplotlib import style
from plotnine import *


from pdb import set_trace
from functools import partial

#SIMULATION
#note that this is split for running a single trial in the nengo gui, and a full simulation

        
#normalise stimuli to be between 0 and 180 degrees orientation
def norm_p(p):
    if p<0:
        return 180+p
    if p>180:
        return p-180
    else:
        return p
        
#Calculate normalised cosine similarity and avoid divide by 0 errors
def cosine_sim(a,b):
    out=np.zeros(a.shape[0])
    for i in range(0,  a.shape[0]):
        if abs(np.linalg.norm(a[i])) > 0.05:
            out[i]=np.dot(a[i], b)/(np.linalg.norm(a[i])*np.linalg.norm(b))
    return out
         

# #SIMULATION CONTROL for GUI
# uncued = False #set if you want to run both the cued and uncued model
# load_gabors_svd=True #set to false if you want to generate new ones
# store_representations = False #store representations of model runs (for Fig 3 & 4)
# store_decisions = False #store decision ensemble (for Fig 5 & 6)
# store_spikes_and_resources = False #store spikes, calcium etc. (Fig 3 & 4)

#specify here which sim you want to run if you do not use the nengo GUI
#1 = simulation to generate Fig 3 & 4
#2 = simulation to generate Fig 5 & 6
# sim_to_run = 1
# sim_no="1"      #simulation number (used in the names of the outputfiles)


#set this if you are using nengo OCL
platform = cl.get_platforms()[0]   #select platform, should be 0
device=platform.get_devices()[0]   #select GPU, use 0 (Nvidia 1) or 1 (Nvidia 3)
context=cl.Context([device])

#LOAD INPUT STIMULI (images created using the psychopy package)
#(Stimuli should be in a subfolder named 'Stimuli') 

#width and height of images
diameter=col=row=128 

#load grating stimuli
angles=np.arange(-90,90,1)  #rotation
phases=np.arange(0,1,0.1)   #phase

try:
    imagearr = np.load('Stimuli/all_stims.npy') #load stims if previously generated
except FileNotFoundError: #or generate
    imagearr=np.zeros((0,diameter**2))
    for phase in phases:
        for angle in angles:
            name="Stimuli/stim"+str(angle)+"_"+str(round(phase,1))+".png"
            img=Image.open(name)
            img=np.array(img.convert('L'))
            imagearr=np.vstack((imagearr,img.ravel())) 
    
    #also load the  bull's eye 'impulse stimulus'  
    name="Stimuli/stim999.png"
    img=Image.open(name)
    img=np.array(img.convert('L'))
    imagearr=np.vstack((imagearr,img.ravel())) 
    
    #normalize to be between -1 and 1
    imagearr=imagearr/255
    imagearr=2 * imagearr - 1
    
    #imagearr is a (1801, 16384) np array containing all stimuli + the impulse
    np.save('Stimuli/all_stims.npy',imagearr)



#INPUT FUNCTIONS

#set default input
# memory_item_cued = 0
# probe_cued = 0 
# memory_item_uncued = 0
# probe_uncued = 0 

# D: Add this tracemalloc function, https://stackoverflow.com/questions/552744/how-do-i-profile-memory-usage-in-python
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




#input stimuli
#250 ms memory items | 0-250
#800 ms fixation | 250-1050 
#20 ms reactivation | 1050-1070
#1080 ms fixation | 1070-2150
#100 ms impulse | 2150-2250
#400 ms fixation | 2250-2650
#250 ms probe | 2650-2900
def input_func_cued(t, memory_item_cued=None, probe_cued=None):
    if t > 0 and t < 0.25:
        return imagearr[memory_item_cued,:]/100
    elif t > 2.15 and t < 2.25:
        return imagearr[-1,:]/50 #impulse, twice the contrast of other items
    elif t > 2.65 and t < 2.90:
        return imagearr[probe_cued,:]/100
    else:
        return np.zeros(128*128) #blank screen

def input_func_uncued(t, memory_item_uncued=None, probe_uncued=None):
    if t > 0 and t < 0.25:
        return imagearr[memory_item_uncued,:]/100
    elif t > 2.15 and t < 2.25:
        return imagearr[-1,:]/50 #impulse, twice the contrast of other items
    elif t > 2.65 and t < 2.90:
        return imagearr[probe_uncued,:]/100
    else:
        return np.zeros(128*128) #blank screen

#reactivate memory cued ensemble with nonspecific signal        
def reactivate_func(t, Nm=None):
    if t>1.050 and t<1.070:
        return np.ones(Nm)*0.0200
    else:
        return np.zeros(Nm)

#Create matrix of sine and cosine values associated with the stimuli
#so that we can later specify a transform from stimuli to rotation        
Fa = np.tile(angles,phases.size) #want to do this for each phase
Frad = (Fa/90) * math.pi #make radians
Sin = np.sin(Frad)
Cos = np.cos(Frad)
sincos = np.vstack((Sin,Cos)) #sincos

#Create eval points so that we can go from sine and cosine of theta in sensory and memory layer
#to the difference in theta between the two
samples = 10000
sinAcosA = nengo.dists.UniformHypersphere(surface=True).sample(samples,2)
thetaA = np.arctan2(sinAcosA[:,0],sinAcosA[:,1])
thetaDiff = (90*np.random.random(samples)-45)/180*np.pi
thetaB = thetaA + thetaDiff

sinBcosB = np.vstack((np.sin(thetaB),np.cos(thetaB)))
scale = np.random.random(samples)*0.9+0.1
sinBcosB = sinBcosB * scale
ep = np.hstack((sinAcosA,sinBcosB.T))


#continuous variant of arctan(a,b)-arctan(c,d)
def arctan_func(v):
    yA, xA, yB, xB = v
    z = np.arctan2(yA, xA) - np.arctan2(yB, xB)
    pos_ans = [z, z+2*np.pi, z-2*np.pi]
    i = np.argmin(np.abs(pos_ans))
    return pos_ans[i]*90/math.pi



#MODEL

#gabor generation for a particular model-participant
def generate_gabors(load_gabors_svd=False, uncued=False, Ns=None, D=None):

    # global e_cued
    # global U_cued
    # global compressed_im_cued

    # global e_uncued
    # global U_uncued
    # global compressed_im_uncued

    #to speed things up, load previously generated ones
    if load_gabors_svd & os.path.isfile('Stimuli/gabors_svd_cued.npz'):
        gabors_svd_cued = np.load('Stimuli/gabors_svd_cued.npz') #load stims if previously generated
        e_cued = gabors_svd_cued['e_cued']
        U_cued = gabors_svd_cued['U_cued']
        compressed_im_cued = gabors_svd_cued['compressed_im_cued']
        if not uncued:
            return e_cued, U_cued, compressed_im_cued
        print("SVD cued loaded")

    else: #or generate and save

        #cued module
        #for each neuron in the sensory layer, generate a Gabor of 1/3 of the image size
        # D: Each time the gabors are generated some of their properties are randomly sampled
        gabors_cued = Gabor().generate(Ns, (int(col/3), int(row/3))) # DANIEL: Added casting to int
        #put gabors on image and make them the same shape as the stimuli
        gabors_cued = Mask((col, row)).populate(gabors_cued, flatten=True).reshape(Ns, -1)
        #normalize
        gabors_cued=gabors_cued/abs(max(np.amax(gabors_cued),abs(np.amin(gabors_cued))))
        #gabors are added to imagearr for SVD
        x_cued=np.vstack((imagearr,gabors_cued))    

        #SVD  
        print("SVD cued started...")
        U_cued, S_cued, V_cued = np.linalg.svd(x_cued.T)
        print("SVD cued done")

        #Use result of SVD to create encoders
        e_cued = np.dot(gabors_cued, U_cued[:,:D]) #encoders
        compressed_im_cued = np.dot(imagearr[:1800,:]/100, U_cued[:,:D]) #D-dimensional vector reps of the images
        compressed_im_cued = np.vstack((compressed_im_cued, np.dot(imagearr[-1,:]/50, U_cued[:,:D])))

        np.savez('Stimuli/gabors_svd_cued.npz', e_cued=e_cued, U_cued=U_cued, compressed_im_cued=compressed_im_cued)
        if not uncued:
            return e_cued, U_cued, compressed_im_cued

    #same for uncued module
    if uncued:

        if load_gabors_svd & os.path.isfile('Stimuli/gabors_svd_uncued.npz'):
            gabors_svd_uncued = np.load('Stimuli/gabors_svd_uncued.npz') #load stims if previously generated
            e_uncued = gabors_svd_uncued['e_uncued']
            U_uncued = gabors_svd_uncued['U_uncued']
            compressed_im_uncued = gabors_svd_uncued['compressed_im_uncued']
            print("SVD uncued loaded")
            return (e_cued, U_cued, compressed_im_cued,
                 e_uncued, U_uncued, compressed_im_uncued)
        else:
            gabors_uncued = Gabor().generate(Ns, (int(col/3), int(row/3)))#.reshape(N, -1) # DANIEL: Added casting to ints
            gabors_uncued = Mask((col, row)).populate(gabors_uncued, flatten=True).reshape(Ns, -1)
            gabors_uncued=gabors_uncued/abs(max(np.amax(gabors_uncued),abs(np.amin(gabors_uncued))))
            x_uncued=np.vstack((imagearr,gabors_uncued))    

            print("SVD uncued started...")
            U_uncued, S_uncued, V_uncued = np.linalg.svd(x_uncued.T)
            print("SVD uncued done")
            e_uncued = np.dot(gabors_uncued, U_uncued[:,:D]) # Due to the indexing until D, the images are limited to D dimension. This is later also used like this in the model
            compressed_im_uncued=np.dot(imagearr[:1800,:]/100, U_uncued[:,:D])
            compressed_im_uncued = np.vstack((compressed_im_uncued, np.dot(imagearr[-1,:]/50, U_uncued[:,:D])))
            
            np.savez('Stimuli/gabors_svd_uncued.npz', e_uncued=e_uncued, U_uncued=U_uncued, compressed_im_uncued=compressed_im_uncued)
            return (e_cued, U_cued, compressed_im_cued,
                 e_uncued, U_uncued, compressed_im_uncued)



def create_model(seed=None, nengo_gui_on=False, store_representations=False, store_spikes_and_resources=False, 
store_decisions=False, uncued=False, e_cued=None, U_cued=None, compressed_im_cued=None, e_uncued=None, U_uncued=None, 
    compressed_im_uncued=None, memory_item_cued=None, memory_item_uncued=None, probe_cued=None,
    probe_uncued=None, Ns=None, D=None, Nm=None, Nc=None, Nd=None, attention=False):

    # global model
    
    #create vocabulary to show representations in gui
    # if nengo_gui_on:
    #     vocab_angles = spa.Vocabulary(D)
    #     for name in [0, 3, 7, 12, 18, 25, 33, 42]:
    #         #vocab_angles.add('D' + str(name), np.linalg.norm(compressed_im_cued[name+90])) #take mean across phases
    #         v = compressed_im_cued[name+90]
    #         nrm = np.linalg.norm(v)
    #         if nrm > 0:
    #             v /= nrm
    #         vocab_angles.add('D' + str(name), v) #take mean across phases

    #     v = np.dot(imagearr[-1,:]/50, U_cued[:,:D])
    #     nrm = np.linalg.norm(v)
    #     if nrm > 0:
    #         v /= nrm
    #     vocab_angles.add('Impulse', v)
    
    #model = nengo.Network(seed=seed)
    model = spa.SPA(seed=seed)

    cued_input_partial = partial(input_func_cued, memory_item_cued=memory_item_cued, probe_cued=probe_cued)
    uncued_input_partial = partial(input_func_uncued, memory_item_uncued=memory_item_uncued, probe_uncued=probe_uncued)
    react_partial = partial(reactivate_func, Nm=Nm)

    with model: # Again some weird global stuff happening..
        # So this defines the architecture of the model. We don't necessarily need to change this
        # This is with only a single memory cell, the uncued version has two memory ensembles??
        # TODO: Implement uncued
        #input nodes
        inputNode_cued=nengo.Node(cued_input_partial,label='input_cued')     
        reactivate=nengo.Node(react_partial,label='reactivate')  # Input here at CUE (CUE is not actually shown)
        # The two ensembles above have their functions directly specified
        # the reactivate ensemble gives a pulse to any ensembles it's connected to, and the input node receives images at specific times

        #sensory ensemble
        # Nengo ensemble creates a seemingly interconnected bunch ('ensemble') of Ns neurons
        # Each neuron is a single vector (inner) product, similar to ANNs. 
        # e_cued here is the 'encoder' for the neurons (if seen as matrix, otherwise it's the collection of encoding vectors)
        # which follows from the trained? gabor filters.
        att_fact = 1.25 # the amount with which the max firing rates are increased due to attentional gain
        min_max_rate = int(att_fact*200)
        max_max_rate = int(att_fact*400)
        
        sensory_cued = nengo.Ensemble(Ns, D, encoders=e_cued, intercepts=Uniform(0.01, .1),radius=1,label='sensory_cued')
        memory_cued = nengo.Ensemble(Nm, D,neuron_type=stpLIF(), intercepts=Uniform(0.01, .1),radius=1,label='memory_cued')
        comparison_cued = nengo.Ensemble(Nc, dimensions=4,radius=math.sqrt(2),intercepts=Uniform(.01, 1),label='comparison_cued') 
        decision_cued = nengo.Ensemble(n_neurons=Nd,  dimensions=1,radius=45,label='decision_cued') 
        if attention==1:
            # apply attentional gain to the sensory ensemble in the cued module
            sensory_cued = nengo.Ensemble(Ns, D, encoders=e_cued, intercepts=Uniform(0.01, .1), max_rates=Uniform(min_max_rate,max_max_rate),radius=1,label='sensory_cued')
        elif attention==2:
            memory_cued = nengo.Ensemble(Nm, D,neuron_type=stpLIF(), intercepts=Uniform(0.01, .1), max_rates=Uniform(min_max_rate,max_max_rate),radius=1,label='memory_cued')
        elif attention==3:
            comparison_cued = nengo.Ensemble(Nc, dimensions=4,radius=math.sqrt(2),intercepts=Uniform(.01, 1),max_rates=Uniform(min_max_rate,max_max_rate),label='comparison_cued')
        elif attention==4:
            decision_cued = nengo.Ensemble(n_neurons=Nd,  dimensions=1,radius=45,max_rates=Uniform(min_max_rate,max_max_rate),label='decision_cued') 
        elif attention==5:
            # apply attentional gain to all ensembles in the cued module
            sensory_cued = nengo.Ensemble(Ns, D, encoders=e_cued, intercepts=Uniform(0.01, .1), max_rates=Uniform(min_max_rate,max_max_rate),radius=1,label='sensory_cued')
            memory_cued = nengo.Ensemble(Nm, D,neuron_type=stpLIF(), intercepts=Uniform(0.01, .1), max_rates=Uniform(min_max_rate,max_max_rate),radius=1,label='memory_cued')
            comparison_cued = nengo.Ensemble(Nc, dimensions=4,radius=math.sqrt(2),intercepts=Uniform(.01, 1),max_rates=Uniform(min_max_rate,max_max_rate),label='comparison_cued')
            decision_cued = nengo.Ensemble(n_neurons=Nd,  dimensions=1,radius=45,max_rates=Uniform(min_max_rate,max_max_rate),label='decision_cued')
        elif attention==6:
            # apply attentional gain to all ensembles in the cued module
            sensory_cued = nengo.Ensemble(Ns, D, encoders=e_cued, intercepts=Uniform(0.01, .1), max_rates=Uniform(min_max_rate,max_max_rate),radius=1,label='sensory_cued')
            memory_cued = nengo.Ensemble(Nm, D,neuron_type=stpLIF(), intercepts=Uniform(0.01, .1), max_rates=Uniform(min_max_rate,max_max_rate),radius=1,label='memory_cued')
            comparison_cued = nengo.Ensemble(Nc, dimensions=4,radius=math.sqrt(2),intercepts=Uniform(.01, 1),max_rates=Uniform(min_max_rate,max_max_rate),label='comparison_cued')
            decision_cued = nengo.Ensemble(n_neurons=Nd,  dimensions=1,radius=45,max_rates=Uniform(min_max_rate,max_max_rate),label='decision_cued')
            
        # How does the encoder function work here? Does it use the e_cued matrix to convert the images into 
        # SVD reduced versions? But how do the gabor filters work


        nengo.Connection(inputNode_cued,sensory_cued,transform=U_cued[:,:D].T)
        # Connection means a connection between two nodes/ensembles. Can include a transformation
        # Not sure what the SVD outcomes actually mean TODO

        #memory ensemble
        nengo.Connection(reactivate,memory_cued.neurons) #potential reactivation --> This is the CUE
        nengo.Connection(sensory_cued, memory_cued, transform=.1) #.1)
        
        #recurrent STSP connection
        nengo.Connection(memory_cued, memory_cued,transform=1, learning_rule_type=STP(), solver=nengo.solvers.LstsqL2(weights=True))

        #comparison represents sin, cosine of theta of both sensory and memory ensemble
        nengo.Connection(sensory_cued, comparison_cued[:2],eval_points=compressed_im_cued[0:-1],function=sincos.T)
        nengo.Connection(memory_cued, comparison_cued[2:],eval_points=compressed_im_cued[0:-1],function=sincos.T)
       
        #decision represents the difference in theta decoded from the sensory and memory ensembles
        nengo.Connection(comparison_cued, decision_cued, eval_points=ep, scale_eval_points=False, function=arctan_func)

        #same for uncued
        if uncued:
            inputNode_uncued=nengo.Node(uncued_input_partial,label='input_uncued')

            sensory_uncued = nengo.Ensemble(Ns, D, encoders=e_uncued, intercepts=Uniform(0.01, .1),radius=1,label='sensory_uncued')
            memory_uncued = nengo.Ensemble(Nm, D,neuron_type=stpLIF(), intercepts=Uniform(0.01, .1),radius=1,label='memory_uncued')
            comparison_uncued = nengo.Ensemble(Nd, dimensions=4,radius=math.sqrt(2),intercepts=Uniform(.01, 1),label='comparison_uncued')
            decision_uncued = nengo.Ensemble(n_neurons=Nd,  dimensions=1,radius=45,label='decision_uncued') 
            if attention==5:
                sensory_uncued = nengo.Ensemble(Ns, D, encoders=e_uncued, intercepts=Uniform(0.01, .1),radius=1,max_rates=Uniform(min_max_rate,max_max_rate),label='sensory_uncued')
                memory_uncued = nengo.Ensemble(Nm, D,neuron_type=stpLIF(), intercepts=Uniform(0.01, .1),radius=1,max_rates=Uniform(min_max_rate,max_max_rate),label='memory_uncued')
                comparison_uncued = nengo.Ensemble(Nd, dimensions=4,radius=math.sqrt(2),intercepts=Uniform(.01, 1),max_rates=Uniform(min_max_rate,max_max_rate),label='comparison_uncued')
                decision_uncued = nengo.Ensemble(n_neurons=Nd,  dimensions=1,radius=45,max_rates=Uniform(min_max_rate,max_max_rate),label='decision_uncued')
                
            nengo.Connection(inputNode_uncued,sensory_uncued,transform=U_uncued[:,:D].T)
            
            nengo.Connection(sensory_uncued, memory_uncued, transform=.1)
   
            nengo.Connection(memory_uncued, memory_uncued,transform=1,learning_rule_type=STP(),solver=nengo.solvers.LstsqL2(weights=True))
    
    
            nengo.Connection(memory_uncued, comparison_uncued[2:],eval_points=compressed_im_uncued[0:-1],function=sincos.T)
            nengo.Connection(sensory_uncued, comparison_uncued[:2],eval_points=compressed_im_uncued[0:-1],function=sincos.T)
            
            nengo.Connection(comparison_uncued, decision_uncued, eval_points=ep, scale_eval_points=False, function=arctan_func)
        
        #decode for gui
        if nengo_gui_on:
            model.sensory_decode = spa.State(D, vocab=vocab_angles, subdimensions=12, label='sensory_decode')
            for ens in model.sensory_decode.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(sensory_cued, model.sensory_decode.input,synapse=None)
     
            model.memory_decode = spa.State(D, vocab=vocab_angles, subdimensions=12, label='memory_decode')
            for ens in model.memory_decode.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(memory_cued, model.memory_decode.input,synapse=None)
            
        #probes
        if not(nengo_gui_on):
            if store_representations: #sim 1 trials 1-100
                #p_dtheta_cued=nengo.Probe(decision_cued, synapse=0.01)
                model.p_mem_cued=nengo.Probe(memory_cued, synapse=0.01)
                #p_sen_cued=nengo.Probe(sensory_cued, synapse=0.01)
           
                if uncued:
                    model.p_mem_uncued=nengo.Probe(memory_uncued, synapse=0.01)
                
            if store_spikes_and_resources: #sim 1 trial 1
                model.p_spikes_mem_cued=nengo.Probe(memory_cued.neurons, 'spikes')
                model.p_res_cued=nengo.Probe(memory_cued.neurons, 'resources')
                model.p_cal_cued=nengo.Probe(memory_cued.neurons, 'calcium')
    
                if uncued:
                    model.p_spikes_mem_uncued=nengo.Probe(memory_uncued.neurons, 'spikes')
                    model.p_res_uncued=nengo.Probe(memory_uncued.neurons, 'resources')
                    model.p_cal_uncued=nengo.Probe(memory_uncued.neurons, 'calcium')
            
            if store_decisions: #sim 2
                model.p_dec_cued=nengo.Probe(decision_cued, synapse=0.01)
    return model
#PLOTTING CODE



def plot_sim_1(sp_c,sp_u,res_c,res_u,cal_c,cal_u=None, mem_cued=None, mem_uncued=None, sim=None, Nm=None, fig3_name='Figure_3.eps', fig4_name='Figure_4.eps'):
    theme = theme_classic()
    plt.style.use('default')
    #FIGURE 31
    with plt.rc_context():
        plt.rcParams.update(theme.rcParams)
    
        fig, axes, = plt.subplots(2,2,squeeze=True)
        theme.setup_figure(fig)
        t = sim.trange()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        #spikes, calcium, resources Cued
        ax1=axes[0,0]
        ax1.set_title("Cued Module")
        ax1.set_ylabel('# cell', color='black')
        ax1.set_yticks(np.arange(0,Nm,500))
        ax1.tick_params('y')#, colors='black')
        rasterplot(sim.trange(), sp_c,ax1,colors=['black']*sp_c.shape[0])
        ax1.set_xticklabels([])
        ax1.set_xticks([])
        ax1.set_xlim(0,3)
        ax2 = ax1.twinx()
        ax2.plot(t, res_c, "#00bfc4",linewidth=2)
        ax2.plot(t, cal_c, "#e38900",linewidth=2)
        ax2.set_yticklabels([])
        ax2.set_yticks([])
        ax2.set_ylim(0,1.1)

        if cal_u is not None:
            #spikes, calcium, resources Uncued
            ax3=axes[0,1]
            ax3.set_title("Uncued Module")
            rasterplot(sim.trange(), sp_u,ax3,colors=['black']*sp_u.shape[0])
            ax3.set_xticklabels([])
            ax3.set_xticks([])
            ax3.set_yticklabels([])
            ax3.set_yticks([])
            ax3.set_xlim(0,3)
            ax4 = ax3.twinx()
            ax4.plot(t, res_u, "#00bfc4",linewidth=2)
            ax4.plot(t, cal_u, "#e38900",linewidth=2)
            ax4.set_ylabel('synaptic variables', color="black",size=11)
            ax4.tick_params('y', labelcolor='#333333',labelsize=9,color='#333333')
            ax4.set_ylim(0,1.1)

        #representations cued
        plot_mc=axes[1,0]
        plot_mc.plot(sim.trange(),(mem_cued));

        plot_mc.set_ylabel("Cosine similarity")
        plot_mc.set_ylim(-0.2,1)
        plot_mc.set_xticks(np.arange(0.0,3.45,0.5))
        plot_mc.set_xticklabels(np.arange(0,3500,500).tolist())
        plot_mc.set_xlabel('time (ms)')
        plot_mc.set_xlim(0,3)
        colors=["#00c094","#00bfc4","#00b6eb","#06a4ff","#a58aff","#df70f8","#fb61d7","#ff66a8", "#c49a00"]
        for i,j in enumerate(plot_mc.lines):
            j.set_color(colors[i])

        if cal_u is not None:
            #representations uncued
            plot_mu=axes[1,1]

            plot_mu.plot(sim.trange(),(mem_uncued));
            plot_mu.set_xticks(np.arange(0.0,3.45,0.5))
            plot_mu.set_xticklabels(np.arange(0,3500,500).tolist())
            plot_mu.set_xlabel('time (ms)')
            plot_mu.set_yticks([])
            plot_mu.set_yticklabels([])
            plot_mu.set_xlim(0,3)
            for i,j in enumerate(plot_mu.lines):
                j.set_color(colors[i])
            plot_mu.legend(["0°","3°","7°","12°","18°","25°","33°","42°", "Impulse"], title="Stimulus", bbox_to_anchor=(1.02, -0.25, .30, 0.8), loc=3,
                ncol=1, mode="expand", borderaxespad=0.)


        fig.set_size_inches(11, 5)
        theme.apply(fig.axes[0])
        theme.apply(fig.axes[1])
        # theme.apply(fig.axes[2]) # Gives error
        theme.apply(fig.axes[3])

        plt.savefig(fig3_name, format='eps', dpi=1000)
        plt.show()
    
    
    #FIGURE 32
    with plt.rc_context():
        plt.rcParams.update(theme.rcParams)
    
        fig, axes, = plt.subplots(1,2,squeeze=True)
        theme.setup_figure(fig)
        t = sim.trange()
        plt.subplots_adjust(wspace=0.1, hspace=0.05)
   
        plot_mc=axes[0]
        plot_mc.set_title("Cued Module")
        plot_mc.plot(sim.trange(),(mem_cued));
        plot_mc.set_ylabel("Cosine similarity")
        plot_mc.set_ylim(0,1)
        plot_mc.set_xticks(np.arange(2.15,2.35,0.05))
        plot_mc.set_xticklabels(np.arange(0,250,50).tolist())
        plot_mc.set_xlabel('time after onset impulse (ms)')
        plot_mc.set_xlim(2.15,2.3)
        plot_mc.set_ylim(0,0.9)
        colors=["#00c094","#00bfc4","#00b6eb","#06a4ff","#a58aff","#df70f8","#fb61d7","#ff66a8", "#c49a00"]
        for i,j in enumerate(plot_mc.lines):
            j.set_color(colors[i])

        plot_mu=axes[1]
        plot_mu.set_title("Uncued Module")
        plot_mu.plot(sim.trange(),(mem_uncued));
        plot_mu.set_xticks(np.arange(2.15,2.35,0.05))
        plot_mu.set_xticklabels(np.arange(0,250,50).tolist())
        plot_mu.set_xlabel('time after onset impulse (ms)')
        plot_mu.set_yticks([])
        plot_mu.set_yticklabels([])
        plot_mu.set_xlim(2.15,2.30)
        plot_mu.set_ylim(0,0.9)
        for i,j in enumerate(plot_mu.lines):
            j.set_color(colors[i])
        plot_mu.legend(["0°","3°","7°","12°","18°","25°","33°","42°", "Impulse"], title="Stimulus", bbox_to_anchor=(0.85, 0.25, .55, 0.8), loc=3,
               ncol=1, mode="expand", borderaxespad=0.)

        fig.set_size_inches(6, 4)

        # theme.apply(fig.axes[0]) # Gives error
        theme.apply(fig.axes[1])
        # theme.apply(plt.gcf().axes[0])
        # theme.apply(plt.gcf().axes[1])
        plt.savefig(fig4_name, format='eps', dpi=1000)
        plt.show()    
    
    
    

if __name__ == '__main__':
    pass