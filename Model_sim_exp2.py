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


#width and height of images
diameter=col=row=128 

#load grating stimuli
angles=np.arange(-90,90,1)  #rotation
phases=np.arange(0,1,0.1)   #phase

#stim2003: 60% grey
try:
    imagearr = np.load('Stimuli/all_stims_exp2_or.npy') #load stims if previously generated
except FileNotFoundError: #or generate
    imagearr=np.zeros((0,diameter**2))
    for phase in phases:
        for angle in angles:
            name="Stimuli/stim"+str(angle)+"_"+str(round(phase,1))+".png"
            img=Image.open(name)
            img=np.array(img.convert('L'))
            imagearr=np.vstack((imagearr,img.ravel())) 
    
    # name="Stimuli/stim2003.png" # D: Manually made this one at 60% grey. Can also use original one using stim999.png
    name="Stimuli/stim999.png"
    img=Image.open(name)
    img=np.array(img.convert('L'))
    imagearr=np.vstack((imagearr,img.ravel())) 
    
    #normalize to be between -1 and 1
    imagearr=imagearr/255
    imagearr=2 * imagearr - 1
    
    #imagearr is a (1801, 16384) np array containing all stimuli + the impulse
    np.save('Stimuli/all_stims_exp2.npy',imagearr)

    


#INPUT FUNCTIONS

#set default input
memory_item_first = 0
probe_first = 0 
memory_item_second = 0
probe_second = 0 

#input stimuli 1
#250 ms memory items | 0-250
#950 ms fixation | 250-1200 
#100 ms impulse | 1200-1300
#500 ms fixation | 1300-1800
#250 ms probe | 1800-2050
#1750 fixation | 2050-3800
#100 ms impulse | 3800-3900
#400 ms fixation | 3900-4300
#250 ms probe | 4300-4550
def input_func_first(t, memory_item_first=None, probe_first=None):
    if t > 0 and t < 0.25:
        return (imagearr[memory_item_first,:]/100) * 1.0
    elif t > 1.2 and t < 1.3:
        return imagearr[-1,:]/50 #impulse, twice the contrast of other items
        # return imagearr[-1,:]/100 #impulse, without increasing contrast
    elif t > 1.8 and t < 2.05:
        return imagearr[probe_first,:]/100
    elif t > 3.8 and t < 3.9:
        return imagearr[-1,:]/50 #impulse, twice the contrast of other items
    #elif t > 4.3 and t < 4.55:
    #    return imagearr[probe_second,:]/100
    else:
        return np.zeros(128*128) #blank screen

def input_func_second(t, memory_item_second=None, probe_second=None):
    if t > 0 and t < 0.25:
        return (imagearr[memory_item_second,:]/100) * .9 #slightly lower input for secondary item
    elif t > 1.2 and t < 1.3:
        return imagearr[-1,:]/50 #impulse, twice the contrast of other items
        # return imagearr[-1,:]/100 #impulse, without increasing contrast
    #elif t > 1.8 and t < 2.05:
    #    return imagearr[probe_first,:]/100
    elif t > 3.8 and t < 3.9:
        return imagearr[-1,:]/50 #impulse, twice the contrast of other items
    elif t > 4.3 and t < 4.55:
        return imagearr[probe_second,:]/100
    else:
        return np.zeros(128*128) #blank screen
        
#reactivate second ensemble based on lateralization     
def reactivate_func(t, Nm=None):
    if t>2.250 and t<2.270:
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
def generate_gabors(load_gabors_svd=False, Ns=None, D=None):

    # global e_first
    # global U_first
    # global compressed_im_first

    # global e_second
    # global U_second
    # global compressed_im_second

    #to speed things up, load previously generated ones
    if load_gabors_svd & os.path.isfile('Stimuli/gabors_svd_first_exp2.npz'):
        gabors_svd_first = np.load('Stimuli/gabors_svd_first_exp2.npz') #load stims if previously generated
        e_first = gabors_svd_first['e_first']
        U_first = gabors_svd_first['U_first']
        compressed_im_first = gabors_svd_first['compressed_im_first']
        print("SVD first loaded")

    else: #or generate and save

        #cued module
        #for each neuron in the sensory layer, generate a Gabor of 1/3 of the image size
        gabors_first = Gabor().generate(Ns, (int(col/3), int(row/3))) 
        #put gabors on image and make them the same shape as the stimuli
        gabors_first = Mask((col, row)).populate(gabors_first, flatten=True).reshape(Ns, -1)
        #normalize
        gabors_first=gabors_first/abs(max(np.amax(gabors_first),abs(np.amin(gabors_first))))
        #gabors are added to imagearr for SVD
        x_first=np.vstack((imagearr,gabors_first))    

        #SVD  
        print("SVD first started...")
        U_first, S_first, V_first = np.linalg.svd(x_first.T)
        print("SVD first done")

        #Use result of SVD to create encoders
        e_first = np.dot(gabors_first, U_first[:,:D]) #encoders
        compressed_im_first = np.dot(imagearr[:1800,:]/100, U_first[:,:D]) #D-dimensional vector reps of the images
        compressed_im_first = np.vstack((compressed_im_first, np.dot(imagearr[-1,:]/50, U_first[:,:D])))

        np.savez('Stimuli/gabors_svd_first_exp2.npz', e_first=e_first, U_first=U_first, compressed_im_first=compressed_im_first)
    

    #same for secondary module

    if load_gabors_svd & os.path.isfile('Stimuli/gabors_svd_second_exp2.npz'):
        gabors_svd_second = np.load('Stimuli/gabors_svd_second_exp2.npz') #load stims if previously generated
        e_second = gabors_svd_second['e_second']
        U_second = gabors_svd_second['U_second']
        compressed_im_second = gabors_svd_second['compressed_im_second']
        print("SVD second loaded")
    else:
        gabors_second = Gabor().generate(Ns, (int(col/3), int(row/3)))#.reshape(N, -1)
        gabors_second = Mask((col, row)).populate(gabors_second, flatten=True).reshape(Ns, -1)
        gabors_second=gabors_second/abs(max(np.amax(gabors_second),abs(np.amin(gabors_second))))
        x_second=np.vstack((imagearr,gabors_second))    

        print("SVD second started...")
        U_second, S_second, V_second = np.linalg.svd(x_second.T)
        print("SVD second done")
        e_second = np.dot(gabors_second, U_second[:,:D]) 
        compressed_im_second=np.dot(imagearr[:1800,:]/100, U_second[:,:D])
        compressed_im_second = np.vstack((compressed_im_second, np.dot(imagearr[-1,:]/50, U_second[:,:D])))
        
        np.savez('Stimuli/gabors_svd_second_exp2.npz', e_second=e_second, U_second=U_second, compressed_im_second=compressed_im_second)
    
    return e_first, U_first, compressed_im_first, e_second, U_second, compressed_im_second

nengo_gui_on = __name__ == 'builtins' #python3

def create_model(seed=None, memory_item_first=None, probe_first=None, memory_item_second=None, probe_second=None,
                Ns=None, D=None, Nm=None, Nc=None, Nd=None, e_first=None, U_first=None, compressed_im_first=None,
                 e_second=None, U_second=None, compressed_im_second=None,
                store_representations=None, store_decisions=None, store_spikes_and_resources=None, args=None):

    # global model
    
    #create vocabulary to show representations in gui
    # if nengo_gui_on:
        # vocab_angles = spa.Vocabulary(D)
        # for name in [0, 5, 10, 16, 24, 32, 40]:
        #     #vocab_angles.add('D' + str(name), np.linalg.norm(compressed_im_first[name+90])) #take mean across phases
        #     v = compressed_im_first[name+90]
        #     nrm = np.linalg.norm(v)
        #     if nrm > 0:
        #         v /= nrm
        #     vocab_angles.add('D' + str(name), v) #take mean across phases

        # v = np.dot(imagearr[-1,:]/50, U_first[:,:D])
        # nrm = np.linalg.norm(v)
        # if nrm > 0:
        #     v /= nrm
        # vocab_angles.add('Impulse', v)

    input_partial_first = partial(input_func_first, memory_item_first=memory_item_first, probe_first=probe_first)
    input_partial_second = partial(input_func_second, memory_item_second=memory_item_second, probe_second=probe_second)
    react_partial = partial(reactivate_func, Nm=Nm)
    #model = nengo.Network(seed=seed)
    model = spa.SPA(seed=seed)
    with model:

        #input nodes
        inputNode_first=nengo.Node(input_partial_first,label='input_first')     
        
        #sensory ensemble
        if '1' in args.gain_module and 'sens' in args.gain_ensemble: 
            sensory_first = nengo.Ensemble(Ns, D, neuron_type=gainLIF(on_time=args.on_time1, off_time=args.off_time1, gain=args.gain_level), 
                encoders=e_first, intercepts=Uniform(0.01, .1),radius=1,label='sensory_first')
        else:
            sensory_first = nengo.Ensemble(Ns, D,
                encoders=e_first, intercepts=Uniform(0.01, .1),radius=1,label='sensory_first')
        nengo.Connection(inputNode_first,sensory_first,transform=U_first[:,:D].T)
        
        #memory ensemble
        if '1' in args.gain_module and 'mem' in args.gain_ensemble: 
            memory_first = nengo.Ensemble(Nm, D,neuron_type=stpLIF(on_time=args.on_time1, off_time=args.off_time1, gain=args.gain_level), 
                intercepts=Uniform(0.01, .1),radius=1,label='memory_first') 
        else:
            memory_first = nengo.Ensemble(Nm, D,neuron_type=stpLIF(), 
                intercepts=Uniform(0.01, .1),radius=1,label='memory_first') 
        nengo.Connection(sensory_first, memory_first, transform=.1) #.1)
        
        #recurrent STSP connection
        nengo.Connection(memory_first, memory_first,transform=1, learning_rule_type=STP(), solver=nengo.solvers.LstsqL2(weights=True))

        #comparison represents sin, cosine of theta of both sensory and memory ensemble
        if '1' in args.gain_module and 'comp' in args.gain_ensemble: 
            comparison_first = nengo.Ensemble(Nc,neuron_type=gainLIF(on_time=args.on_time1, off_time=args.off_time1, gain=args.gain_level), 
                dimensions=4,radius=math.sqrt(2),intercepts=Uniform(.01, 1),label='comparison_first') 
        else:
            comparison_first = nengo.Ensemble(Nc,
                dimensions=4,radius=math.sqrt(2),intercepts=Uniform(.01, 1),label='comparison_first') 
        nengo.Connection(sensory_first, comparison_first[:2],eval_points=compressed_im_first[0:-1],function=sincos.T)
        nengo.Connection(memory_first, comparison_first[2:],eval_points=compressed_im_first[0:-1],function=sincos.T)
       
        #decision represents the difference in theta decoded from the sensory and memory ensembles
        if '1' in args.gain_module and 'dec' in args.gain_ensemble: 
            decision_first = nengo.Ensemble(neuron_type=gainLIF(on_time=args.on_time1, off_time=args.off_time1, gain=args.gain_level), 
                n_neurons=Nd,  dimensions=1, radius=45,label='decision_first') 
        else:
            decision_first = nengo.Ensemble(n_neurons=Nd,  dimensions=1, radius=45,label='decision_first') 
        nengo.Connection(comparison_first, decision_first, eval_points=ep, scale_eval_points=False, function=arctan_func)

        #same for secondary module
        inputNode_second=nengo.Node(input_partial_second,label='input_second')
        reactivate=nengo.Node(react_partial,label='reactivate') 
    
        if '2' in args.gain_module and 'sens' in args.gain_ensemble: 
            sensory_second = nengo.Ensemble(Ns, D, neuron_type=gainLIF(on_time=args.on_time2, off_time=args.off_time2, gain=args.gain_level), 
                encoders=e_second, intercepts=Uniform(0.01, .1),radius=1,label='sensory_second')
        else:
            sensory_second = nengo.Ensemble(Ns, D, encoders=e_second, intercepts=Uniform(0.01, .1),radius=1,label='sensory_second')
        nengo.Connection(inputNode_second,sensory_second,transform=U_second[:,:D].T)
        
        if '2' in args.gain_module and 'mem' in args.gain_ensemble: 
            memory_second = nengo.Ensemble(Nm, D,neuron_type=stpLIF(on_time=args.on_time2, off_time=args.off_time2, gain=args.gain_level), 
                intercepts=Uniform(0.01, .1),radius=1,label='memory_second') 
        else:
            memory_second = nengo.Ensemble(Nm, D,neuron_type=stpLIF(), intercepts=Uniform(0.01, .1),radius=1,label='memory_second')
        nengo.Connection(sensory_second, memory_second, transform=.1)
        nengo.Connection(reactivate,memory_second.neurons) #potential reactivation

        nengo.Connection(memory_second, memory_second,transform=1,learning_rule_type=STP(),solver=nengo.solvers.LstsqL2(weights=True))

        if '2' in args.gain_module and 'comp' in args.gain_ensemble: 
            comparison_second = nengo.Ensemble(Nc,neuron_type=gainLIF(on_time=args.on_time2, off_time=args.off_time2, gain=args.gain_level), 
                dimensions=4,radius=math.sqrt(2),intercepts=Uniform(.01, 1),label='comparison_second') 
        else:
            comparison_second = nengo.Ensemble(Nc, dimensions=4,radius=math.sqrt(2),intercepts=Uniform(.01, 1),label='comparison_second')

        nengo.Connection(memory_second, comparison_second[2:],eval_points=compressed_im_second[0:-1],function=sincos.T)
        nengo.Connection(sensory_second, comparison_second[:2],eval_points=compressed_im_second[0:-1],function=sincos.T)
        
        if '2' in args.gain_module and 'dec' in args.gain_ensemble: 
            decision_second = nengo.Ensemble(neuron_type=gainLIF(on_time=args.on_time2, off_time=args.off_time2, gain=args.gain_level), 
                n_neurons=Nd,  dimensions=1, radius=45,label='decision_second') 
        else:
            decision_second = nengo.Ensemble(n_neurons=Nd,  dimensions=1,radius=45,label='decision_second') 
        nengo.Connection(comparison_second, decision_second, eval_points=ep, scale_eval_points=False, function=arctan_func)
     
        # #decode for gui
        # if nengo_gui_on:
        #     model.sensory_decode = spa.State(D, vocab=vocab_angles, subdimensions=12, label='sensory_decode')
        #     for ens in model.sensory_decode.all_ensembles:
        #         ens.neuron_type = nengo.Direct()
        #     nengo.Connection(sensory_first, model.sensory_decode.input,synapse=None)
     
        #     model.memory_decode = spa.State(D, vocab=vocab_angles, subdimensions=12, label='memory_decode')
        #     for ens in model.memory_decode.all_ensembles:
        #         ens.neuron_type = nengo.Direct()
        #     nengo.Connection(memory_first, model.memory_decode.input,synapse=None)
            
        #probes
        if not(nengo_gui_on):
            if store_representations: #sim 1 trials 1-100
 
                model.p_mem_first=nengo.Probe(memory_first, synapse=0.01)
                model.p_mem_second=nengo.Probe(memory_second, synapse=0.01)
                model.p_sens_first = nengo.Probe(sensory_first, synapse=0.01)
                model.p_sens_second = nengo.Probe(sensory_second, synapse=0.01)
            if store_spikes_and_resources: #sim 1 trial 1
                model.p_spikes_mem_first=nengo.Probe(memory_first.neurons, 'spikes')
                model.p_res_first=nengo.Probe(memory_first.neurons, 'resources')
                model.p_cal_first=nengo.Probe(memory_first.neurons, 'calcium')
    
                model.p_spikes_mem_second=nengo.Probe(memory_second.neurons, 'spikes')
                model.p_res_second=nengo.Probe(memory_second.neurons, 'resources')
                model.p_cal_second=nengo.Probe(memory_second.neurons, 'calcium')
            
            if store_decisions: #sim 2
                model.p_dec_first=nengo.Probe(decision_first, synapse=0.01)
                model.p_dec_second=nengo.Probe(decision_second, synapse=0.01)
        return model


#PLOTTING CODE
from nengo.utils.matplotlib import rasterplot
from matplotlib import style
from plotnine import *
theme = theme_classic()
plt.style.use('default')

def plot_sim_1(sp_1,sp_2,res_1,res_2,cal_1,cal_2, mem_1, mem_2, sim=None, Nm=None):

    #representations  & spikes
    with plt.rc_context():
        plt.rcParams.update(theme.rcParams)
    
        fig, axes, = plt.subplots(2,2,squeeze=True)
        theme.setup_figure(fig)
        t = sim.trange()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        #spikes, calcium, resources first
        ax1=axes[0,0]
        ax1.set_title("First Module")
        ax1.set_ylabel('# cell', color='black')
        ax1.set_yticks(np.arange(0,Nm,500))
        ax1.tick_params('y')#, colors='black')
        rasterplot(sim.trange(), sp_1,ax1,colors=['black']*sp_1.shape[0])
        ax1.set_xticklabels([])
        ax1.set_xticks([])
        ax1.set_xlim(0,4.6)
        ax2 = ax1.twinx()
        ax2.plot(t, res_1, "#00bfc4",linewidth=2)
        ax2.plot(t, cal_1, "#e38900",linewidth=2)
        ax2.set_yticklabels([])
        ax2.set_yticks([])
        ax2.set_ylim(0,1.1)

        #spikes, calcium, resources second
        ax3=axes[0,1]
        ax3.set_title("Second Module")
        rasterplot(sim.trange(), sp_2,ax3,colors=['black']*sp_2.shape[0])
        ax3.set_xticklabels([])
        ax3.set_xticks([])
        ax3.set_yticklabels([])
        ax3.set_yticks([])
        ax3.set_xlim(0,4.6)
        ax4 = ax3.twinx()
        ax4.plot(t, res_2, "#00bfc4",linewidth=2)
        ax4.plot(t, cal_2, "#e38900",linewidth=2)
        ax4.set_ylabel('synaptic variables', color="black",size=11)
        ax4.tick_params('y', labelcolor='#333333',labelsize=9,color='#333333')
        ax4.set_ylim(0,1.1)

        #representations first
        plot_mc=axes[1,0]
        plot_mc.plot(sim.trange(),(mem_1));

        plot_mc.set_ylabel("Cosine similarity")
        plot_mc.set_xticks(np.arange(0.0,4.6,0.5))
        plot_mc.set_xticklabels(np.arange(0,4600,500).tolist())
        plot_mc.set_xlabel('time (ms)')
        plot_mc.set_xlim(0,4.6)
        colors=["#00c094","#00bfc4","#00b6eb","#06a4ff","#a58aff","#df70f8","#fb61d7", "#c49a00"]
        for i,j in enumerate(plot_mc.lines):
            j.set_color(colors[i])

        #representations uncued
        plot_mu=axes[1,1]

        plot_mu.plot(sim.trange(),(mem_2))
        plot_mu.set_xticks(np.arange(0.0,4.6,0.5))
        plot_mu.set_xticklabels(np.arange(0,4600,500).tolist())
        plot_mu.set_xlabel('time (ms)')
        plot_mu.set_yticks([])
        plot_mu.set_yticklabels([])
        plot_mu.set_xlim(0,4.6)
        for i,j in enumerate(plot_mu.lines):
            j.set_color(colors[i])
        plot_mu.legend(["0°","5°","10°","16°","24°","32°","40°", "Impulse"], title="Stimulus", bbox_to_anchor=(1.02, -0.25, .30, 0.8), loc=3,
               ncol=1, mode="expand", borderaxespad=0.)

        fig.set_size_inches(11, 5)
        theme.apply(plt.gcf().axes[0])
        theme.apply(plt.gcf().axes[1])
        # theme.apply(plt.gcf().axes[2])
        theme.apply(plt.gcf().axes[3])
        plt.savefig('representations_exp2_2003.eps', format='eps', dpi=1000)
        plt.show()
    
    
    # impulse 1
    with plt.rc_context():
        plt.rcParams.update(theme.rcParams)
    
        fig, axes, = plt.subplots(1,2,squeeze=True)
        theme.setup_figure(fig)
        t = sim.trange()
        plt.subplots_adjust(wspace=0.1, hspace=0.05)
   
        plot_mc=axes[0]
        plot_mc.set_title("First Module")
        plot_mc.plot(sim.trange(),(mem_1));
        plot_mc.set_ylabel("Cosine similarity")
        plot_mc.set_xticks(np.arange(1.2,1.4,0.05)) # TODO: Check if this is correct
        plot_mc.set_xticklabels(np.arange(0,200,50).tolist())
        plot_mc.set_xlabel('time after onset impulse (ms)')
        plot_mc.set_xlim(1.2,1.35)
        plot_mc.set_ylim(0,0.9)
        colors=["#00c094","#00bfc4","#00b6eb","#06a4ff","#a58aff","#df70f8","#fb61d7", "#c49a00"]
        for i,j in enumerate(plot_mc.lines):
            j.set_color(colors[i])

        plot_mu=axes[1]
        plot_mu.set_title("Second Module")
        plot_mu.plot(sim.trange(),(mem_2));
        plot_mu.set_xticks(np.arange(1.2,1.4,0.05)) # TODO: Check if this is correct
        plot_mu.set_xticklabels(np.arange(0,200,50).tolist())
        plot_mu.set_xlabel('time after onset impulse (ms)')
        plot_mu.set_yticks([])
        plot_mu.set_yticklabels([])
        plot_mu.set_xlim(1.2,1.35)
        plot_mu.set_ylim(0,0.9)
        for i,j in enumerate(plot_mu.lines):
            j.set_color(colors[i])
        plot_mu.legend(["0°","5°","10°","16°","24°","32°","40°", "Impulse"], title="Stimulus", bbox_to_anchor=(0.85, 0.25, .55, 0.8), loc=3,
               ncol=1, mode="expand", borderaxespad=0.)

        fig.set_size_inches(6, 4)

        # theme.apply(plt.gcf().axes[0])
        theme.apply(plt.gcf().axes[1])
        plt.savefig('Impulse1_exp2_2003.eps', format='eps', dpi=1000)
        plt.show()    
  
    # impulse 2
    with plt.rc_context():
        plt.rcParams.update(theme.rcParams)
    
        fig, axes, = plt.subplots(1,2,squeeze=True)
        theme.setup_figure(fig)
        t = sim.trange()
        plt.subplots_adjust(wspace=0.1, hspace=0.05)
   
        plot_mc=axes[0]
        plot_mc.set_title("First Module")
        plot_mc.plot(sim.trange(),(mem_1));
        plot_mc.set_ylabel("Cosine similarity")
        plot_mc.set_xticks(np.arange(3.8,4.0,0.05))
        plot_mc.set_xticklabels(np.arange(0,250,50).tolist())
        plot_mc.set_xlabel('time after onset impulse (ms)')
        plot_mc.set_xlim(3.8,3.95)
        plot_mc.set_ylim(0,0.9)
        colors=["#00c094","#00bfc4","#00b6eb","#06a4ff","#a58aff","#df70f8","#fb61d7", "#c49a00"]
        for i,j in enumerate(plot_mc.lines):
            j.set_color(colors[i])

        plot_mu=axes[1]
        plot_mu.set_title("Second Module")
        plot_mu.plot(sim.trange(),(mem_2));
        plot_mu.set_xticks(np.arange(3.8,4.0,0.05))
        plot_mu.set_xticklabels(np.arange(0,250,50).tolist())
        plot_mu.set_xlabel('time after onset impulse (ms)')
        plot_mu.set_yticks([])
        plot_mu.set_yticklabels([])
        plot_mu.set_xlim(3.8,3.95)
        plot_mu.set_ylim(0,0.9)
        for i,j in enumerate(plot_mu.lines):
            j.set_color(colors[i])
        plot_mu.legend(["0°","5°","10°","16°","24°","32°","40°", "Impulse"], title="Stimulus", bbox_to_anchor=(0.85, 0.25, .55, 0.8), loc=3,
               ncol=1, mode="expand", borderaxespad=0.)

        fig.set_size_inches(6, 4)

        # theme.apply(plt.gcf().axes[0])
        theme.apply(plt.gcf().axes[1])
        plt.savefig('Impulse2_exp2_2003.eps', format='eps', dpi=1000)
        plt.show()      

 
    
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
            out[i]=abs(np.dot(a[i], b)/(np.linalg.norm(a[i])*np.linalg.norm(b)))
    return out
                
                

            
