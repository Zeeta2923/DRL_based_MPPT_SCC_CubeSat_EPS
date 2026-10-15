"""Reproducible demonstration DQN; NumPy, replay buffer, target network, Adam.
Policy for the reduced FSSCC environment. Reward uses simulated PV power.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np, json, time
rng=np.random.default_rng(42)
N=32; actions=np.array([-3,-1,0,1,3]); DIMS=[5,48,48,5]
W=[rng.normal(0,np.sqrt(2/a),(a,b)).astype(np.float32) for a,b in zip(DIMS[:-1],DIMS[1:])]
B=[np.zeros(b,np.float32) for b in DIMS[1:]]
TW=[w.copy() for w in W]; TB=[b.copy() for b in B]
mom=[np.zeros_like(x) for x in W+B]; var=[np.zeros_like(x) for x in W+B]
def net(s,weights=W,bias=B,cache=False):
    hs=[s]
    for j,(w,b) in enumerate(zip(weights,bias)):
        z=hs[-1]@w+b; hs.append(np.maximum(z,0) if j<2 else z)
    return hs if cache else hs[-1]
vc=np.zeros(N); isc=np.zeros(N); shape=np.zeros(N); G=np.zeros(N); T=np.zeros(N); bus=np.zeros(N); C=np.zeros(N); tau=np.zeros(N); ref=np.zeros(N); v=np.zeros(N)
def panel():
    voc=vc*np.maximum(.4,1+.036*np.log(np.maximum(G,1)/1000)-.0025*(T-25))
    ip=isc*G/1000*np.maximum(.1,1+.0005*(T-25))
    return voc,ip

def current(x,voc,ip):return ip*np.maximum(0,1-(np.clip(x/voc,0,1))**shape)
def f_to_v(f,voc,ip):
    R=.15+1/(f*C*1e-6); lo=np.minimum(bus*3,voc); hi=voc.copy()
    for _ in range(20):
        mid=(lo+hi)/2; equation=(mid/9-bus/3)/R-current(mid,voc,ip)
        hi=np.where(equation>0,mid,hi);lo=np.where(equation>0,lo,mid)
    return (lo+hi)/2

def reset(mask):
    k=mask.sum()
    vc[mask]=rng.uniform(22,26,k);isc[mask]=rng.uniform(.6,1.2,k);shape[mask]=rng.uniform(8,16,k)
    G[mask]=rng.uniform(200,1200,k);T[mask]=rng.uniform(0,70,k);bus[mask]=rng.uniform(4.5,5.5,k);C[mask]=rng.uniform(30,80,k);tau[mask]=rng.uniform(.02,.15,k)
    voc,ip=panel();v[mask]=f_to_v(rng.uniform(1000,100000,N),voc,ip)[mask];ref[mask]=v[mask]
reset(np.ones(N,bool)); p=v*current(v,*panel())
s=np.stack([v/vc,current(v,*panel())/isc,np.zeros(N),np.zeros(N),ref/vc],1).astype(np.float32)
capacity=50000; SS=np.zeros((capacity,5),np.float32);NS=SS.copy();AA=np.zeros(capacity,int);RR=np.zeros(capacity,np.float32);DD=RR.copy();ptr=0;count=0;start=time.time();loss=0
for step in range(6500):
    eps=max(.04,1-step/4000)
    a=np.argmax(net(s),1);a=np.where(rng.random(N)<eps,rng.integers(0,5,N),a)
    oldv=v.copy(); oldp=p.copy()
    ref=np.clip(ref+actions[a]*.01*vc,.25*vc,1.1*vc)
    # Ambient steps for policy training; no environment parameters are observations.
    if step%25==20:
        change=rng.random(N)<.3;G[change]=rng.uniform(200,1200,change.sum())
    voc,ip=panel();iref=current(ref,voc,ip)
    req=(ref/9-bus/3)/np.maximum(iref,1e-8)
    f=np.where(ref<=bus*3,100000,np.where(iref<1e-7,1000,1/(np.maximum(req-.15,1e-8)*C*1e-6)))
    f=np.clip(f,1000,100000)
    target=f_to_v(f,voc,ip);v=np.clip(v+(target-v)*(1-np.exp(-.1/tau)),0,voc)
    i=current(v,voc,ip);p=v*i
    vm=voc/(shape+1)**(1/shape);pm=vm*current(vm,voc,ip)
    reward=(p/np.maximum(pm,1e-8))-.002*np.abs(actions[a])
    ns=np.stack([v/vc,i/isc,(p-oldp)/(vc*isc)*5,(v-oldv)/vc*5,ref/vc],1).astype(np.float32)
    done=np.full(N,step%80==79,dtype=float)
    inds=(ptr+np.arange(N))%capacity;SS[inds]=s;NS[inds]=ns;AA[inds]=a;RR[inds]=reward;DD[inds]=done;ptr=(ptr+N)%capacity;count=min(capacity,count+N)
    if step>64:
        ix=rng.integers(0,count,128);h=net(SS[ix],cache=True)
        targetq=RR[ix]+.92*(1-DD[ix])*np.max(net(NS[ix],TW,TB),1)
        delta=h[-1][np.arange(128),AA[ix]]-targetq;loss=float(np.mean(delta**2))
        grad=np.zeros_like(h[-1]);grad[np.arange(128),AA[ix]]=np.clip(delta,-1,1)/128
        gw=[None]*3;gb=[None]*3
        for j in (2,1,0):
            gw[j]=h[j].T@grad;gb[j]=grad.sum(0)
            if j:grad=(grad@W[j].T)*(h[j]>0)
        for j,(param,g) in enumerate(zip(W+B,gw+gb)):
            mom[j]=.9*mom[j]+.1*g;var[j]=.999*var[j]+.001*g*g
            param-=.0007*(mom[j]/(1-.9**(step-63)))/(np.sqrt(var[j]/(1-.999**(step-63)))+1e-8)
    if step%100==0:TW=[w.copy() for w in W];TB=[b.copy() for b in B]
    if done[0]:
        reset(np.ones(N,bool));p=v*current(v,*panel());ns=np.stack([v/vc,current(v,*panel())/isc,np.zeros(N),np.zeros(N),ref/vc],1).astype(np.float32)
    s=ns
    if step%1000==0:print(f'update {step}, transitions {(step+1)*N}, loss {loss:.4f}, seconds {time.time()-start:.1f}',flush=True)
metadata={'seed':42,'transitions':6500*N,'episodes':int(6500*N/80),'layers':DIMS,'gamma':.92,'replayCapacity':capacity,'batchSize':128,'learningRate':.0007,'targetUpdate':100,'trainingStepVFraction':.01,'trainingG':[200,1200],'trainingT':[0,70],'trainingVoc':[22,26],'trainingIsc':[.6,1.2],'trainingShape':[8,16],'trainingC':[30,80],'trainingBus':[4.5,5.5],'trainingRatio':1/3,'trainingTau':[.02,.15],'trainingDt':.1,'trainingNoise':0,'reward':'P/Pmpp - 0.002*abs(action multiplier)','note':'Demonstration policy, no thesis validation or performance guarantee.'}
with open('dist/policy.json','w') as f:json.dump({'weights':[w.tolist() for w in W],'biases':[b.tolist() for b in B],'metadata':metadata},f,separators=(',',':'))
print('saved policy',time.time()-start,flush=True)
