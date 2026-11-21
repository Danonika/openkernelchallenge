"""Experimental one-bit signs + FP16 scale + exact exception values.

This is a storage/error research codec, not a competition solution or a speed
claim. NF4 scales remain shared with the reference and are excluded on both
sides of the byte comparison. The variable-record format counts mode bytes,
32-bit offsets, scales, bitmaps and exact exception payloads.
"""
import numpy as np

BLOCK=64
NF4=np.array([-1.,-.6961928009986877,-.5250730514526367,-.39491748809814453,
 -.28444138169288635,-.18477343022823334,-.09105003625154495,0.,
 .07958029955625534,.16093020141124725,.24611230194568634,.33791524171829224,
 .44070982933044434,.5626170039176941,.7229568362236023,1.],dtype=np.float32)

def nibbles(values):
    a=np.asarray(values,dtype=np.uint8).reshape(-1)
    if np.any(a>15):raise ValueError('NF4 code outside 0..15')
    if a.size%2:a=np.concatenate((a,np.zeros(1,dtype=np.uint8)))
    return ((a[::2]<<4)|a[1::2]).tobytes()

def unnibbles(raw,count):
    a=np.frombuffer(raw,dtype=np.uint8)
    out=np.empty(a.size*2,dtype=np.uint8);out[::2]=a>>4;out[1::2]=a&15
    return out[:count]

def reference(values,bits,codebook=NF4):
    a=np.asarray(values)
    if a.ndim!=2 or a.shape[1]!=BLOCK or a.shape[0]==0:raise ValueError('Expected nonempty [blocks,64]')
    if bits==4:
        if np.any(a<0) or np.any(a>15) or np.any(a!=a.astype(np.uint8)):raise ValueError('Invalid NF4 codes')
        return np.asarray(codebook,dtype=np.float32)[a.astype(np.uint8)]
    if bits==16:
        h=a.astype(np.float16)
        if not np.isfinite(h).all():raise ValueError('Finite FP16 reference required')
        return h.astype(np.float32)
    raise ValueError('Reference bits must be 4 or 16')

def encode(values,bits,budget,codebook=NF4):
    """Bound normalized block RMS error; exact exceptions and native fallback."""
    if not np.isfinite(budget) or budget<0:raise ValueError('Invalid error budget')
    original=np.asarray(values);x=reference(original,bits,codebook)
    alpha=np.mean(np.abs(x),axis=1,dtype=np.float32).astype(np.float16)
    signs=x>=0
    approx=np.where(signs,1.,-1.).astype(np.float32)*alpha.astype(np.float32)[:,None]
    error=(x.astype(np.float64)-approx.astype(np.float64))**2
    energy=np.sum(x.astype(np.float64)**2,axis=1)
    order=np.argsort(-error,axis=1,kind='stable')
    sorted_error=np.take_along_axis(error,order,axis=1)
    remaining=np.maximum(error.sum(axis=1)[:,None]-np.concatenate((np.zeros((len(x),1)),np.cumsum(sorted_error,axis=1)),axis=1),0.)
    # All 64 values restored means exactly zero error, regardless of reduction
    # rounding in the difference of two independently accumulated FP64 sums.
    remaining[:,-1]=0.
    counts=np.argmax(remaining<=((budget*budget)*energy)[:,None]+1e-30,axis=1)
    raw=bytearray();offsets=[0];modes=[0,0,0];exception_count=0
    for i in range(len(x)):
        dense=nibbles(original[i]) if bits==4 else x[i].astype('<f2').tobytes()
        k=int(counts[i]);mask=np.zeros(BLOCK,dtype=np.bool_);mask[order[i,:k]]=True
        binary=np.packbits(signs[i],bitorder='little').tobytes()+np.asarray(alpha[i],dtype='<f2').tobytes()
        if k==0:record=b'\x01'+binary;mode=1
        else:
            exceptions=nibbles(original[i][mask]) if bits==4 else x[i][mask].astype('<f2').tobytes()
            record=b'\x02'+binary+np.packbits(mask,bitorder='little').tobytes()+exceptions;mode=2
        if len(record)>=1+len(dense):record=b'\x00'+dense;mode=0;k=0
        raw.extend(record);offsets.append(len(raw));modes[mode]+=1;exception_count+=k
    offsets=np.asarray(offsets,dtype='<u4')
    baseline=x.size*bits//8;candidate=len(raw)+offsets.nbytes
    packet={'bits':bits,'count':len(x),'raw':bytes(raw),'offsets':offsets,
            'codebook':np.asarray(codebook,dtype=np.float32),'native':candidate>=baseline,
            'original':original.copy() if candidate>=baseline else None}
    stats={'block_rms_budget':budget,'blocks':len(x),'reference_bits_per_weight':bits,
           'reference_code_bytes':baseline,'encoded_candidate_bytes':candidate,
           'selected_bytes':min(candidate,baseline),'selected_bits_per_weight':min(candidate,baseline)*8/x.size,
           'selected_native_reference':candidate>=baseline,'candidate_modes':{'native':modes[0],'onebit':modes[1],'onebit_with_exceptions':modes[2]},
           'exception_values':exception_count,'shared_nf4_scale_metadata_included':False}
    return packet,stats

def decode(packet):
    if packet['native']:return reference(packet['original'],packet['bits'],packet['codebook'])
    bits=packet['bits'];out=np.empty((packet['count'],BLOCK),dtype=np.float32)
    for i in range(packet['count']):
        raw=packet['raw'][int(packet['offsets'][i]):int(packet['offsets'][i+1])];mode=raw[0]
        if mode==0:
            out[i]=packet['codebook'][unnibbles(raw[1:],BLOCK)] if bits==4 else np.frombuffer(raw[1:],dtype='<f2')
            continue
        signs=np.unpackbits(np.frombuffer(raw[1:9],dtype=np.uint8),bitorder='little').astype(bool)
        alpha=np.frombuffer(raw[9:11],dtype='<f2')[0]
        out[i]=np.where(signs,1.,-1.).astype(np.float32)*np.float32(alpha)
        if mode==2:
            mask=np.unpackbits(np.frombuffer(raw[11:19],dtype=np.uint8),bitorder='little').astype(bool);k=int(mask.sum())
            out[i,mask]=packet['codebook'][unnibbles(raw[19:],k)] if bits==4 else np.frombuffer(raw[19:],dtype='<f2')
        elif mode!=1:raise ValueError('Unknown record mode')
    return out

def error_summary(actual,expected):
    a=np.asarray(actual,dtype=np.float64);e=np.asarray(expected,dtype=np.float64)
    diff=a-e;den=np.sum(e*e,axis=1);num=np.sum(diff*diff,axis=1)
    rel=np.sqrt(np.divide(num,den,out=np.zeros_like(num),where=den!=0))
    if np.any((den==0)&(num!=0)):raise ValueError('Zero reference changed')
    return {'relative_l2':float(np.sqrt(np.sum(num)/max(np.sum(den),1e-300))),
            'max_block_relative_l2':float(np.max(rel)),'max_abs':float(np.max(np.abs(diff)))}

def analyze(values,bits,codebook=NF4):
    x=reference(values,bits,codebook)
    alpha=np.mean(np.abs(x),axis=1,dtype=np.float32).astype(np.float16).astype(np.float32)
    binary=np.where(x>=0,1.,-1.).astype(np.float32)*alpha[:,None]
    result={'pure_onebit':{'bits_per_weight':1.25,'scale_bits_per_block':16,
            'error':error_summary(binary,x)},'adaptive':[]}
    for budget in (0.,.01,.03,.1,.2,.4,.6):
        packet,stats=encode(values,bits,budget,codebook);decoded=decode(packet)
        stats['error']=error_summary(decoded,x)
        assert stats['error']['max_block_relative_l2']<=budget+1e-6
        if budget==0:assert np.array_equal(decoded,x)
        result['adaptive'].append(stats)
    return result

def self_check():
    assert nibbles(np.arange(16,dtype=np.uint8))==bytes.fromhex('0123456789abcdef')
    assert np.array_equal(unnibbles(bytes.fromhex('0123456789abcdef'),16),np.arange(16))
    rng=np.random.default_rng(20261031)
    cases=[('all_codes',np.tile(np.arange(16,dtype=np.uint8),128).reshape(-1,BLOCK),4),
           ('zero_nf4',np.full((8,BLOCK),7,dtype=np.uint8),4),
           ('binary_exact_nf4',rng.choice([0,15],(32,BLOCK)).astype(np.uint8),4),
           ('half_random',rng.normal(0,.02,(32,BLOCK)).astype(np.float16),16),
           ('half_binary',rng.choice([-1.,1.],(32,BLOCK)).astype(np.float16),16)]
    for _,values,bits in cases:analyze(values,bits)
    return {'cases':[x[0] for x in cases],'budgets_per_case':7,'golden_nibble_layout':True,
            'roundtrip_and_error_bound_checks':35,'scope':'CPU_CODEC_ONLY_NOT_TRAINING_QUALITY'}
