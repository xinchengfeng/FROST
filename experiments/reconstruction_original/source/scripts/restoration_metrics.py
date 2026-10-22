"""RGB PSNR and single-scale SSIM with explicit range/window/cropping."""
import numpy as np
from scipy.ndimage import convolve1d

def ssim_rgb(a,b):
    # Gaussian 11x11, sigma=1.5, population covariance, valid-window mean.
    x=np.arange(-5,6,dtype=np.float64);g=np.exp(-x*x/(2*1.5**2));g/=g.sum()
    def local(v):return convolve1d(convolve1d(v,g,axis=0,mode='reflect'),g,axis=1,mode='reflect')
    ma,mb=local(a),local(b)
    va=np.maximum(local(a*a)-ma*ma,0);vb=np.maximum(local(b*b)-mb*mb,0)
    cov=local(a*b)-ma*mb
    s=((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    return float(s[5:-5,5:-5].mean())

def quality(output,truth,task):
    output=np.clip(np.asarray(output,dtype=np.float64),0,1)
    truth=np.asarray(truth,dtype=np.float64)
    if task=='sr2':output=output[2:-2,2:-2];truth=truth[2:-2,2:-2]
    mse=float(np.square(output-truth).mean())
    return dict(psnr=float(-10*np.log10(mse)) if mse>0 else float('inf'),ssim=ssim_rgb(output,truth),mse=mse)
