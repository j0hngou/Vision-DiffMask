from scipy.signal import lfiltic, lfilter
# careful not to mix between scipy.signal and standard python signal 
# (https://docs.python.org/3/library/signal.html) if your code handles some processes

def ema_mva(array, smoothing=0.99):
    smoothing = 1 - smoothing
    b = [smoothing]
    a = [1, smoothing-1]
    zi = lfiltic(b, a, array[0:1], [0])
    return lfilter(b, a, array, zi=zi)[0]