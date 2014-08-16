# Short-time Fourier Transform

# blackman window
function blackman(n::Integer)
    const a0, a1, a2 = 0.42, 0.5, 0.08
    t = 2*pi/(n-1)
    [a0 - a1*cos(t*k) + a2*cos(t*k*2) for k=0:n-1]
end

# hanning window
function hanning(n::Integer)
    [0.5*(1-cos(2*pi*k/(n-1))) for k=0:n-1]
end

# countframes returns the number of frames that will be processed.
function countframes{T<:Number}(x::Vector{T}, framelen::Int, hopsize::Int)
    int((length(x) - framelen) / hopsize) + 1
end

# splitframes performs overlapping frame split.
function splitframes{T<:Number}(x::Vector{T};
                                framelen::Int=1024, 
                                hopsize::Int=framelen/2)
    N = countframes(x, framelen, hopsize)
    frames = zeros(N, framelen)
    
    for i=1:N
        frames[i,:] = x[(i-1)*hopsize+1:(i-1)*hopsize+framelen]
    end

    return frames
end

# stft performs Short-Time Fourier Transform (STFT).
function stft{T<:Real}(x::Vector{T}; 
                       framelen::Int=1024,
                       hopsize::Int=int(framelen/2),
                       window=hanning(framelen))
    frames = splitframes(x, framelen=framelen, hopsize=hopsize)

    spectrogram = complex(zeros(size(frames)))
    for i=1:size(frames,1)
        spectrogram[i,:] = fft(reshape(frames[i,:], framelen) .* window)
    end

    return spectrogram
end

# istft peforms Inverse Short-Time Fourier Transform
function istft{T<:Complex}(spectrogram::Matrix{T};
                           framelen::Int=1024,
                           hopsize::Int=int(framelen/2),
                           window=hanning(framelen))
    numframes = size(spectrogram, 1)
    expectedLen = framelen + (numframes-1)*hopsize
    reconstructed = zeros(expectedLen)
    windowSum = zeros(expectedLen)
    const windowSquare = window .* window
    
    for i=1:numframes
        s, e = (i-1)*hopsize+1, (i-1)*hopsize+framelen
        r = real(ifft(reshape(spectrogram[i,:], framelen)))
        reconstructed[s:e] += r .* window
        windowSum[s:e] += windowSquare
    end

    for i=1:endof(reconstructed)
        if windowSum[i] > 1.0e-7
            reconstructed[i] /= windowSum[i]
        end
    end

    return reconstructed
end
